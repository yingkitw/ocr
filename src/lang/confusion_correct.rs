//! Latin confusion-pair post-correction (WebOCR `correct.ts` style).
//!
//! Depth-1 and depth-2 character swaps gated on the language dictionary
//! (O/0, l/1/I, S/s, Z/z, …).

use crate::lang::dictionary::DictionaryHandler;

const MAX_ITER: usize = 8;

/// Apply iterative confusion correction to Latin alphanumeric text.
pub fn correct_latin(text: &str, dict: &DictionaryHandler) -> String {
    if !text.chars().any(|c| c.is_ascii_alphanumeric()) {
        return text.to_string();
    }
    let mut current = text.to_string();
    for _ in 0..MAX_ITER {
        let next = correct_once(&current, dict);
        if next == current {
            return current;
        }
        current = next;
    }
    current
}

fn correct_once(text: &str, dict: &DictionaryHandler) -> String {
    let tokens: Vec<&str> = text.split_inclusive(|c: char| c.is_whitespace()).collect();
    let mut out = String::new();
    for tok in tokens {
        if tok.chars().all(|c| c.is_whitespace()) {
            out.push_str(tok);
        } else {
            out.push_str(&correct_token(tok, dict));
        }
    }
    out
}

fn correct_token(token: &str, dict: &DictionaryHandler) -> String {
    let Some((start, end, alpha)) = alphabetic_span(token) else {
        return token.to_string();
    };
    if alpha.len() < 2 || dict.is_word_valid(alpha) {
        return token.to_string();
    }

    if let Some(rebuilt) = try_depth1(alpha, dict) {
        return rebuild_token(token, start, end, &rebuilt);
    }
    if let Some(rebuilt) = try_depth2(alpha, dict) {
        return rebuild_token(token, start, end, &rebuilt);
    }
    token.to_string()
}

fn alphabetic_span(token: &str) -> Option<(usize, usize, &str)> {
    let start = token.find(|c: char| c.is_ascii_alphanumeric())?;
    let end = token
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_ascii_alphanumeric())
        .map(|(i, c)| i + c.len_utf8())?;
    Some((start, end, &token[start..end]))
}

fn rebuild_token(token: &str, start: usize, end: usize, alpha: &str) -> String {
    format!("{}{}{}", &token[..start], alpha, &token[end..])
}

fn try_depth1(alpha: &str, dict: &DictionaryHandler) -> Option<String> {
    let chars: Vec<char> = alpha.chars().collect();
    for i in 0..chars.len() {
        for &alt in confusions(chars[i]) {
            if alt == chars[i] {
                continue;
            }
            let mut candidate: String = chars
                .iter()
                .enumerate()
                .map(|(j, c)| if j == i { alt } else { *c })
                .collect();
            if dict.is_word_valid(&candidate) {
                return Some(candidate.to_lowercase());
            }
        }
    }
    None
}

fn try_depth2(alpha: &str, dict: &DictionaryHandler) -> Option<String> {
    let chars: Vec<char> = alpha.chars().collect();
    let mut swaps = Vec::new();
    for i in 0..chars.len() {
        for &alt in confusions(chars[i]) {
            if alt != chars[i] {
                swaps.push((i, alt));
            }
        }
    }
    for a in 0..swaps.len() {
        for b in a + 1..swaps.len() {
            let (i, ci) = swaps[a];
            let (j, cj) = swaps[b];
            if i == j {
                continue;
            }
            let (lo_i, lo_c, hi_i, hi_c) = if i < j {
                (i, ci, j, cj)
            } else {
                (j, cj, i, ci)
            };
            let out: String = chars
                .iter()
                .enumerate()
                .map(|(k, c)| {
                    if k == lo_i {
                        lo_c
                    } else if k == hi_i {
                        hi_c
                    } else {
                        *c
                    }
                })
                .collect();
            if dict.is_word_valid(&out) || dict.is_word_valid(&out.to_lowercase()) {
                return Some(out.to_lowercase());
            }
        }
    }
    None
}

fn confusions(ch: char) -> &'static [char] {
    match ch {
        'O' | 'o' => &['O', 'o', '0'],
        '0' => &['0', 'O', 'o'],
        'l' | 'I' | '1' => &['l', 'I', '1'],
        'S' | 's' => &['S', 's', '5'],
        '5' => &['5', 'S', 's'],
        'Z' | 'z' => &['Z', 'z', '2'],
        '2' => &['2', 'Z', 'z'],
        'B' | '8' => &['B', '8'],
        'G' | '6' => &['G', '6'],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth1_fixes_digit_for_letter() {
        let dict = DictionaryHandler::new_for_language("en");
        assert_eq!(correct_latin("w0rd", &dict), "word");
    }

    #[test]
    fn depth2_fixes_mixed_digit_letter_word() {
        let dict = DictionaryHandler::new_for_language("en");
        assert_eq!(correct_latin("d0Zen", &dict), "dozen");
    }

    #[test]
    fn leaves_unknown_tokens_alone() {
        let dict = DictionaryHandler::new_for_language("en");
        let s = "xyzzyplugh";
        assert_eq!(correct_latin(s, &dict), s);
    }

    #[test]
    fn preserves_whitespace_tokens() {
        let dict = DictionaryHandler::new_for_language("en");
        let fixed = correct_latin("  w0rd  ", &dict);
        assert!(fixed.contains("word"));
    }
}
