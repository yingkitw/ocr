use anyhow::Result;

use ocr::api::Ocr;

pub async fn handle_list_languages() -> Result<()> {
    let ocr = Ocr::new()?;
    let base = ocr.get_supported_languages();
    let mut languages: Vec<String> = base.iter().cloned().collect();

    let cjk_codes = ["zh", "ja", "ko"];
    for code in &cjk_codes {
        let entry = format!("{} (CJK)", code);
        if !languages.contains(&entry) {
            languages.push(entry);
        }
    }
    languages.sort();

    println!("Supported languages:");
    for lang in &languages {
        println!("  - {}", lang);
    }
    println!();
    println!("Note: Use --engine to select recognition engine (pattern, lstm, hybrid)");
    println!("      Use --dict-correct to enable dictionary-based post-correction");

    Ok(())
}
