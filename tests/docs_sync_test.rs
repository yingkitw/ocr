//! Docs-sync tests: assert the root Markdown files describe the code as it is.
//!
//! The 2026-11-05 markdown audit found the same fact carrying four different
//! values across the doc set (test count: 368 / 413 / 423 / 428) and an entire
//! module (`src/webocr/`) missing from every architecture tree. These tests
//! turn the highest-value drift classes into build failures.
//!
//! They are deliberately cheap and structural — no network, no fixtures.

use std::fs;
use std::path::Path;

/// The crate root — `CARGO_MANIFEST_DIR` already points at it.
const ROOT: &str = env!("CARGO_MANIFEST_DIR");

fn read_doc(name: &str) -> String {
    let path = Path::new(ROOT).join(name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {name}: {e}"))
}

/// Every root markdown file that participates in the doc set.
const DOC_FILES: &[&str] = &[
    "README.md",
    "ARCHITECTURE.md",
    "SPEC.md",
    "TODO.md",
    "MEMORY.md",
    "AGENTS.md",
    "TESSERACT_PARITY.md",
];

/// The `src/` submodules that must appear in the architecture trees.
const SRC_MODULES: &[&str] = &[
    "api",
    "cli",
    "core",
    "image",
    "lang",
    "layout",
    "recognition",
    "synthetic",
    "training",
    "webocr",
    "compute",
    "onnx",
    "server",
    "utils",
];

/// Extracts the fenced code block that holds the `src/` module tree.
fn module_tree(text: &str) -> String {
    let mut in_fence = false;
    let mut block = String::new();
    let mut saw_src = false;
    for line in text.lines() {
        if line.trim().starts_with("```") {
            if in_fence {
                if saw_src {
                    return block;
                }
                block.clear();
            } else {
                block.clear();
            }
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            if line.contains("src/") {
                saw_src = true;
            }
            block.push_str(line);
            block.push('\n');
        }
    }
    block
}

/// The set of `src/` submodules a document claims to cover, from either
/// convention: a fenced tree (README) or `### Section (`src/<mod>/`)` headings
/// (ARCHITECTURE.md).
fn documented_modules(text: &str) -> Vec<String> {
    let tree = module_tree(text);
    let mut found = Vec::new();
    if !tree.is_empty() {
        for module in SRC_MODULES {
            if tree.contains(module) {
                found.push((*module).to_string());
            }
        }
        return found;
    }
    // Heading convention: `### Something (`src/<mod>/`)`.
    for line in text.lines() {
        if !line.starts_with("###") {
            continue;
        }
        if let Some(start) = line.find("src/") {
            let rest = &line[start + 4..];
            if let Some(end) = rest.find('/') {
                found.push(rest[..end].to_string());
            }
        }
    }
    found
}

#[test]
fn test_src_modules_are_documented_in_architecture_trees() {
    // Every `src/` submodule must be covered by the README and ARCHITECTURE
    // module documentation. `webocr` was missing from both for its entire life.
    for doc in ["README.md", "ARCHITECTURE.md"] {
        let text = read_doc(doc);
        let found = documented_modules(&text);
        assert!(
            !found.is_empty(),
            "{doc} documents no `src/` modules at all"
        );
        for module in SRC_MODULES {
            // The server module is feature-gated; ARCHITECTURE.md covers it
            // in prose rather than with its own section.
            if *module == "server" && doc == "ARCHITECTURE.md" {
                continue;
            }
            assert!(
                found.iter().any(|m| m == module),
                "{doc} does not document the `src/{module}` module — the module list is stale"
            );
        }
    }
}

#[test]
fn test_webocr_submodules_are_documented() {
    // The WebOCR module is the most feature-dense part of the crate; its
    // per-file table in ARCHITECTURE.md must cover every file.
    let text = read_doc("ARCHITECTURE.md");
    let webocr_dir = Path::new(ROOT).join("src/webocr");
    let mut missing = Vec::new();
    for entry in fs::read_dir(&webocr_dir).expect("src/webocr must exist") {
        let entry = entry.expect("readable dir entry");
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.ends_with(".rs") {
            continue;
        }
        if !text.contains(&name) {
            missing.push(name);
        }
    }
    assert!(
        missing.is_empty(),
        "ARCHITECTURE.md is missing WebOCR module(s): {missing:?}"
    );
}

#[test]
fn test_no_phantom_cli_subcommands() {
    // A doc may only mention a CLI subcommand that actually exists.
    // `ocr onnx import` was documented for months but never built.
    let known = [
        "extract",
        "batch",
        "layout",
        "list-languages",
        "check",
        "info",
        "validate",
        "train",
        "benchmark",
        "makebox",
        "serve",
    ];
    for doc in DOC_FILES {
        let text = read_doc(doc);
        let mut in_fence = false;
        // Only look at inline-code spans and fenced code blocks — prose like
        // "OCR engine and command-line tool" is not a CLI invocation.
        for (line_no, line) in text.lines().enumerate() {
            if line.trim().starts_with("```") {
                in_fence = !in_fence;
                continue;
            }
            // Skip lines that *describe* a phantom command rather than
            // invoking one — e.g. "said ONNX import is `ocr onnx import` —
            // there is no `onnx` subcommand". Those are meta-discussion.
            let lower = line.to_lowercase();
            if lower.contains("there is no")
                || lower.contains("no such")
                || lower.contains("not a command")
                || lower.contains("never built")
                || lower.contains("does not exist")
                || lower.contains("phantom")
            {
                continue;
            }
            for occurrence in find_cli_invocations_in_line(line, in_fence) {
                let word: String = occurrence
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '-')
                    .collect();
                if word.is_empty() || word.starts_with('-') {
                    // `ocr --help` / `ocr -V` are not subcommands.
                    continue;
                }
                assert!(
                    known.contains(&word.as_str()),
                    "{doc}:{} mentions unknown CLI subcommand `ocr {word}`",
                    line_no + 1
                );
            }
        }
    }
}

/// Yields the word that follows each `ocr ` token inside inline code spans
/// (`` `ocr ...` ``) or fenced code blocks — i.e. real command invocations.
fn find_cli_invocations_in_line(line: &str, in_fence: bool) -> Vec<String> {
    let mut found = Vec::new();
    // Inline code spans on this line.
    let mut spans: Vec<&str> = Vec::new();
    let mut rest = line;
    while let Some(start) = rest.find('`') {
        let after = &rest[start + 1..];
        match after.find('`') {
            Some(end) => {
                spans.push(&after[..end]);
                rest = &after[end + 1..];
            }
            None => break,
        }
    }
    if in_fence {
        // Inside a fenced block the whole line is a command context.
        spans.push(line);
    } else if spans.is_empty() {
        return found;
    }
    for span in spans {
        let mut tokens = span.split_whitespace();
        while let Some(tok) = tokens.next() {
            if tok == "ocr" {
                if let Some(next) = tokens.next() {
                    found.push(next.to_string());
                }
            }
        }
    }
    found
}

#[test]
fn test_web_api_endpoint_matches_server_routes() {
    // The README documented `POST /ocr`; the server actually mounts
    // `POST /recognize`.
    let server = fs::read_to_string(Path::new(ROOT).join("src/server/mod.rs"))
        .expect("src/server/mod.rs must exist");
    let mut routes = Vec::new();
    for line in server.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix(".route(") {
            if let Some(end) = rest.find('"') {
                let after = &rest[end + 1..];
                if let Some(end2) = after.find('"') {
                    routes.push(after[..end2].to_string());
                }
            }
        }
    }
    assert!(
        routes.iter().any(|r| r == "/recognize"),
        "expected a /recognize route in src/server/mod.rs, got {routes:?}"
    );

    let readme = read_doc("README.md");
    assert!(
        !readme.contains("/ocr\""),
        "README.md still points at the non-existent `/ocr` endpoint — it is `/recognize`"
    );
    assert!(
        readme.contains("/recognize"),
        "README.md must document the `/recognize` endpoint"
    );
}

#[test]
fn test_test_count_is_consistent_across_docs() {
    // The 2026-11-05 audit found four different test counts in four files.
    // The current count is asserted in README.md, SPEC.md, TODO.md,
    // MEMORY.md and AGENTS.md — they must agree.
    let readme = read_doc("README.md");
    let expected = readme
        .lines()
        .find_map(|l| {
            let l = l.trim();
            let rest = l.strip_prefix("- [x] ")?;
            let num = rest.strip_suffix(" tests passing")?;
            Some(num.trim().to_string())
        })
        .expect("README.md must state the passing-test count in its roadmap");

    for doc in ["SPEC.md", "TODO.md", "MEMORY.md", "AGENTS.md"] {
        let text = read_doc(doc);
        assert!(
            text.contains(&expected),
            "{doc} does not mention the current test count ({expected})"
        );
    }

    // The README badge must agree with the roadmap number — they are two
    // separate literals and drifted apart before.
    let readme = read_doc("README.md");
    let badge = readme
        .lines()
        .find_map(|l| l.trim().strip_prefix("[![Tests](https://img.shields.io/badge/tests-"))
        .and_then(|rest| rest.split("_passing").next())
        .map(|s| s.trim().to_string())
        .expect("README.md must carry a tests-<N>_passing badge");
    assert_eq!(
        badge, expected,
        "README.md badge says {badge} but its roadmap says {expected}"
    );
}
