//! SentencePiece vocabulary: a `vocab.txt` file with `token id` lines.
//! "▁" (U+2581) marks the start of a word and becomes a space.

use anyhow::{Context, Result};
use std::path::Path;

pub struct Vocab {
    tokens: Vec<String>,
    pub blank_idx: usize,
}

impl Vocab {
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("lettura vocab {}", path.display()))?;
        let mut entries: Vec<(usize, String)> = Vec::new();
        for line in content.lines() {
            let (token, id) = line
                .rsplit_once(' ')
                .with_context(|| format!("riga vocab malformata: {line:?}"))?;
            let id: usize = id.parse()?;
            entries.push((id, token.replace('\u{2581}', " ")));
        }
        entries.sort_by_key(|(id, _)| *id);
        let tokens: Vec<String> = entries.into_iter().map(|(_, t)| t).collect();
        let blank_idx = tokens
            .iter()
            .position(|t| t == "<blk>")
            .context("<blk> assente dal vocab")?;
        Ok(Self { tokens, blank_idx })
    }

    pub fn size(&self) -> usize {
        self.tokens.len()
    }

    /// Concatenates tokens and normalizes spaces like onnx-asr:
    /// initial space removed, space kept only preceding word characters.
    pub fn decode(&self, ids: &[usize]) -> String {
        let joined: String = ids.iter().map(|&id| self.tokens[id].as_str()).collect();
        let mut out = String::with_capacity(joined.len());
        let mut chars = joined.chars().peekable();
        while let Some(c) = chars.next() {
            if c.is_whitespace() {
                let keep = !out.is_empty()
                    && chars
                        .peek()
                        .is_some_and(|n| n.is_alphanumeric() || *n == '_');
                if keep {
                    out.push(' ');
                }
            } else {
                out.push(c);
            }
        }
        out
    }
}
