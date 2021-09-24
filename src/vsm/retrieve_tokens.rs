use std::collections::HashSet;
use std::fs::{read_dir, File};
use std::io::prelude::*;
use std::io::BufReader;

use tokenizers::tokenizer::{Result, Tokenizer};

pub fn tokenize_document(tokenizer: &Tokenizer, document_path: &str) -> Result<Vec<String>> {
    let file = File::open(document_path)?;
    let mut buf_reader = BufReader::new(file);
    let mut contents = String::new();
    buf_reader.read_to_string(&mut contents)?;
    let tokens = tokenizer.encode(contents, false)?;
    Ok(tokens
        .get_tokens()
        .iter()
        .map(|t| t.to_lowercase())
        .collect::<Vec<String>>())
}

pub fn get_document_names(dir_path: &str) -> Result<Vec<String>> {
    let mut doc_paths = read_dir(dir_path)?
        .map(|path| path.unwrap().file_name().into_string().unwrap())
        .collect::<Vec<String>>();
    doc_paths.sort_by(|a, b| {
        let n_a: u32 = a.split(".").next().unwrap().parse::<u32>().unwrap();
        let n_b: u32 = b.split(".").next().unwrap().parse::<u32>().unwrap();
        n_a.cmp(&n_b)
    });
    Ok(doc_paths)
}

fn get_words(path: &'static str) -> Result<HashSet<String>> {
    let file = File::open(path)?;
    let mut buf_reader = BufReader::new(file);
    let mut contents = String::new();
    buf_reader.read_to_string(&mut contents)?;
    Ok(contents
        .lines()
        .map(|w| w.to_string())
        .collect::<HashSet<String>>())
}

pub fn get_stop_words() -> Result<HashSet<String>> {
    get_words("data/stopwords.txt")
}

pub fn get_punctuations() -> Result<HashSet<String>> {
    get_words("data/punctuations.txt")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lists_file_names() {
        let paths = get_document_names("data/documents");
        println!("{:?}", paths);
    }

    #[test]
    fn test_tokenize_document() {
        let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
        tokenize_document(&tokenizer, "data/documents/1.txt").expect("Could not tokenize");
    }
}
