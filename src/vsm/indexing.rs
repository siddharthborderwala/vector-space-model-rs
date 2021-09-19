use std::collections::{HashMap, HashSet};

use super::{
    posting_list::{DocId, Frequency, PostingList},
    retrieve_tokens::{get_document_names, get_stop_words, tokenize_document},
};
use tokenizers::Tokenizer;

pub struct IndexData {
    pub normalized_index: HashMap<String, PostingList>,
    pub document_lengths: HashMap<DocId, Frequency>,
    pub stop_words: HashSet<String>,
    pub total_tokens_count: Frequency,
}

pub fn build_index() -> IndexData {
    let stop_words = get_stop_words().unwrap();
    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
    let mut index = HashMap::<String, PostingList>::new();
    let mut document_lengths = HashMap::<DocId, Frequency>::new();
    let parent = "data/documents".to_string();
    let mut total_tokens_count: Frequency = 0;
    for name in get_document_names(&parent).unwrap() {
        let mut path = parent.clone();
        path.push('/');
        path.push_str(&name);
        let doc_id = name.split(".").next().unwrap().parse::<DocId>().unwrap();
        let tokens = tokenize_document(&tokenizer, &path).unwrap();
        let final_tokens = tokens
            .iter()
            .filter(|&token| {
                if stop_words.contains(token) {
                    false
                } else {
                    true
                }
            })
            .map(|a| a.to_string());
        let mut counter: Frequency = 0;
        for token in final_tokens {
            counter += 1;
            if let Some(list) = index.get_mut(&token) {
                list.document_frequency += 1;
                list.dimension_map
                    .entry(doc_id)
                    .and_modify(|f| *f += 1.0)
                    .or_insert(1.0);
            } else {
                index.insert(token, PostingList::with_doc_id(doc_id));
            }
        }
        document_lengths.insert(doc_id, counter);
        total_tokens_count += counter;
    }
    index.iter_mut().for_each(|(_, value)| {
        let sum_of_squares: f64 = value.dimension_map.values().fold(0.0, |acc, x| acc + x * x);
        let magnitude = sum_of_squares.sqrt();
        for v in value.dimension_map.values_mut() {
            *v = *v / magnitude;
        }
    });
    IndexData {
        normalized_index: index,
        document_lengths,
        stop_words,
        total_tokens_count,
    }
}
