use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{stdin, Write};
use tokenizers::tokenizer::Tokenizer;
use vsm::indexing::build_index;
use vsm::posting_list::{DocId, Frequency};

fn get_input() -> String {
    println!("Enter your query:");
    let mut query = String::new();
    stdin().read_line(&mut query).unwrap();
    query.trim().to_string()
}

fn get_query_map(
    raw_tokens: Vec<String>,
    stop_words: &HashSet<String>,
    punctuations: &HashSet<String>,
) -> HashMap<String, (Frequency, f64, f64)> {
    let tokens = raw_tokens
        .iter()
        .map(|token| token.to_lowercase())
        .filter(|token| !stop_words.contains(token) && !punctuations.contains(token));
    // term-frequency, idf, weight
    let mut query_map = HashMap::<String, (Frequency, f64, f64)>::new();
    for token in tokens {
        query_map
            .entry(token)
            .and_modify(|entry| entry.0 = entry.0 + 1)
            .or_insert((1, 0.0, 0.0));
    }
    query_map
}

fn main() {
    let index_data = build_index();
    let normalized_index = index_data.normalized_index;
    let mut f = File::create("data/indexes/normalized.txt").unwrap();
    f.write_fmt(format_args!("{:#?}", normalized_index))
        .unwrap();
    let stop_words = index_data.stop_words;
    let punctuations = index_data.punctuations;
    let total_tokens_count = index_data.total_tokens_count;
    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
    println!("Welcome to VSM based search engine");
    loop {
        let raw_tokens = tokenizer
            .encode(get_input(), false)
            .unwrap()
            .get_tokens()
            .to_vec();
        let mut query_map = get_query_map(raw_tokens, &stop_words, &punctuations);
        // calculate inverse document frequencies
        query_map.iter_mut().for_each(|(key, value)| {
            // doc frequency
            let df: usize = match normalized_index.get(key) {
                Some(v) => v.document_frequency,
                None => 0,
            };
            // idf
            value.1 = (total_tokens_count as f64 / df as f64).log10();
            // weight
            value.2 = (value.0 as f64) * value.1;
        });
        // calculate term frequencies
        // calculate dot product for all documents and select top 10 at max
        let mut dot_product_map = HashMap::<DocId, f64>::new();
        for (key, (_, _, weight)) in query_map {
            // 1. get posting-list from normalized-index
            match normalized_index.get(&key) {
                Some(list) => {
                    // 2. for each doc-id in the posting-list, insert a key in the
                    // dot-product-map and multiply the weight (query-term) with cosine
                    // similarity (doc-term)
                    list.dimension_map.iter().for_each(|(k, v)| {
                        dot_product_map
                            .entry(*k)
                            .and_modify(|sum| *sum += v * weight)
                            .or_insert(v * weight);
                    });
                }
                None => continue,
            }
        }

        let mut results = dot_product_map.into_iter().collect::<Vec<_>>();
        results.sort_by(|(a_id, va), (b_id, vb)| {
            if va > vb {
                Ordering::Greater
            } else if va < vb {
                Ordering::Less
            } else if a_id > b_id {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        });

        if results.len() < 10 {
            for v in results {
                println!("Doc {:2} - Relevance {:.4}", v.0, v.1);
            }
        } else {
            for k in results.get(0..=10) {
                for v in k {
                    println!("Doc {:2} - Relevance {:.4}", v.0, v.1);
                }
            }
        }
    }
}
