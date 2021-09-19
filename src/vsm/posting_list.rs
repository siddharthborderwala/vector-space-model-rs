use std::collections::HashMap;

pub type DocId = usize;

pub type Frequency = usize;

pub type TFWeight = f64;

pub struct PostingList {
    pub document_frequency: Frequency,
    pub dimension_map: HashMap<DocId, TFWeight>,
}

impl PostingList {
    pub fn new() -> Self {
        PostingList {
            document_frequency: 0,
            dimension_map: HashMap::new(),
        }
    }

    pub fn with_doc_id(doc_id: DocId) -> Self {
        let mut list = PostingList {
            document_frequency: 1,
            dimension_map: HashMap::new(),
        };
        list.dimension_map.insert(doc_id, 1.0);
        list
    }
}
