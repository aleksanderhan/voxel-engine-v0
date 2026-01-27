use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct ChunkEdits {
    // local linear index -> material
    pub map: HashMap<u32, u32>,
    pub version: u64,
}
