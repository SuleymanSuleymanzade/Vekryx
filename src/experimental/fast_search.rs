/// Retrieval only; serialization is handled independently through Serde.
pub trait FastSearch {
    type Vector: ?Sized;

    fn nearest_neighbor(&self, query: &Self::Vector) -> Option<&Self::Vector>;
}
