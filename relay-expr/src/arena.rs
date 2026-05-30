//! Arena allocator for plan nodes — integer-indexed, zero-clone manipulation.
//!
//! Inspired by Polars' Arena<AExpr> design: instead of Rc/Arc everywhere,
//! we store nodes in a flat Vec and reference them by integer index.
//! This makes tree rewriting O(1) per node instead of O(depth) with Arc clones.

/// A unique index into the Arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

impl NodeId {
    pub const INVALID: NodeId = NodeId(u32::MAX);

    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A typed arena that stores nodes of type T, referenced by NodeId.
///
/// Nodes are append-only (never removed). The arena can be shared
/// across optimization passes by cloning (cheap — just Vec clone).
#[derive(Debug, Clone)]
pub struct Arena<T> {
    nodes: Vec<T>,
}

impl<T> Arena<T> {
    /// Create an empty arena.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Create an arena with pre-allocated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(cap),
        }
    }

    /// Add a node, returning its NodeId.
    #[inline]
    pub fn add(&mut self, node: T) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    /// Get a reference to a node by ID.
    #[inline]
    pub fn get(&self, id: NodeId) -> &T {
        &self.nodes[id.index()]
    }

    /// Get a mutable reference to a node by ID.
    #[inline]
    pub fn get_mut(&mut self, id: NodeId) -> &mut T {
        &mut self.nodes[id.index()]
    }

    /// Number of nodes in the arena.
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the arena is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Replace a node in-place and return the old value.
    #[inline]
    pub fn replace(&mut self, id: NodeId, node: T) -> T {
        std::mem::replace(self.get_mut(id), node)
    }

    /// Iterate over all (NodeId, &T) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &T)> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (NodeId(i as u32), node))
    }
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_basic() {
        let mut arena = Arena::new();
        let a = arena.add("hello");
        let b = arena.add("world");
        assert_eq!(arena.get(a), &"hello");
        assert_eq!(arena.get(b), &"world");
        assert_eq!(arena.len(), 2);
    }

    #[test]
    fn test_arena_replace() {
        let mut arena = Arena::new();
        let id = arena.add(42);
        let old = arena.replace(id, 99);
        assert_eq!(old, 42);
        assert_eq!(arena.get(id), &99);
    }
}
