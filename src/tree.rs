//! Memory-mapped B+tree implementation with thread-safe concurrent access.
//!
//! This module provides the public API ([`MmapBTree`], [`MmapBTreeBuilder`])
//! and the internal state type [`MmapBTreeInner`].  Storage-level concerns
//! live in [`crate::storage`]; typed node views live in [`crate::node`].

use std::io;
use std::marker::PhantomData;
use std::ops::RangeBounds;
use std::path::{Path, PathBuf};
use std::sync::{RwLock, RwLockReadGuard};

use bytemuck::Pod;

use crate::storage::{MmapStore, NodeLayout, PAGE_SIZE};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during B+tree operations.
#[derive(Debug, Clone)]
pub enum BTreeError {
    /// An I/O error from file operations.
    Io(String),
    /// Structural corruption detected in the on-disk tree.
    Corruption(String),
    /// Other operation errors (e.g. poisoned lock).
    Other(String),
}

impl From<io::Error> for BTreeError {
    fn from(err: io::Error) -> Self {
        BTreeError::Io(err.to_string())
    }
}

impl std::fmt::Display for BTreeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BTreeError::Io(msg) => write!(f, "I/O error: {}", msg),
            BTreeError::Corruption(msg) => write!(f, "Tree corruption: {}", msg),
            BTreeError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for BTreeError {}

/// Result type alias for all B+tree operations.
pub type Result<T> = std::result::Result<T, BTreeError>;

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Configuration builder for [`MmapBTree`].
///
/// # Example
///
/// ```no_run
/// use mappedbptree::MmapBTreeBuilder;
///
/// let tree = MmapBTreeBuilder::<i32, u64>::new()
///     .path("tree.db")
///     .build()?;
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
pub struct MmapBTreeBuilder<K, V> {
    path: Option<PathBuf>,
    _phantom: PhantomData<(K, V)>,
}

impl<K, V> MmapBTreeBuilder<K, V>
where
    K: Ord + Pod,
    V: Pod,
{
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self { path: None, _phantom: PhantomData }
    }

    /// Sets the file path for the B+tree storage.
    ///
    /// If the file does not exist it will be created.
    /// If it exists it will be opened and validated.
    pub fn path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Builds and returns the [`MmapBTree`].
    ///
    /// # Errors
    ///
    /// - [`BTreeError::Io`] — file cannot be created or opened.
    /// - [`BTreeError::Corruption`] — existing file has a bad header.
    pub fn build(self) -> Result<MmapBTree<K, V>> {
        let path = self.path.ok_or_else(|| {
            BTreeError::Other("Path must be set via .path()".to_string())
        })?;
        MmapBTree::open(path)
    }
}

impl<K, V> Default for MmapBTreeBuilder<K, V>
where
    K: Ord + Pod,
    V: Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MmapBTree — public API
// ---------------------------------------------------------------------------

/// A memory-mapped, persistent B+tree supporting thread-safe concurrent reads.
///
/// ## Type constraints
///
/// Both `K` and `V` must implement [`bytemuck::Pod`], which guarantees they
/// are plain-data types safe to store as raw bytes in an mmap file.
/// `bytemuck::Pod` implies `Copy + Clone + Sized`.
///
/// `K` additionally requires [`Ord`] for tree ordering.
///
/// ## Thread safety
///
/// An [`RwLock`] wraps the internal state: multiple threads may read
/// concurrently, but writes are exclusive.  The iterators ([`MmapBTreeIter`],
/// [`MmapBTreeRangeIter`]) hold a read lock for their entire lifetime.
///
/// ## Example
///
/// ```no_run
/// use mappedbptree::MmapBTreeBuilder;
///
/// let tree = MmapBTreeBuilder::<i32, u64>::new()
///     .path("data.db")
///     .build()?;
///
/// tree.insert(1, 100)?;
///
/// if let Some(v) = tree.get(&1)? {
///     println!("Found: {}", v);
/// }
///
/// for (k, v) in tree.range(1..10)? {
///     println!("{}: {}", k, v);
/// }
///
/// tree.remove(&1)?;
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
pub struct MmapBTree<K, V> {
    inner: RwLock<MmapBTreeInner<K, V>>,
}

// ---------------------------------------------------------------------------
// MmapBTreeInner — internal state (behind the RwLock)
// ---------------------------------------------------------------------------

/// Internal mutable state of the B+tree.
///
/// Owns the [`MmapStore`] (file + mapping) and the pre-computed
/// [`NodeLayout`] (byte offsets and node capacities).
///
/// The flush-on-drop logic lives in a bounds-free `impl` block so that
/// `Drop` can be implemented without repeating the `K: Ord + Pod, V: Pod`
/// bounds (Rust does not allow `Drop` to add bounds not on the struct).
struct MmapBTreeInner<K, V> {
    store: MmapStore,
    layout: NodeLayout,
    _phantom: PhantomData<(K, V)>,
}

// Flush logic is bounds-free: actual I/O doesn't depend on K or V.
impl<K, V> MmapBTreeInner<K, V> {
    fn flush_impl(&self) -> Result<()> {
        self.store.flush()
    }
}

impl<K, V> Drop for MmapBTreeInner<K, V> {
    fn drop(&mut self) {
        // Best-effort flush on drop; errors are silently ignored.
        let _ = self.flush_impl();
    }
}

// ---------------------------------------------------------------------------
// MmapBTree — construction and lock helpers
// ---------------------------------------------------------------------------

impl<K, V> MmapBTree<K, V>
where
    K: Ord + Pod,
    V: Pod,
{
    /// Opens or creates the B+tree file at `path`.
    fn open(path: PathBuf) -> Result<Self> {
        let store = MmapStore::open(
            &path,
            std::mem::size_of::<K>(),
            std::mem::size_of::<V>(),
        )?;
        let layout = NodeLayout::new(
            PAGE_SIZE,
            std::mem::size_of::<K>(),
            std::mem::align_of::<K>(),
            std::mem::size_of::<V>(),
            std::mem::align_of::<V>(),
        );
        Ok(Self {
            inner: RwLock::new(MmapBTreeInner {
                store,
                layout,
                _phantom: PhantomData,
            }),
        })
    }

    fn write_guard(
        &self,
    ) -> Result<std::sync::RwLockWriteGuard<'_, MmapBTreeInner<K, V>>> {
        self.inner.write().map_err(|_| {
            BTreeError::Other("RwLock poisoned on write".to_string())
        })
    }

    fn read_guard(&self) -> Result<RwLockReadGuard<'_, MmapBTreeInner<K, V>>> {
        self.inner.read().map_err(|_| {
            BTreeError::Other("RwLock poisoned on read".to_string())
        })
    }
}

// ---------------------------------------------------------------------------
// MmapBTree — public methods
// ---------------------------------------------------------------------------

impl<K, V> MmapBTree<K, V>
where
    K: Ord + Pod,
    V: Pod,
{
    /// Inserts a key-value pair, returning the previous value if the key
    /// already existed.
    ///
    /// # Errors
    ///
    /// Returns an error on I/O failure or tree corruption.
    pub fn insert(&self, key: K, value: V) -> Result<Option<V>> {
        self.write_guard()?.insert_impl(key, value)
    }

    /// Returns the value associated with `key`, or `None` if absent.
    ///
    /// # Errors
    ///
    /// Returns an error on I/O failure or tree corruption.
    pub fn get(&self, key: &K) -> Result<Option<V>> {
        self.read_guard()?.get_impl(key)
    }

    /// Returns `true` if `key` is present in the tree.
    ///
    /// # Errors
    ///
    /// Returns an error on I/O failure or tree corruption.
    pub fn contains_key(&self, key: &K) -> Result<bool> {
        self.read_guard()?.contains_key_impl(key)
    }

    /// Removes `key` and returns its associated value, or `None` if absent.
    ///
    /// # Errors
    ///
    /// Returns an error on I/O failure or tree corruption.
    pub fn remove(&self, key: &K) -> Result<Option<V>> {
        self.write_guard()?.remove_impl(key)
    }

    /// Returns the number of key-value pairs in the tree.
    ///
    /// # Errors
    ///
    /// Returns an error on I/O failure or tree corruption.
    pub fn len(&self) -> Result<usize> {
        self.read_guard()?.len_impl()
    }

    /// Returns `true` if the tree contains no elements.
    ///
    /// # Errors
    ///
    /// Returns an error on I/O failure or tree corruption.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Returns an iterator over all key-value pairs in ascending key order.
    ///
    /// The iterator holds a read lock for its entire lifetime — writes are
    /// blocked until it is dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if the lock is poisoned.
    pub fn iter(&self) -> Result<MmapBTreeIter<'_, K, V>> {
        let guard = self.read_guard()?;
        Ok(MmapBTreeIter::new(guard))
    }

    /// Returns an iterator over key-value pairs whose keys fall within `range`.
    ///
    /// The iterator holds a read lock for its entire lifetime.
    ///
    /// # Errors
    ///
    /// Returns an error if the lock is poisoned.
    pub fn range<R: RangeBounds<K>>(
        &self,
        range: R,
    ) -> Result<MmapBTreeRangeIter<'_, K, V>> {
        let guard = self.read_guard()?;
        Ok(MmapBTreeRangeIter::new(guard, range))
    }

    /// Removes all key-value pairs from the tree.
    ///
    /// # Errors
    ///
    /// Returns an error on I/O failure.
    pub fn clear(&self) -> Result<()> {
        self.write_guard()?.clear_impl()
    }

    /// Flushes all pending changes to disk.
    ///
    /// Called automatically on drop (best-effort), but can be called
    /// explicitly to guarantee durability.
    ///
    /// # Errors
    ///
    /// Returns an error on I/O failure.
    pub fn flush(&self) -> Result<()> {
        self.read_guard()?.flush_impl()
    }
}

// ---------------------------------------------------------------------------
// MmapBTreeInner — operation stubs (bounds-constrained)
// ---------------------------------------------------------------------------

impl<K, V> MmapBTreeInner<K, V>
where
    K: Ord + Pod,
    V: Pod,
{
    fn insert_impl(&mut self, _key: K, _value: V) -> Result<Option<V>> {
        // TODO: Walk from root, split full nodes on the way down (proactive
        //       split), insert into the correct leaf.
        Ok(None)
    }

    fn get_impl(&self, _key: &K) -> Result<Option<V>> {
        // TODO: Binary-search each internal node from root to leaf.
        Ok(None)
    }

    fn contains_key_impl(&self, key: &K) -> Result<bool> {
        Ok(self.get_impl(key)?.is_some())
    }

    fn remove_impl(&mut self, _key: &K) -> Result<Option<V>> {
        // TODO: Walk to the correct leaf, remove, rebalance/merge upward.
        Ok(None)
    }

    fn len_impl(&self) -> Result<usize> {
        Ok(self.store.header().num_entries as usize)
    }

    fn clear_impl(&mut self) -> Result<()> {
        // TODO: Truncate the file to one page and reinitialise the header.
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Iterators
// ---------------------------------------------------------------------------

/// Iterator over all key-value pairs in a [`MmapBTree`], in ascending order.
///
/// Holds a read lock for its lifetime — no writes can proceed concurrently.
pub struct MmapBTreeIter<'a, K, V> {
    _guard: RwLockReadGuard<'a, MmapBTreeInner<K, V>>,
    // TODO: leaf-page cursor: (page_index: u64, slot: usize)
}

impl<'a, K: Ord + Pod, V: Pod> MmapBTreeIter<'a, K, V> {
    fn new(guard: RwLockReadGuard<'a, MmapBTreeInner<K, V>>) -> Self {
        // TODO: seek to the leftmost leaf and initialise the cursor.
        Self { _guard: guard }
    }
}

impl<'a, K: Ord + Pod, V: Pod> Iterator for MmapBTreeIter<'a, K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        // TODO: Read current slot; advance cursor along the leaf linked-list.
        None
    }
}

/// Iterator over key-value pairs within a key range of a [`MmapBTree`].
///
/// Holds a read lock for its lifetime.
pub struct MmapBTreeRangeIter<'a, K, V> {
    _guard: RwLockReadGuard<'a, MmapBTreeInner<K, V>>,
    // TODO: stored end bound, leaf-page cursor
}

impl<'a, K: Ord + Pod, V: Pod> MmapBTreeRangeIter<'a, K, V> {
    fn new<R: RangeBounds<K>>(
        guard: RwLockReadGuard<'a, MmapBTreeInner<K, V>>,
        _range: R,
    ) -> Self {
        // TODO: seek to the first leaf key ≥ start bound; store end bound.
        Self { _guard: guard }
    }
}

impl<'a, K: Ord + Pod, V: Pod> Iterator for MmapBTreeRangeIter<'a, K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        // TODO: Read current slot; stop when key exceeds end bound.
        None
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl<K: Ord + Pod, V: Pod> std::fmt::Debug for MmapBTree<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = self.len().unwrap_or(0);
        f.debug_struct("MmapBTree").field("len", &len).finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_requires_path() {
        let result: Result<MmapBTree<i32, u64>> = MmapBTreeBuilder::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tree.db");
        let result: Result<MmapBTree<i32, u64>> =
            MmapBTreeBuilder::new().path(&path).build();
        assert!(result.is_ok());
        assert!(path.exists());
    }

    #[test]
    fn empty_tree_len_is_zero() {
        let dir = tempfile::tempdir().unwrap();
        let tree: MmapBTree<i32, u64> = MmapBTreeBuilder::new()
            .path(dir.path().join("t.db"))
            .build()
            .unwrap();
        assert_eq!(tree.len().unwrap(), 0);
        assert!(tree.is_empty().unwrap());
    }
}
