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

use crate::node::{InternalView, InternalViewMut, LeafView, LeafViewMut};
use crate::storage::{
    MmapStore, NodeHeader, NodeLayout, NODE_HEADER_SIZE, NODE_KIND_LEAF,
    NULL_PAGE, PAGE_SIZE,
};

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
/// concurrently, but writes are exclusive.
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
struct MmapBTreeInner<K, V> {
    store: MmapStore,
    layout: NodeLayout,
    _phantom: PhantomData<(K, V)>,
}

/// Flush logic is in a bounds-free impl so that `Drop` can be implemented
/// without re-stating `K: Ord + Pod, V: Pod` (Rust forbids `Drop` bounds
/// that aren't on the struct definition itself).
impl<K, V> MmapBTreeInner<K, V> {
    fn flush_impl(&self) -> Result<()> {
        self.store.flush()
    }
}

impl<K, V> Drop for MmapBTreeInner<K, V> {
    fn drop(&mut self) {
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
    /// Inserts `key` → `value`.  Returns the previous value if the key existed.
    pub fn insert(&self, key: K, value: V) -> Result<Option<V>> {
        self.write_guard()?.insert_impl(key, value)
    }

    /// Returns the value for `key`, or `None` if absent.
    pub fn get(&self, key: &K) -> Result<Option<V>> {
        self.read_guard()?.get_impl(key)
    }

    /// Returns `true` if `key` is present.
    pub fn contains_key(&self, key: &K) -> Result<bool> {
        self.read_guard()?.contains_key_impl(key)
    }

    /// Removes `key` and returns its value, or `None` if absent.
    pub fn remove(&self, key: &K) -> Result<Option<V>> {
        self.write_guard()?.remove_impl(key)
    }

    /// Returns the number of key-value pairs.
    pub fn len(&self) -> Result<usize> {
        self.read_guard()?.len_impl()
    }

    /// Returns `true` if the tree contains no elements.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Returns an iterator over all key-value pairs in ascending key order.
    ///
    /// Holds a read lock for its entire lifetime — writes are blocked until
    /// the iterator is dropped.
    pub fn iter(&self) -> Result<MmapBTreeIter<'_, K, V>> {
        let guard = self.read_guard()?;
        Ok(MmapBTreeIter::new(guard))
    }

    /// Returns an iterator over key-value pairs whose keys fall within `range`.
    ///
    /// Holds a read lock for its entire lifetime.
    pub fn range<R: RangeBounds<K>>(
        &self,
        range: R,
    ) -> Result<MmapBTreeRangeIter<'_, K, V>> {
        let guard = self.read_guard()?;
        Ok(MmapBTreeRangeIter::new(guard, range))
    }

    /// Removes all key-value pairs and frees all node pages.
    pub fn clear(&self) -> Result<()> {
        self.write_guard()?.clear_impl()
    }

    /// Flushes all pending writes to disk.
    ///
    /// Called automatically on drop (best-effort), but may be called
    /// explicitly to guarantee durability.
    pub fn flush(&self) -> Result<()> {
        self.read_guard()?.flush_impl()
    }
}

// ---------------------------------------------------------------------------
// MmapBTreeInner — algorithm helpers (bounds-constrained)
// ---------------------------------------------------------------------------

impl<K, V> MmapBTreeInner<K, V>
where
    K: Ord + Pod,
    V: Pod,
{
    // -----------------------------------------------------------------------
    // Small utilities
    // -----------------------------------------------------------------------

    /// Returns the kind byte of the node at `page_idx`.
    #[inline]
    fn node_kind(&self, page_idx: u64) -> u8 {
        self.store.page(page_idx)[0]
    }

    /// Returns true if the node at `page_idx` is at maximum capacity.
    #[inline]
    fn node_is_full(&self, page_idx: u64) -> bool {
        let page = self.store.page(page_idx);
        let hdr = bytemuck::from_bytes::<NodeHeader>(&page[..NODE_HEADER_SIZE]);
        let n = hdr.num_keys as usize;
        match hdr.node_kind {
            NODE_KIND_LEAF => n == self.layout.leaf_capacity,
            _ => n == self.layout.internal_capacity,
        }
    }

    /// Returns the child slot to follow for `key` in an internal node,
    /// using `partition_point` on the live separator keys.
    ///
    /// Invariant: `separator[i]` is the smallest key in `children[i+1]`, so
    /// the correct child for `key` is `children[partition_point(sep <= key)]`.
    #[inline]
    fn internal_child_slot(&self, page_idx: u64, key: &K) -> (usize, u64) {
        let page = self.store.page(page_idx);
        let view = InternalView::<K>::new(page, &self.layout);
        let slot = view.keys().partition_point(|k| k <= key);
        (slot, view.children()[slot])
    }

    // -----------------------------------------------------------------------
    // get_impl
    // -----------------------------------------------------------------------

    fn get_impl(&self, key: &K) -> Result<Option<V>> {
        let root = self.store.header().root_page;
        if root == NULL_PAGE {
            return Ok(None);
        }

        let mut current = root;
        loop {
            match self.node_kind(current) {
                NODE_KIND_LEAF => {
                    let page = self.store.page(current);
                    let view = LeafView::<K, V>::new(page, &self.layout);
                    return Ok(match view.keys().binary_search(key) {
                        Ok(i) => Some(view.values()[i]),
                        Err(_) => None,
                    });
                }
                _ => {
                    let (_slot, child) = self.internal_child_slot(current, key);
                    current = child;
                }
            }
        }
    }

    fn contains_key_impl(&self, key: &K) -> Result<bool> {
        Ok(self.get_impl(key)?.is_some())
    }

    // -----------------------------------------------------------------------
    // len_impl
    // -----------------------------------------------------------------------

    fn len_impl(&self) -> Result<usize> {
        Ok(self.store.header().num_entries as usize)
    }

    // -----------------------------------------------------------------------
    // insert_impl — proactive top-down split strategy
    // -----------------------------------------------------------------------
    //
    // Proactive splits guarantee that every node we descend through already
    // has room, so we never need to propagate splits upward.
    //
    //   1. Empty tree  →  create a leaf root and insert.
    //   2. Root is full  →  split_root (root page stays at same index but
    //                        becomes an internal node with two leaf/internal
    //                        children).
    //   3. Descend.  Before following each child pointer, if the child is
    //      full, split it in-place (parent has room because of step 2 /
    //      prior iterations).
    //   4. Current node is a leaf  →  leaf_insert.

    fn insert_impl(&mut self, key: K, value: V) -> Result<Option<V>> {
        // ── 1. Empty tree ─────────────────────────────────────────────────
        if self.store.header().root_page == NULL_PAGE {
            return self.insert_first(key, value);
        }

        // ── 2. Full root ───────────────────────────────────────────────────
        let root = self.store.header().root_page;
        if self.node_is_full(root) {
            self.split_root()?;
        }

        // ── 3. Descend with proactive splits ──────────────────────────────
        let mut current = self.store.header().root_page;
        loop {
            if self.node_kind(current) == NODE_KIND_LEAF {
                break;
            }

            // Find the child we would normally descend into.
            let (child_slot, child_idx) = self.internal_child_slot(current, &key);

            if self.node_is_full(child_idx) {
                // Split the child before descending — parent has room.
                self.split_child(current, child_slot)?;
                // Re-derive the slot: the split inserted a new separator key
                // into `current`, potentially shifting our target child.
                let new_child = {
                    let page = self.store.page(current);
                    let view = InternalView::<K>::new(page, &self.layout);
                    let slot = view.keys().partition_point(|k| k <= &key);
                    view.children()[slot]
                };
                current = new_child;
            } else {
                current = child_idx;
            }
        }

        // ── 4. Insert into the leaf ────────────────────────────────────────
        self.leaf_insert(current, key, value)
    }

    /// Insert the very first key into an empty tree.
    fn insert_first(&mut self, key: K, value: V) -> Result<Option<V>> {
        let leaf_idx = self.store.alloc_page()?;
        {
            let page = self.store.page_mut(leaf_idx);
            let mut view = LeafViewMut::<K, V>::new(page, &self.layout);
            view.init();
            view.keys_mut()[0] = key;
            view.values_mut()[0] = value;
            view.set_num_keys(1);
        }
        self.store.header_mut().root_page = leaf_idx;
        self.store.header_mut().num_entries = 1;
        Ok(None)
    }

    // -----------------------------------------------------------------------
    // split_root
    // -----------------------------------------------------------------------
    //
    // The root page stays at its current index but is reinitialised as an
    // internal node with one separator key and two children.  Both children
    // are freshly allocated pages containing the left and right halves of
    // the former root.

    fn split_root(&mut self) -> Result<()> {
        let root_idx = self.store.header().root_page;

        match self.node_kind(root_idx) {
            NODE_KIND_LEAF => self.split_root_leaf(root_idx),
            _ => self.split_root_internal(root_idx),
        }
    }

    fn split_root_leaf(&mut self, root_idx: u64) -> Result<()> {
        let lc = self.layout.leaf_capacity;
        let mid = lc / 2;

        // Copy everything out of the root page.
        let (all_keys, all_values, old_next) = {
            let page = self.store.page(root_idx);
            let view = LeafView::<K, V>::new(page, &self.layout);
            // Root is full: num_keys == lc.
            let keys: Vec<K> = view.keys().to_vec();
            let vals: Vec<V> = view.values().to_vec();
            (keys, vals, view.next_leaf())
        };

        let left_idx = self.store.alloc_page()?;
        let right_idx = self.store.alloc_page()?;

        // Left leaf: keys/values [0..mid].
        {
            let page = self.store.page_mut(left_idx);
            let mut view = LeafViewMut::<K, V>::new(page, &self.layout);
            view.init();
            view.keys_mut()[..mid].copy_from_slice(&all_keys[..mid]);
            view.values_mut()[..mid].copy_from_slice(&all_values[..mid]);
            view.set_num_keys(mid);
            view.set_next_leaf(right_idx);
        }

        // Right leaf: keys/values [mid..lc].
        let right_n = lc - mid;
        let separator = all_keys[mid]; // smallest key in the right child
        {
            let page = self.store.page_mut(right_idx);
            let mut view = LeafViewMut::<K, V>::new(page, &self.layout);
            view.init();
            view.keys_mut()[..right_n].copy_from_slice(&all_keys[mid..]);
            view.values_mut()[..right_n].copy_from_slice(&all_values[mid..]);
            view.set_num_keys(right_n);
            view.set_next_leaf(old_next);
        }

        // Reinitialise the root page as a 1-key internal node.
        {
            let page = self.store.page_mut(root_idx);
            let mut view = InternalViewMut::<K>::new(page, &self.layout);
            view.init();
            view.keys_mut()[0] = separator;
            view.children_mut()[0] = left_idx;
            view.children_mut()[1] = right_idx;
            view.set_num_keys(1);
        }

        Ok(())
    }

    fn split_root_internal(&mut self, root_idx: u64) -> Result<()> {
        let ic = self.layout.internal_capacity;
        let mid = ic / 2;

        // Copy everything out.
        let (all_keys, all_children) = {
            let page = self.store.page(root_idx);
            let view = InternalView::<K>::new(page, &self.layout);
            // Root is full: num_keys == ic, num_children == ic+1.
            (view.keys().to_vec(), view.children().to_vec())
        };

        let left_idx = self.store.alloc_page()?;
        let right_idx = self.store.alloc_page()?;

        // Left internal: keys [0..mid], children [0..mid+1].
        {
            let page = self.store.page_mut(left_idx);
            let mut view = InternalViewMut::<K>::new(page, &self.layout);
            view.init();
            view.keys_mut()[..mid].copy_from_slice(&all_keys[..mid]);
            view.children_mut()[..mid + 1].copy_from_slice(&all_children[..mid + 1]);
            view.set_num_keys(mid);
        }

        // Separator key is pushed up (not duplicated).
        let separator = all_keys[mid];

        // Right internal: keys [mid+1..ic], children [mid+1..ic+1].
        let right_n = ic - mid - 1;
        {
            let page = self.store.page_mut(right_idx);
            let mut view = InternalViewMut::<K>::new(page, &self.layout);
            view.init();
            view.keys_mut()[..right_n].copy_from_slice(&all_keys[mid + 1..]);
            view.children_mut()[..right_n + 1].copy_from_slice(&all_children[mid + 1..]);
            view.set_num_keys(right_n);
        }

        // Reinitialise root as a 1-key internal node.
        {
            let page = self.store.page_mut(root_idx);
            let mut view = InternalViewMut::<K>::new(page, &self.layout);
            view.init();
            view.keys_mut()[0] = separator;
            view.children_mut()[0] = left_idx;
            view.children_mut()[1] = right_idx;
            view.set_num_keys(1);
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // split_child — splits a full child of `parent_idx` at `child_slot`
    // -----------------------------------------------------------------------
    //
    // After the split:
    //   parent.children[child_slot]   = left half  (in-place modification)
    //   parent.children[child_slot+1] = right half (newly allocated)
    //   parent.keys[child_slot]       = separator  (inserted, shifting right)
    //
    // Precondition: parent has room (num_keys < internal_capacity).

    fn split_child(&mut self, parent_idx: u64, child_slot: usize) -> Result<()> {
        let child_idx = {
            let page = self.store.page(parent_idx);
            InternalView::<K>::new(page, &self.layout).children()[child_slot]
        };

        let (separator, right_idx) = match self.node_kind(child_idx) {
            NODE_KIND_LEAF => self.split_leaf_child(child_idx)?,
            _ => self.split_internal_child(child_idx)?,
        };

        // Insert separator and right child pointer into parent.
        self.insert_into_internal(parent_idx, child_slot, separator, right_idx);
        Ok(())
    }

    /// Splits the full leaf at `leaf_idx` in-place.
    ///
    /// Keeps the left half in `leaf_idx`, allocates a new page for the right
    /// half.  Returns `(separator_key, right_page_idx)`.
    fn split_leaf_child(&mut self, leaf_idx: u64) -> Result<(K, u64)> {
        let lc = self.layout.leaf_capacity;
        let mid = lc / 2;

        let (all_keys, all_values, old_next) = {
            let page = self.store.page(leaf_idx);
            let view = LeafView::<K, V>::new(page, &self.layout);
            (view.keys().to_vec(), view.values().to_vec(), view.next_leaf())
        };

        let right_idx = self.store.alloc_page()?;
        let right_n = lc - mid;
        let separator = all_keys[mid];

        // Write right leaf.
        {
            let page = self.store.page_mut(right_idx);
            let mut view = LeafViewMut::<K, V>::new(page, &self.layout);
            view.init();
            view.keys_mut()[..right_n].copy_from_slice(&all_keys[mid..]);
            view.values_mut()[..right_n].copy_from_slice(&all_values[mid..]);
            view.set_num_keys(right_n);
            view.set_next_leaf(old_next);
        }

        // Truncate the left leaf and relink.
        {
            let page = self.store.page_mut(leaf_idx);
            let mut view = LeafViewMut::<K, V>::new(page, &self.layout);
            view.set_num_keys(mid);
            view.set_next_leaf(right_idx);
        }

        Ok((separator, right_idx))
    }

    /// Splits the full internal node at `node_idx` in-place.
    ///
    /// Keeps the left half in `node_idx`, allocates a new page for the right
    /// half.  The middle key is **pushed up** (not retained in either child).
    /// Returns `(pushed_up_key, right_page_idx)`.
    fn split_internal_child(&mut self, node_idx: u64) -> Result<(K, u64)> {
        let ic = self.layout.internal_capacity;
        let mid = ic / 2;

        let (all_keys, all_children) = {
            let page = self.store.page(node_idx);
            let view = InternalView::<K>::new(page, &self.layout);
            (view.keys().to_vec(), view.children().to_vec())
        };

        let right_idx = self.store.alloc_page()?;
        let separator = all_keys[mid];
        let right_n = ic - mid - 1;

        // Write right internal node.
        {
            let page = self.store.page_mut(right_idx);
            let mut view = InternalViewMut::<K>::new(page, &self.layout);
            view.init();
            view.keys_mut()[..right_n].copy_from_slice(&all_keys[mid + 1..]);
            view.children_mut()[..right_n + 1].copy_from_slice(&all_children[mid + 1..]);
            view.set_num_keys(right_n);
        }

        // Truncate the left internal node (drop keys[mid..] and children[mid+1..]).
        {
            let page = self.store.page_mut(node_idx);
            let mut view = InternalViewMut::<K>::new(page, &self.layout);
            view.set_num_keys(mid);
        }

        Ok((separator, right_idx))
    }

    // -----------------------------------------------------------------------
    // insert_into_internal
    // -----------------------------------------------------------------------
    //
    // Inserts `key` at position `slot` and `right_child` at `slot+1` in the
    // internal node at `node_idx`, shifting existing entries right.
    //
    // Precondition: num_keys < internal_capacity.

    fn insert_into_internal(
        &mut self,
        node_idx: u64,
        slot: usize,
        key: K,
        right_child: u64,
    ) {
        let page = self.store.page_mut(node_idx);
        let mut view = InternalViewMut::<K>::new(page, &self.layout);
        let n = view.num_keys();
        debug_assert!(n < self.layout.internal_capacity);

        // Shift keys right to make room at `slot`.
        {
            let keys = view.keys_mut();
            keys.copy_within(slot..n, slot + 1);
            keys[slot] = key;
        }

        // Shift children right to make room at `slot+1`.
        {
            let children = view.children_mut();
            children.copy_within(slot + 1..n + 1, slot + 2);
            children[slot + 1] = right_child;
        }

        view.set_num_keys(n + 1);
    }

    // -----------------------------------------------------------------------
    // leaf_insert
    // -----------------------------------------------------------------------
    //
    // Inserts or updates a key-value pair in an already-confirmed-not-full
    // leaf.  Shifts existing entries right to maintain sorted order.
    //
    // Returns `Some(old_value)` on update, `None` on fresh insert.

    fn leaf_insert(&mut self, leaf_idx: u64, key: K, value: V) -> Result<Option<V>> {
        // Binary search to find the insertion slot.
        let (slot, exists) = {
            let page = self.store.page(leaf_idx);
            let view = LeafView::<K, V>::new(page, &self.layout);
            match view.keys().binary_search(&key) {
                Ok(i) => (i, true),
                Err(i) => (i, false),
            }
        };

        if exists {
            // Update existing entry.
            let old = {
                let page = self.store.page(leaf_idx);
                LeafView::<K, V>::new(page, &self.layout).values()[slot]
            };
            let page = self.store.page_mut(leaf_idx);
            LeafViewMut::<K, V>::new(page, &self.layout).values_mut()[slot] = value;
            return Ok(Some(old));
        }

        // Fresh insert: shift right and place.
        let page = self.store.page_mut(leaf_idx);
        let mut view = LeafViewMut::<K, V>::new(page, &self.layout);
        let n = view.num_keys();
        debug_assert!(n < self.layout.leaf_capacity, "leaf overflowed");

        {
            let keys = view.keys_mut();
            keys.copy_within(slot..n, slot + 1);
            keys[slot] = key;
        }
        {
            let vals = view.values_mut();
            vals.copy_within(slot..n, slot + 1);
            vals[slot] = value;
        }
        view.set_num_keys(n + 1);

        self.store.header_mut().num_entries += 1;
        Ok(None)
    }

    // -----------------------------------------------------------------------
    // remove_impl (stub — not yet implemented)
    // -----------------------------------------------------------------------

    fn remove_impl(&mut self, _key: &K) -> Result<Option<V>> {
        // TODO: Implement deletion with merge/borrow rebalancing.
        Ok(None)
    }

    // -----------------------------------------------------------------------
    // clear_impl
    // -----------------------------------------------------------------------
    //
    // Walks the tree post-order and returns every page to the free list, then
    // resets the header.  The file is not truncated; free pages are available
    // for future inserts.

    fn clear_impl(&mut self) -> Result<()> {
        let root = self.store.header().root_page;
        if root != NULL_PAGE {
            self.free_subtree(root);
        }
        {
            let hdr = self.store.header_mut();
            hdr.root_page = NULL_PAGE;
            hdr.num_entries = 0;
        }
        Ok(())
    }

    /// Recursively frees every page in the subtree rooted at `page_idx`.
    fn free_subtree(&mut self, page_idx: u64) {
        match self.node_kind(page_idx) {
            NODE_KIND_LEAF => {
                self.store.free_page(page_idx);
            }
            _ => {
                // Copy children out before freeing the page itself.
                let children: Vec<u64> = {
                    let page = self.store.page(page_idx);
                    InternalView::<K>::new(page, &self.layout).children().to_vec()
                };
                for child in children {
                    self.free_subtree(child);
                }
                self.store.free_page(page_idx);
            }
        }
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
        Self { _guard: guard }
    }
}

impl<'a, K: Ord + Pod, V: Pod> Iterator for MmapBTreeIter<'a, K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        // TODO: Walk the leaf linked-list via next_leaf pointers.
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
        Self { _guard: guard }
    }
}

impl<'a, K: Ord + Pod, V: Pod> Iterator for MmapBTreeRangeIter<'a, K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        // TODO: Walk the leaf linked-list, stopping at the end bound.
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

    fn open<K: Ord + Pod, V: Pod>(dir: &tempfile::TempDir) -> MmapBTree<K, V> {
        MmapBTreeBuilder::new()
            .path(dir.path().join("t.db"))
            .build()
            .unwrap()
    }

    #[test]
    fn builder_requires_path() {
        let result: Result<MmapBTree<i32, u64>> = MmapBTreeBuilder::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tree.db");
        MmapBTreeBuilder::<i32, u64>::new().path(&path).build().unwrap();
        assert!(path.exists());
    }

    #[test]
    fn empty_tree() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, u64>(&dir);
        assert!(tree.is_empty().unwrap());
        assert_eq!(tree.len().unwrap(), 0);
        assert_eq!(tree.get(&1).unwrap(), None);
    }

    #[test]
    fn insert_and_get_single() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, u64>(&dir);
        assert_eq!(tree.insert(42, 100).unwrap(), None);
        assert_eq!(tree.get(&42).unwrap(), Some(100));
        assert_eq!(tree.len().unwrap(), 1);
    }

    #[test]
    fn insert_updates_existing_key() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, u64>(&dir);
        tree.insert(1, 10).unwrap();
        let old = tree.insert(1, 99).unwrap();
        assert_eq!(old, Some(10));
        assert_eq!(tree.get(&1).unwrap(), Some(99));
        assert_eq!(tree.len().unwrap(), 1); // count unchanged
    }

    #[test]
    fn insert_sequential_many() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        let n = 2000_i32;
        for i in 0..n {
            assert_eq!(tree.insert(i, i * 10).unwrap(), None);
        }
        assert_eq!(tree.len().unwrap(), n as usize);
        for i in 0..n {
            assert_eq!(tree.get(&i).unwrap(), Some(i * 10));
        }
        assert_eq!(tree.get(&n).unwrap(), None);
    }

    #[test]
    fn insert_reverse_order() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        let n = 500_i32;
        for i in (0..n).rev() {
            tree.insert(i, i).unwrap();
        }
        for i in 0..n {
            assert_eq!(tree.get(&i).unwrap(), Some(i));
        }
    }

    #[test]
    fn clear_empties_tree() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        for i in 0..100_i32 {
            tree.insert(i, i).unwrap();
        }
        assert_eq!(tree.len().unwrap(), 100);
        tree.clear().unwrap();
        assert!(tree.is_empty().unwrap());
        assert_eq!(tree.get(&0).unwrap(), None);
        // Can still insert after clear.
        tree.insert(7, 77).unwrap();
        assert_eq!(tree.get(&7).unwrap(), Some(77));
    }

    #[test]
    fn persistence_across_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.db");
        {
            let tree = MmapBTreeBuilder::<i32, i32>::new()
                .path(&path)
                .build()
                .unwrap();
            for i in 0..50_i32 {
                tree.insert(i, i * 2).unwrap();
            }
            tree.flush().unwrap();
        }
        // Re-open and verify data survived.
        let tree = MmapBTreeBuilder::<i32, i32>::new().path(&path).build().unwrap();
        assert_eq!(tree.len().unwrap(), 50);
        for i in 0..50_i32 {
            assert_eq!(tree.get(&i).unwrap(), Some(i * 2));
        }
    }

    #[test]
    fn concurrent_reads() {
        use std::sync::Arc;
        let dir = tempfile::tempdir().unwrap();
        let tree = Arc::new(open::<i32, i32>(&dir));
        for i in 0..100_i32 {
            tree.insert(i, i).unwrap();
        }
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let t = Arc::clone(&tree);
                std::thread::spawn(move || {
                    for i in 0..100_i32 {
                        assert_eq!(t.get(&i).unwrap(), Some(i));
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }
}
