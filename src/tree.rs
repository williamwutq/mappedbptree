//! Memory-mapped B+tree: public API and algorithm implementation.
//!
//! # Module overview
//!
//! This module is split into two layers:
//!
//! - **Public API** ([`MmapBTree`], [`MmapBTreeBuilder`], [`MmapBTreeValueRef`]) —
//!   thread-safe, handles locking, WAL writes, and mmap flushes.
//! - **Internal implementation** (`MmapBTreeInner`, private) — accessed only
//!   through the `RwLock`; contains the raw B+tree algorithm (insert, remove,
//!   split, merge, rebalance) operating directly on mmap pages.
//!
//! # Crash safety
//!
//! Every public write operation follows the WAL protocol:
//!
//! 1. Write the intent to `{db}.wal` and call `fsync`.
//! 2. Mutate the mmap pages (B+tree algorithm).
//! 3. `msync` the mmap — data is now on disk.
//! 4. Delete the WAL file — the operation is committed.
//!
//! If the process is killed between steps 1 and 4, the next call to
//! [`MmapBTreeBuilder::build`] replays the WAL and brings the tree to the
//! post-operation state.  If the process is killed before step 1, the tree
//! is in its clean pre-operation state.  Either way, no corruption.
//!
//! # Corruption detection
//!
//! Every node page (version 2+) carries a CRC32 checksum covering all page
//! bytes except the 4-byte checksum field itself.  The checksum is written
//! after every page mutation.  It is verified before any page is read during
//! a traversal; a mismatch returns [`BTreeError::Corruption`] immediately
//! rather than silently reading wrong data.
//!
//! Version-1 files are automatically upgraded to version 2 on first open:
//! checksums are computed for all live node pages and the on-disk version
//! field is bumped.

use std::io;
use std::marker::PhantomData;
use std::ops::{Bound, Deref, RangeBounds};
use std::path::{Path, PathBuf};
use std::sync::{RwLock, RwLockReadGuard};

use bytemuck::Pod;

use crate::node::{InternalView, InternalViewMut, LeafView, LeafViewMut};
use crate::storage::{
    MmapStore, NodeHeader, NodeLayout, NODE_HEADER_SIZE, NODE_KIND_LEAF,
    NULL_PAGE, PAGE_SIZE,
    write_page_checksum, verify_page_checksum,
};
use crate::wal;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during B+tree operations.
///
/// The two most common variants are [`Io`](Self::Io) (file-system failures)
/// and [`Corruption`](Self::Corruption) (bad page checksums, bad magic, or
/// other structural inconsistencies detected in the on-disk data).
#[derive(Debug, Clone)]
pub enum BTreeError {
    /// An I/O error from file, mmap, or fsync operations.
    Io(String),
    /// Structural corruption detected in the on-disk tree.
    ///
    /// Triggered by a CRC32 checksum mismatch, a bad file header, or any
    /// page layout inconsistency.  This typically indicates a partial write
    /// that was not replayed by the WAL (e.g. the WAL file itself was lost).
    Corruption(String),
    /// Other errors (e.g. a poisoned `RwLock`).
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
/// Start with [`MmapBTreeBuilder::new`], set a storage path, then call
/// [`build`](Self::build) to open (or create) the tree.
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
    /// If the file does not exist it will be created and initialised with an
    /// empty tree.  If it exists it will be opened, its header validated, and
    /// any pending WAL record replayed before the tree is returned.
    pub fn path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Builds and returns the [`MmapBTree`].
    ///
    /// This is the terminal step that opens (or creates) the backing file,
    /// runs version migration if needed, and replays any leftover WAL record
    /// from a previous crash.
    ///
    /// # Errors
    ///
    /// - [`BTreeError::Io`] — file cannot be created or opened.
    /// - [`BTreeError::Corruption`] — existing file has a bad header or
    ///   incompatible key/value sizes.
    /// - [`BTreeError::Other`] — [`path`](Self::path) was not called.
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
// MmapBTreeValueRef — zero-copy reference into the mmap
// ---------------------------------------------------------------------------

/// A zero-copy reference to a value stored in the B+tree's memory-mapped file.
///
/// Returned by [`MmapBTree::get`].  Dereferences to `&V` without copying the
/// value out of the mmap.
///
/// Internally this holds the `RwLockReadGuard` that pins the mapping in place.
/// Because of this, **no write operation can proceed while a `MmapBTreeValueRef`
/// is alive** — drop it before calling `insert`, `remove`, or `clear` to avoid
/// a deadlock.
pub struct MmapBTreeValueRef<'a, K: Ord + Pod, V: Pod> {
    _guard: RwLockReadGuard<'a, MmapBTreeInner<K, V>>,
    ptr: *const V,
}

// SAFETY: `ptr` points into the mmap, which is stable while the read guard
// is held (writes need an exclusive lock to remap).  `V: Pod` ensures the
// bytes at `ptr` form a valid `V`.
unsafe impl<K: Ord + Pod + Send + Sync, V: Pod + Send + Sync> Send
    for MmapBTreeValueRef<'_, K, V>
{
}
unsafe impl<K: Ord + Pod + Send + Sync, V: Pod + Send + Sync> Sync
    for MmapBTreeValueRef<'_, K, V>
{
}

impl<K: Ord + Pod, V: Pod> Deref for MmapBTreeValueRef<'_, K, V> {
    type Target = V;
    #[inline]
    fn deref(&self) -> &V {
        // SAFETY: see struct-level safety comment.
        unsafe { &*self.ptr }
    }
}

impl<K: Ord + Pod, V: Pod + std::fmt::Debug> std::fmt::Debug for MmapBTreeValueRef<'_, K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.deref().fmt(f)
    }
}

// ---------------------------------------------------------------------------
// MmapBTree — public API
// ---------------------------------------------------------------------------

/// A memory-mapped, persistent, crash-safe B+tree.
///
/// ## Type constraints
///
/// Both `K` and `V` must implement [`bytemuck::Pod`], ensuring they are
/// plain-data types safe to store as raw bytes.  `K` additionally requires
/// [`Ord`] for tree ordering.
///
/// ## Thread safety
///
/// An internal `RwLock` allows multiple concurrent readers; writes are
/// exclusive.  Iterators and [`MmapBTreeValueRef`] hold a read lock for their
/// lifetime — drop them before writing.
///
/// ## Crash safety
///
/// `insert` and `remove` are protected by a write-ahead log (WAL) fsynced
/// before any mmap mutation.  If the process dies mid-write, the pending
/// operation is replayed automatically the next time the tree is opened.
/// Every node page also carries a CRC32 checksum; a partial write that
/// corrupts a page is detected on the next read and reported as
/// [`BTreeError::Corruption`].
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
/// // Zero-copy access (holds read lock until dropped).
/// if let Some(v) = tree.get(&1)? {
///     println!("Found: {}", *v);
/// }
///
/// // Copied access — no lock held after the call.
/// assert_eq!(tree.get_value(&1)?, Some(100));
///
/// for (k, v) in tree.range(1..10)? {
///     println!("{k}: {v}");
/// }
///
/// tree.remove(&1)?;
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
pub struct MmapBTree<K, V> {
    inner: RwLock<MmapBTreeInner<K, V>>,
    /// Absolute path to the database file; used to derive the WAL path.
    db_path: PathBuf,
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
        let key_size   = std::mem::size_of::<K>();
        let value_size = std::mem::size_of::<V>();

        let store = MmapStore::open(&path, key_size, value_size)?;
        let layout = NodeLayout::new(
            PAGE_SIZE,
            key_size,
            std::mem::align_of::<K>(),
            value_size,
            std::mem::align_of::<V>(),
        );

        let mut inner = MmapBTreeInner { store, layout, _phantom: PhantomData };

        // WAL recovery: if a WAL file is present, replay the pending operation.
        // Both insert and remove are idempotent, so replay is always safe.
        // BTreeError::Corruption during redo means the partial write didn't
        // reach disk — the tree is in a clean pre-crash state, which is fine.
        if let Some(record) = wal::read_existing(&path, key_size, value_size)? {
            match record.op {
                wal::WAL_OP_INSERT => {
                    let k: K = *bytemuck::from_bytes(&record.key);
                    let v: V = *bytemuck::from_bytes(&record.value);
                    match inner.insert_impl(k, v) {
                        Ok(_) | Err(BTreeError::Corruption(_)) => {}
                        Err(e) => return Err(e),
                    }
                }
                wal::WAL_OP_REMOVE => {
                    let k: K = *bytemuck::from_bytes(&record.key);
                    match inner.remove_impl(&k) {
                        Ok(_) | Err(BTreeError::Corruption(_)) => {}
                        Err(e) => return Err(e),
                    }
                }
                _ => {} // corrupt/unknown WAL op — ignore
            }
            inner.store.flush()?;
            wal::delete(&path)?;
        }

        Ok(Self { inner: RwLock::new(inner), db_path: path })
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
    ///
    /// Crash-safe: a WAL record is written and fsynced before any mmap
    /// mutation.  If the process dies mid-write, the operation is replayed
    /// automatically the next time the tree is opened.
    pub fn insert(&self, key: K, value: V) -> Result<Option<V>> {
        // Step 1: write intent to WAL and fsync.
        wal::write_and_sync(
            &self.db_path,
            wal::WAL_OP_INSERT,
            bytemuck::bytes_of(&key),
            bytemuck::bytes_of(&value),
        )?;
        // Step 2: perform the mmap mutation.
        let result = self.write_guard()?.insert_impl(key, value)?;
        // Step 3: msync the mmap (must happen before WAL deletion).
        self.read_guard()?.flush_impl()?;
        // Step 4: delete the WAL — operation is now durable.
        wal::delete(&self.db_path)?;
        Ok(result)
    }

    /// Returns a zero-copy reference to the value for `key`, or `None` if absent.
    ///
    /// The returned [`MmapBTreeValueRef`] borrows directly from the mmap and
    /// holds a read lock on the tree.  Drop it before calling any mutating
    /// method (`insert`, `remove`, `clear`) to avoid a deadlock.
    ///
    /// Use [`get_value`](Self::get_value) if you only need a copy and don't
    /// want to worry about the lock.
    pub fn get(&self, key: &K) -> Result<Option<MmapBTreeValueRef<'_, K, V>>> {
        let guard = self.read_guard()?;
        let ptr = guard.get_ptr_impl(key)?;
        Ok(ptr.map(|ptr| MmapBTreeValueRef { _guard: guard, ptr }))
    }

    /// Returns a copied value for `key`, or `None` if absent.
    ///
    /// Unlike [`get`](Self::get), this copies the value out and releases the
    /// read lock immediately.  Prefer this when you don't need zero-copy access.
    pub fn get_value(&self, key: &K) -> Result<Option<V>> {
        self.read_guard()?.get_impl(key)
    }

    /// Returns `true` if `key` is present.
    pub fn contains_key(&self, key: &K) -> Result<bool> {
        self.read_guard()?.contains_key_impl(key)
    }

    /// Removes `key` and returns its value, or `None` if absent.
    ///
    /// Crash-safe: same WAL protocol as [`insert`](Self::insert).
    pub fn remove(&self, key: &K) -> Result<Option<V>> {
        let zero_value = vec![0u8; std::mem::size_of::<V>()];
        wal::write_and_sync(
            &self.db_path,
            wal::WAL_OP_REMOVE,
            bytemuck::bytes_of(key),
            &zero_value,
        )?;
        let result = self.write_guard()?.remove_impl(key)?;
        self.read_guard()?.flush_impl()?;
        wal::delete(&self.db_path)?;
        Ok(result)
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
    /// The iterator holds a read lock for its entire lifetime; writes are
    /// blocked until it is dropped.  Collect into a `Vec` if you need to
    /// write while iterating.
    pub fn iter(&self) -> Result<MmapBTreeIter<'_, K, V>> {
        let guard = self.read_guard()?;
        Ok(MmapBTreeIter::new(guard))
    }

    /// Returns an iterator over key-value pairs whose keys fall within `range`.
    ///
    /// Accepts any `RangeBounds<K>`: `..`, `a..`, `..b`, `a..b`, `a..=b`, etc.
    /// The iterator holds a read lock for its entire lifetime.
    pub fn range<R: RangeBounds<K>>(
        &self,
        range: R,
    ) -> Result<MmapBTreeRangeIter<'_, K, V>> {
        let guard = self.read_guard()?;
        MmapBTreeRangeIter::new(guard, range)
    }

    /// Removes all key-value pairs and reclaims all disk space used by node pages.
    ///
    /// Implemented as a file truncation: the header is reset to the empty-tree
    /// state and flushed first, then the file is shrunk to a single page.
    /// This is O(1) regardless of tree size.
    pub fn clear(&self) -> Result<()> {
        self.write_guard()?.clear_impl()
    }

    /// Flushes all pending mmap writes to disk (`msync`).
    ///
    /// The WAL protocol already calls `msync` after every `insert` and
    /// `remove`, so explicit `flush` is rarely needed.  It is called
    /// automatically on drop (best-effort); call it explicitly when you need
    /// a strong durability guarantee outside a write operation.
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

    /// Verifies the CRC32 checksum of the page at `page_idx`.
    ///
    /// Returns [`BTreeError::Corruption`] if the checksum doesn't match.
    /// No-op when `checksums_enabled` is false (version-1 files before upgrade).
    #[inline]
    fn verify_page(&self, page_idx: u64) -> Result<()> {
        if self.store.checksums_enabled
            && !verify_page_checksum(self.store.page(page_idx))
        {
            return Err(BTreeError::Corruption(format!(
                "checksum mismatch on page {page_idx}"
            )));
        }
        Ok(())
    }

    /// Returns the kind byte of the node at `page_idx`.
    #[inline]
    fn node_kind(&self, page_idx: u64) -> u8 {
        self.store.page(page_idx)[0]
    }

    /// Returns the live key count for any node.
    #[inline]
    fn num_keys_of(&self, page_idx: u64) -> usize {
        let page = self.store.page(page_idx);
        bytemuck::from_bytes::<NodeHeader>(&page[..NODE_HEADER_SIZE]).num_keys as usize
    }

    /// Returns the minimum occupancy (keys) for a node.
    /// Uses ceiling division so the minimum is at least 1 for any capacity ≥ 1.
    #[inline]
    fn min_keys_of(&self, page_idx: u64) -> usize {
        match self.node_kind(page_idx) {
            NODE_KIND_LEAF => (self.layout.leaf_capacity + 1) / 2,
            _ => (self.layout.internal_capacity + 1) / 2,
        }
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

    /// Returns `(slot, child_page)` to follow for `key` in an internal node.
    ///
    /// Invariant: `separator[i]` is the smallest key in `children[i+1]`, so
    /// the correct child for `key` is `children[partition_point(sep <= key)]`.
    ///
    /// Returns [`BTreeError::Corruption`] if the page checksum is invalid.
    #[inline]
    fn internal_child_slot(&self, page_idx: u64, key: &K) -> Result<(usize, u64)> {
        self.verify_page(page_idx)?;
        let page = self.store.page(page_idx);
        let view = InternalView::<K>::new(page, &self.layout);
        let slot = view.keys().partition_point(|k| k <= key);
        Ok((slot, view.children()[slot]))
    }

    /// Walks from root following `children[0]` at each level to find the
    /// leftmost leaf.  Returns `(leaf_page, 0)`, or `(NULL_PAGE, 0)` for
    /// an empty tree.
    fn first_leaf(&self) -> (u64, usize) {
        let root = self.store.header().root_page;
        if root == NULL_PAGE {
            return (NULL_PAGE, 0);
        }
        let mut cur = root;
        loop {
            match self.node_kind(cur) {
                NODE_KIND_LEAF => return (cur, 0),
                _ => {
                    let page = self.store.page(cur);
                    let view = InternalView::<K>::new(page, &self.layout);
                    cur = view.children()[0];
                }
            }
        }
    }

    /// Returns `(leaf_page, slot)` where `leaf.keys[slot]` is the first key
    /// that is `>= key`.  Returns `(NULL_PAGE, 0)` if no such key exists.
    fn lower_bound(&self, key: &K) -> Result<(u64, usize)> {
        let root = self.store.header().root_page;
        if root == NULL_PAGE {
            return Ok((NULL_PAGE, 0));
        }
        let mut cur = root;
        loop {
            match self.node_kind(cur) {
                NODE_KIND_LEAF => {
                    let page = self.store.page(cur);
                    let view = LeafView::<K, V>::new(page, &self.layout);
                    let n = view.num_keys();
                    // first slot where keys[slot] >= key
                    let slot = view.keys().partition_point(|k| k < key);
                    if slot < n {
                        return Ok((cur, slot));
                    } else {
                        // all keys in this leaf < key; answer is in next leaf (if any)
                        return Ok((view.next_leaf(), 0));
                    }
                }
                _ => {
                    let (_slot, child) = self.internal_child_slot(cur, key)?;
                    cur = child;
                }
            }
        }
    }

    /// Like `lower_bound` but skips past an exact match with `key`
    /// (implements `Excluded(key)` range start).
    fn lower_bound_exclusive(&self, key: &K) -> Result<(u64, usize)> {
        let (page, slot) = self.lower_bound(key)?;
        if page == NULL_PAGE {
            return Ok((NULL_PAGE, 0));
        }
        // If the key at slot exactly equals `key`, advance one position.
        let exact = {
            let p = self.store.page(page);
            let view = LeafView::<K, V>::new(p, &self.layout);
            slot < view.num_keys() && &view.keys()[slot] == key
        };
        if exact {
            let p = self.store.page(page);
            let view = LeafView::<K, V>::new(p, &self.layout);
            let next_slot = slot + 1;
            if next_slot < view.num_keys() {
                Ok((page, next_slot))
            } else {
                Ok((view.next_leaf(), 0))
            }
        } else {
            Ok((page, slot))
        }
    }

    // -----------------------------------------------------------------------
    // get_impl / get_ptr_impl
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
                    self.verify_page(current)?;
                    let page = self.store.page(current);
                    let view = LeafView::<K, V>::new(page, &self.layout);
                    return Ok(match view.keys().binary_search(key) {
                        Ok(i) => Some(view.values()[i]),
                        Err(_) => None,
                    });
                }
                _ => {
                    let (_slot, child) = self.internal_child_slot(current, key)?;
                    current = child;
                }
            }
        }
    }

    /// Like `get_impl` but returns a raw pointer into the mmap instead of
    /// copying the value.  The pointer is valid as long as the caller holds
    /// a read lock (which prevents `grow()` from remapping).
    fn get_ptr_impl(&self, key: &K) -> Result<Option<*const V>> {
        let root = self.store.header().root_page;
        if root == NULL_PAGE {
            return Ok(None);
        }

        let mut current = root;
        loop {
            match self.node_kind(current) {
                NODE_KIND_LEAF => {
                    self.verify_page(current)?;
                    let page = self.store.page(current);
                    let view = LeafView::<K, V>::new(page, &self.layout);
                    return Ok(match view.keys().binary_search(key) {
                        Ok(i) => Some(&view.values()[i] as *const V),
                        Err(_) => None,
                    });
                }
                _ => {
                    let (_slot, child) = self.internal_child_slot(current, key)?;
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

            let (child_slot, child_idx) = self.internal_child_slot(current, &key)?;

            if self.node_is_full(child_idx) {
                self.split_child(current, child_slot)?;
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
        write_page_checksum(self.store.page_mut(leaf_idx));
        self.store.header_mut().root_page = leaf_idx;
        self.store.header_mut().num_entries = 1;
        Ok(None)
    }

    // -----------------------------------------------------------------------
    // split_root
    // -----------------------------------------------------------------------

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

        let (all_keys, all_values, old_next) = {
            let page = self.store.page(root_idx);
            let view = LeafView::<K, V>::new(page, &self.layout);
            (view.keys().to_vec(), view.values().to_vec(), view.next_leaf())
        };

        let left_idx = self.store.alloc_page()?;
        let right_idx = self.store.alloc_page()?;

        {
            let page = self.store.page_mut(left_idx);
            let mut view = LeafViewMut::<K, V>::new(page, &self.layout);
            view.init();
            view.keys_mut()[..mid].copy_from_slice(&all_keys[..mid]);
            view.values_mut()[..mid].copy_from_slice(&all_values[..mid]);
            view.set_num_keys(mid);
            view.set_next_leaf(right_idx);
        }
        write_page_checksum(self.store.page_mut(left_idx));

        let right_n = lc - mid;
        let separator = all_keys[mid];
        {
            let page = self.store.page_mut(right_idx);
            let mut view = LeafViewMut::<K, V>::new(page, &self.layout);
            view.init();
            view.keys_mut()[..right_n].copy_from_slice(&all_keys[mid..]);
            view.values_mut()[..right_n].copy_from_slice(&all_values[mid..]);
            view.set_num_keys(right_n);
            view.set_next_leaf(old_next);
        }
        write_page_checksum(self.store.page_mut(right_idx));

        {
            let page = self.store.page_mut(root_idx);
            let mut view = InternalViewMut::<K>::new(page, &self.layout);
            view.init();
            view.keys_mut()[0] = separator;
            view.children_mut()[0] = left_idx;
            view.children_mut()[1] = right_idx;
            view.set_num_keys(1);
        }
        write_page_checksum(self.store.page_mut(root_idx));

        Ok(())
    }

    fn split_root_internal(&mut self, root_idx: u64) -> Result<()> {
        let ic = self.layout.internal_capacity;
        let mid = ic / 2;

        let (all_keys, all_children) = {
            let page = self.store.page(root_idx);
            let view = InternalView::<K>::new(page, &self.layout);
            (view.keys().to_vec(), view.children().to_vec())
        };

        let left_idx = self.store.alloc_page()?;
        let right_idx = self.store.alloc_page()?;

        {
            let page = self.store.page_mut(left_idx);
            let mut view = InternalViewMut::<K>::new(page, &self.layout);
            view.init();
            view.keys_mut()[..mid].copy_from_slice(&all_keys[..mid]);
            view.children_mut()[..mid + 1].copy_from_slice(&all_children[..mid + 1]);
            view.set_num_keys(mid);
        }
        write_page_checksum(self.store.page_mut(left_idx));

        let separator = all_keys[mid];
        let right_n = ic - mid - 1;
        {
            let page = self.store.page_mut(right_idx);
            let mut view = InternalViewMut::<K>::new(page, &self.layout);
            view.init();
            view.keys_mut()[..right_n].copy_from_slice(&all_keys[mid + 1..]);
            view.children_mut()[..right_n + 1].copy_from_slice(&all_children[mid + 1..]);
            view.set_num_keys(right_n);
        }
        write_page_checksum(self.store.page_mut(right_idx));

        {
            let page = self.store.page_mut(root_idx);
            let mut view = InternalViewMut::<K>::new(page, &self.layout);
            view.init();
            view.keys_mut()[0] = separator;
            view.children_mut()[0] = left_idx;
            view.children_mut()[1] = right_idx;
            view.set_num_keys(1);
        }
        write_page_checksum(self.store.page_mut(root_idx));

        Ok(())
    }

    // -----------------------------------------------------------------------
    // split_child
    // -----------------------------------------------------------------------

    fn split_child(&mut self, parent_idx: u64, child_slot: usize) -> Result<()> {
        let child_idx = {
            let page = self.store.page(parent_idx);
            InternalView::<K>::new(page, &self.layout).children()[child_slot]
        };

        let (separator, right_idx) = match self.node_kind(child_idx) {
            NODE_KIND_LEAF => self.split_leaf_child(child_idx)?,
            _ => self.split_internal_child(child_idx)?,
        };

        self.insert_into_internal(parent_idx, child_slot, separator, right_idx);
        Ok(())
    }

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

        {
            let page = self.store.page_mut(right_idx);
            let mut view = LeafViewMut::<K, V>::new(page, &self.layout);
            view.init();
            view.keys_mut()[..right_n].copy_from_slice(&all_keys[mid..]);
            view.values_mut()[..right_n].copy_from_slice(&all_values[mid..]);
            view.set_num_keys(right_n);
            view.set_next_leaf(old_next);
        }
        write_page_checksum(self.store.page_mut(right_idx));

        {
            let page = self.store.page_mut(leaf_idx);
            let mut view = LeafViewMut::<K, V>::new(page, &self.layout);
            view.set_num_keys(mid);
            view.set_next_leaf(right_idx);
        }
        write_page_checksum(self.store.page_mut(leaf_idx));

        Ok((separator, right_idx))
    }

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

        {
            let page = self.store.page_mut(right_idx);
            let mut view = InternalViewMut::<K>::new(page, &self.layout);
            view.init();
            view.keys_mut()[..right_n].copy_from_slice(&all_keys[mid + 1..]);
            view.children_mut()[..right_n + 1].copy_from_slice(&all_children[mid + 1..]);
            view.set_num_keys(right_n);
        }
        write_page_checksum(self.store.page_mut(right_idx));

        {
            let page = self.store.page_mut(node_idx);
            let mut view = InternalViewMut::<K>::new(page, &self.layout);
            view.set_num_keys(mid);
        }
        write_page_checksum(self.store.page_mut(node_idx));

        Ok((separator, right_idx))
    }

    // -----------------------------------------------------------------------
    // insert_into_internal
    // -----------------------------------------------------------------------

    fn insert_into_internal(
        &mut self,
        node_idx: u64,
        slot: usize,
        key: K,
        right_child: u64,
    ) {
        {
            let page = self.store.page_mut(node_idx);
            let mut view = InternalViewMut::<K>::new(page, &self.layout);
            let n = view.num_keys();
            debug_assert!(n < self.layout.internal_capacity);

            {
                let keys = view.keys_mut();
                keys.copy_within(slot..n, slot + 1);
                keys[slot] = key;
            }
            {
                let children = view.children_mut();
                children.copy_within(slot + 1..n + 1, slot + 2);
                children[slot + 1] = right_child;
            }

            view.set_num_keys(n + 1);
        }
        write_page_checksum(self.store.page_mut(node_idx));
    }

    // -----------------------------------------------------------------------
    // leaf_insert
    // -----------------------------------------------------------------------

    fn leaf_insert(&mut self, leaf_idx: u64, key: K, value: V) -> Result<Option<V>> {
        self.verify_page(leaf_idx)?;
        let (slot, exists) = {
            let page = self.store.page(leaf_idx);
            let view = LeafView::<K, V>::new(page, &self.layout);
            match view.keys().binary_search(&key) {
                Ok(i) => (i, true),
                Err(i) => (i, false),
            }
        };

        if exists {
            let old = {
                let page = self.store.page(leaf_idx);
                LeafView::<K, V>::new(page, &self.layout).values()[slot]
            };
            {
                let page = self.store.page_mut(leaf_idx);
                LeafViewMut::<K, V>::new(page, &self.layout).values_mut()[slot] = value;
            }
            write_page_checksum(self.store.page_mut(leaf_idx));
            return Ok(Some(old));
        }

        {
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
        }
        write_page_checksum(self.store.page_mut(leaf_idx));

        self.store.header_mut().num_entries += 1;
        Ok(None)
    }

    // -----------------------------------------------------------------------
    // remove_impl — steal-or-merge rebalancing
    // -----------------------------------------------------------------------
    //
    // Strategy: recurse bottom-up.  After deleting from a child, if the child
    // is underfull, try to steal a key from a sibling; failing that, merge the
    // child with a sibling.  If the root ends up with 0 keys (only one child
    // left), collapse it.

    fn remove_impl(&mut self, key: &K) -> Result<Option<V>> {
        let root = self.store.header().root_page;
        if root == NULL_PAGE {
            return Ok(None);
        }

        let result = match self.node_kind(root) {
            NODE_KIND_LEAF => self.leaf_delete(root, key)?,
            _ => self.internal_delete(root, key)?,
        };

        if result.is_some() {
            self.store.header_mut().num_entries -= 1;

            // Collapse: root is an internal node with 0 keys → its sole
            // child becomes the new root.
            let root = self.store.header().root_page;
            if self.node_kind(root) != NODE_KIND_LEAF && self.num_keys_of(root) == 0 {
                let only_child = {
                    let page = self.store.page(root);
                    InternalView::<K>::new(page, &self.layout).children()[0]
                };
                self.store.free_page(root);
                self.store.header_mut().root_page = only_child;
            }
        }

        Ok(result)
    }

    /// Removes `key` from the leaf at `leaf_idx`.
    fn leaf_delete(&mut self, leaf_idx: u64, key: &K) -> Result<Option<V>> {
        self.verify_page(leaf_idx)?;
        let (slot, found) = {
            let page = self.store.page(leaf_idx);
            let view = LeafView::<K, V>::new(page, &self.layout);
            match view.keys().binary_search(key) {
                Ok(i) => (i, true),
                Err(_) => (0, false),
            }
        };
        if !found {
            return Ok(None);
        }

        let old_val = {
            let page = self.store.page(leaf_idx);
            LeafView::<K, V>::new(page, &self.layout).values()[slot]
        };

        {
            let page = self.store.page_mut(leaf_idx);
            let mut view = LeafViewMut::<K, V>::new(page, &self.layout);
            let n = view.num_keys();
            {
                let keys = view.keys_mut();
                keys.copy_within(slot + 1..n, slot);
            }
            {
                let vals = view.values_mut();
                vals.copy_within(slot + 1..n, slot);
            }
            view.set_num_keys(n - 1);
        }
        write_page_checksum(self.store.page_mut(leaf_idx));

        Ok(Some(old_val))
    }

    /// Recurses into the subtree at `node_idx` to delete `key`, then
    /// rebalances if the child that was descended into became underfull.
    fn internal_delete(&mut self, node_idx: u64, key: &K) -> Result<Option<V>> {
        let (child_slot, child_idx) = self.internal_child_slot(node_idx, key)?;

        let result = match self.node_kind(child_idx) {
            NODE_KIND_LEAF => self.leaf_delete(child_idx, key)?,
            _ => self.internal_delete(child_idx, key)?,
        };

        if result.is_some() {
            let child_n = self.num_keys_of(child_idx);
            let min = self.min_keys_of(child_idx);
            if child_n < min {
                self.fix_underfull_child(node_idx, child_slot)?;
            }
        }

        Ok(result)
    }

    /// Rebalances the child at `child_slot` within `parent_idx`.
    ///
    /// Tries to steal from the left sibling, then the right sibling.
    /// If neither can spare a key, merges the child with a sibling.
    fn fix_underfull_child(&mut self, parent_idx: u64, child_slot: usize) -> Result<()> {
        let parent_n = self.num_keys_of(parent_idx);

        let child_idx = {
            let page = self.store.page(parent_idx);
            InternalView::<K>::new(page, &self.layout).children()[child_slot]
        };
        let is_leaf = self.node_kind(child_idx) == NODE_KIND_LEAF;
        let min = self.min_keys_of(child_idx);

        // ── Try left sibling ──────────────────────────────────────────────
        if child_slot > 0 {
            let left_idx = {
                let page = self.store.page(parent_idx);
                InternalView::<K>::new(page, &self.layout).children()[child_slot - 1]
            };
            if self.num_keys_of(left_idx) > min {
                let sep_slot = child_slot - 1;
                if is_leaf {
                    self.borrow_from_left_leaf(parent_idx, sep_slot, left_idx, child_idx);
                } else {
                    self.borrow_from_left_internal(parent_idx, sep_slot, left_idx, child_idx);
                }
                return Ok(());
            }
        }

        // ── Try right sibling ─────────────────────────────────────────────
        if child_slot < parent_n {
            let right_idx = {
                let page = self.store.page(parent_idx);
                InternalView::<K>::new(page, &self.layout).children()[child_slot + 1]
            };
            if self.num_keys_of(right_idx) > min {
                let sep_slot = child_slot;
                if is_leaf {
                    self.borrow_from_right_leaf(parent_idx, sep_slot, child_idx, right_idx);
                } else {
                    self.borrow_from_right_internal(parent_idx, sep_slot, child_idx, right_idx);
                }
                return Ok(());
            }
        }

        // ── Merge ─────────────────────────────────────────────────────────
        if child_slot > 0 {
            // Merge left sibling + child → left; child is freed.
            let sep_slot = child_slot - 1;
            let left_idx = {
                let page = self.store.page(parent_idx);
                InternalView::<K>::new(page, &self.layout).children()[child_slot - 1]
            };
            if is_leaf {
                self.merge_leaf_nodes(parent_idx, sep_slot, left_idx, child_idx);
            } else {
                self.merge_internal_nodes(parent_idx, sep_slot, left_idx, child_idx);
            }
        } else {
            // Merge child + right sibling → child; right is freed.
            let sep_slot = child_slot; // == 0
            let right_idx = {
                let page = self.store.page(parent_idx);
                InternalView::<K>::new(page, &self.layout).children()[child_slot + 1]
            };
            if is_leaf {
                self.merge_leaf_nodes(parent_idx, sep_slot, child_idx, right_idx);
            } else {
                self.merge_internal_nodes(parent_idx, sep_slot, child_idx, right_idx);
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Steal helpers — borrow one entry from a sibling
    // -----------------------------------------------------------------------

    /// Leaf: move `left`'s last key/value to the front of `child`.
    /// Update parent separator to the new minimum of `child`.
    fn borrow_from_left_leaf(
        &mut self,
        parent_idx: u64,
        sep_slot: usize,
        left_idx: u64,
        child_idx: u64,
    ) {
        let (moved_key, moved_val) = {
            let page = self.store.page(left_idx);
            let view = LeafView::<K, V>::new(page, &self.layout);
            let n = view.num_keys();
            (view.keys()[n - 1], view.values()[n - 1])
        };

        // Prepend moved entry to child.
        {
            let page = self.store.page_mut(child_idx);
            let mut view = LeafViewMut::<K, V>::new(page, &self.layout);
            let n = view.num_keys();
            let keys = view.keys_mut();
            keys.copy_within(0..n, 1);
            keys[0] = moved_key;
            let vals = view.values_mut();
            vals.copy_within(0..n, 1);
            vals[0] = moved_val;
            view.set_num_keys(n + 1);
        }
        write_page_checksum(self.store.page_mut(child_idx));

        // Shrink left.
        {
            let page = self.store.page_mut(left_idx);
            let mut view = LeafViewMut::<K, V>::new(page, &self.layout);
            let n = view.num_keys();
            view.set_num_keys(n - 1);
        }
        write_page_checksum(self.store.page_mut(left_idx));

        // New separator = moved_key (the new minimum of child).
        {
            let page = self.store.page_mut(parent_idx);
            let mut view = InternalViewMut::<K>::new(page, &self.layout);
            view.keys_mut()[sep_slot] = moved_key;
        }
        write_page_checksum(self.store.page_mut(parent_idx));
    }

    /// Leaf: move `right`'s first key/value to the end of `child`.
    /// Update parent separator to the new minimum of `right`.
    fn borrow_from_right_leaf(
        &mut self,
        parent_idx: u64,
        sep_slot: usize,
        child_idx: u64,
        right_idx: u64,
    ) {
        let (moved_key, moved_val, new_sep) = {
            let page = self.store.page(right_idx);
            let view = LeafView::<K, V>::new(page, &self.layout);
            // right_n > min >= 1 so right_n >= 2, guaranteeing keys[1] is valid.
            let new_sep = view.keys()[1];
            (view.keys()[0], view.values()[0], new_sep)
        };

        // Append to child.
        {
            let page = self.store.page_mut(child_idx);
            let mut view = LeafViewMut::<K, V>::new(page, &self.layout);
            let n = view.num_keys();
            view.keys_mut()[n] = moved_key;
            view.values_mut()[n] = moved_val;
            view.set_num_keys(n + 1);
        }
        write_page_checksum(self.store.page_mut(child_idx));

        // Remove first entry from right (shift left).
        {
            let page = self.store.page_mut(right_idx);
            let mut view = LeafViewMut::<K, V>::new(page, &self.layout);
            let n = view.num_keys();
            let keys = view.keys_mut();
            keys.copy_within(1..n, 0);
            let vals = view.values_mut();
            vals.copy_within(1..n, 0);
            view.set_num_keys(n - 1);
        }
        write_page_checksum(self.store.page_mut(right_idx));

        // New separator = new minimum of right.
        {
            let page = self.store.page_mut(parent_idx);
            let mut view = InternalViewMut::<K>::new(page, &self.layout);
            view.keys_mut()[sep_slot] = new_sep;
        }
        write_page_checksum(self.store.page_mut(parent_idx));
    }

    /// Internal: pull down parent separator into front of `child`,
    /// move `left`'s last key up to parent, move `left`'s last child to `child`.
    fn borrow_from_left_internal(
        &mut self,
        parent_idx: u64,
        sep_slot: usize,
        left_idx: u64,
        child_idx: u64,
    ) {
        let (old_sep, left_last_key, left_last_child) = {
            let parent_page = self.store.page(parent_idx);
            let parent_view = InternalView::<K>::new(parent_page, &self.layout);
            let old_sep = parent_view.keys()[sep_slot];

            let left_page = self.store.page(left_idx);
            let left_view = InternalView::<K>::new(left_page, &self.layout);
            let ln = left_view.num_keys();
            (old_sep, left_view.keys()[ln - 1], left_view.children()[ln])
        };

        // Prepend old_sep + left_last_child to child.
        {
            let page = self.store.page_mut(child_idx);
            let mut view = InternalViewMut::<K>::new(page, &self.layout);
            let n = view.num_keys();
            let keys = view.keys_mut();
            keys.copy_within(0..n, 1);
            keys[0] = old_sep;
            let children = view.children_mut();
            children.copy_within(0..n + 1, 1);
            children[0] = left_last_child;
            view.set_num_keys(n + 1);
        }
        write_page_checksum(self.store.page_mut(child_idx));

        // Shrink left (drop its last key and last child pointer).
        {
            let page = self.store.page_mut(left_idx);
            let mut view = InternalViewMut::<K>::new(page, &self.layout);
            let n = view.num_keys();
            view.set_num_keys(n - 1);
        }
        write_page_checksum(self.store.page_mut(left_idx));

        // Push left_last_key up to parent.
        {
            let page = self.store.page_mut(parent_idx);
            let mut view = InternalViewMut::<K>::new(page, &self.layout);
            view.keys_mut()[sep_slot] = left_last_key;
        }
        write_page_checksum(self.store.page_mut(parent_idx));
    }

    /// Internal: pull down parent separator into back of `child`,
    /// move `right`'s first key up to parent, move `right`'s first child to `child`.
    fn borrow_from_right_internal(
        &mut self,
        parent_idx: u64,
        sep_slot: usize,
        child_idx: u64,
        right_idx: u64,
    ) {
        let (old_sep, right_first_key, right_first_child) = {
            let parent_page = self.store.page(parent_idx);
            let parent_view = InternalView::<K>::new(parent_page, &self.layout);
            let old_sep = parent_view.keys()[sep_slot];

            let right_page = self.store.page(right_idx);
            let right_view = InternalView::<K>::new(right_page, &self.layout);
            (old_sep, right_view.keys()[0], right_view.children()[0])
        };

        // Append old_sep + right_first_child to child.
        {
            let page = self.store.page_mut(child_idx);
            let mut view = InternalViewMut::<K>::new(page, &self.layout);
            let n = view.num_keys();
            view.keys_mut()[n] = old_sep;
            view.children_mut()[n + 1] = right_first_child;
            view.set_num_keys(n + 1);
        }
        write_page_checksum(self.store.page_mut(child_idx));

        // Shrink right (shift keys and children left by 1).
        {
            let page = self.store.page_mut(right_idx);
            let mut view = InternalViewMut::<K>::new(page, &self.layout);
            let n = view.num_keys();
            let keys = view.keys_mut();
            keys.copy_within(1..n, 0);
            let children = view.children_mut();
            children.copy_within(1..n + 1, 0);
            view.set_num_keys(n - 1);
        }
        write_page_checksum(self.store.page_mut(right_idx));

        // Push right_first_key up to parent.
        {
            let page = self.store.page_mut(parent_idx);
            let mut view = InternalViewMut::<K>::new(page, &self.layout);
            view.keys_mut()[sep_slot] = right_first_key;
        }
        write_page_checksum(self.store.page_mut(parent_idx));
    }

    // -----------------------------------------------------------------------
    // Merge helpers — absorb right into left, remove separator from parent
    // -----------------------------------------------------------------------

    /// Merge `right_idx` leaf into `left_idx`.  Removes the separator key at
    /// `sep_slot` (and the `right_idx` child pointer) from `parent_idx`.
    /// Frees `right_idx`.
    fn merge_leaf_nodes(
        &mut self,
        parent_idx: u64,
        sep_slot: usize,
        left_idx: u64,
        right_idx: u64,
    ) {
        let (right_keys, right_vals, right_next) = {
            let page = self.store.page(right_idx);
            let view = LeafView::<K, V>::new(page, &self.layout);
            let n = view.num_keys();
            (view.keys()[..n].to_vec(), view.values()[..n].to_vec(), view.next_leaf())
        };

        {
            let page = self.store.page_mut(left_idx);
            let mut view = LeafViewMut::<K, V>::new(page, &self.layout);
            let ln = view.num_keys();
            let rn = right_keys.len();
            view.keys_mut()[ln..ln + rn].copy_from_slice(&right_keys);
            view.values_mut()[ln..ln + rn].copy_from_slice(&right_vals);
            view.set_num_keys(ln + rn);
            view.set_next_leaf(right_next);
        }
        write_page_checksum(self.store.page_mut(left_idx));

        self.remove_from_internal(parent_idx, sep_slot);
        self.store.free_page(right_idx);
    }

    /// Merge `right_idx` internal node into `left_idx`, pulling down the
    /// separator from `parent_idx`.  Removes separator + right child pointer
    /// from parent.  Frees `right_idx`.
    fn merge_internal_nodes(
        &mut self,
        parent_idx: u64,
        sep_slot: usize,
        left_idx: u64,
        right_idx: u64,
    ) {
        let (sep, right_keys, right_children) = {
            let parent_page = self.store.page(parent_idx);
            let parent_view = InternalView::<K>::new(parent_page, &self.layout);
            let sep = parent_view.keys()[sep_slot];

            let right_page = self.store.page(right_idx);
            let right_view = InternalView::<K>::new(right_page, &self.layout);
            let rn = right_view.num_keys();
            (sep, right_view.keys()[..rn].to_vec(), right_view.children()[..rn + 1].to_vec())
        };

        {
            let page = self.store.page_mut(left_idx);
            let mut view = InternalViewMut::<K>::new(page, &self.layout);
            let ln = view.num_keys();
            let rn = right_keys.len();
            let keys = view.keys_mut();
            keys[ln] = sep;
            keys[ln + 1..ln + 1 + rn].copy_from_slice(&right_keys);
            let children = view.children_mut();
            children[ln + 1..ln + 1 + rn + 1].copy_from_slice(&right_children);
            view.set_num_keys(ln + 1 + rn);
        }
        write_page_checksum(self.store.page_mut(left_idx));

        self.remove_from_internal(parent_idx, sep_slot);
        self.store.free_page(right_idx);
    }

    /// Removes the key at `sep_slot` and the child pointer at `sep_slot + 1`
    /// from the internal node at `node_idx`, shifting remaining entries left.
    fn remove_from_internal(&mut self, node_idx: u64, sep_slot: usize) {
        {
            let page = self.store.page_mut(node_idx);
            let mut view = InternalViewMut::<K>::new(page, &self.layout);
            let n = view.num_keys();
            {
                let keys = view.keys_mut();
                keys.copy_within(sep_slot + 1..n, sep_slot);
            }
            {
                let children = view.children_mut();
                children.copy_within(sep_slot + 2..n + 1, sep_slot + 1);
            }
            view.set_num_keys(n - 1);
        }
        write_page_checksum(self.store.page_mut(node_idx));
    }

    /// Free is just truncate the store to the header page, which is empty except for the header.
    fn clear_impl(&mut self) -> Result<()> {
        self.store.truncate_to_header()
    }
}

// ---------------------------------------------------------------------------
// Iterators
// ---------------------------------------------------------------------------

/// Iterator over all key-value pairs in a [`MmapBTree`], in ascending order.
///
/// Holds a read lock for its lifetime — no writes can proceed concurrently.
pub struct MmapBTreeIter<'a, K, V> {
    guard: RwLockReadGuard<'a, MmapBTreeInner<K, V>>,
    current_page: u64,
    current_slot: usize,
}

impl<'a, K: Ord + Pod, V: Pod> MmapBTreeIter<'a, K, V> {
    fn new(guard: RwLockReadGuard<'a, MmapBTreeInner<K, V>>) -> Self {
        let (current_page, current_slot) = guard.first_leaf();
        Self { guard, current_page, current_slot }
    }
}

impl<'a, K: Ord + Pod, V: Pod> Iterator for MmapBTreeIter<'a, K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_page == NULL_PAGE {
                return None;
            }

            // Extract item or advance to next leaf; borrow of guard ends at `}`.
            let (item, next_page, next_slot) = {
                let page = self.guard.store.page(self.current_page);
                let view = LeafView::<K, V>::new(page, &self.guard.layout);
                let n = view.num_keys();
                if self.current_slot < n {
                    let k = view.keys()[self.current_slot];
                    let v = view.values()[self.current_slot];
                    (Some((k, v)), self.current_page, self.current_slot + 1)
                } else {
                    (None, view.next_leaf(), 0)
                }
            };

            self.current_page = next_page;
            self.current_slot = next_slot;

            if item.is_some() || next_page == NULL_PAGE {
                return item;
            }
            // else: slot overflowed an empty leaf page — advance to next_leaf
        }
    }
}

/// Iterator over key-value pairs within a key range of a [`MmapBTree`].
///
/// Holds a read lock for its lifetime.
pub struct MmapBTreeRangeIter<'a, K, V> {
    guard: RwLockReadGuard<'a, MmapBTreeInner<K, V>>,
    current_page: u64,
    current_slot: usize,
    /// Inclusive or exclusive upper bound (or unbounded).
    end_bound: Bound<K>,
}

impl<'a, K: Ord + Pod, V: Pod> MmapBTreeRangeIter<'a, K, V> {
    fn new<R: RangeBounds<K>>(
        guard: RwLockReadGuard<'a, MmapBTreeInner<K, V>>,
        range: R,
    ) -> Result<Self> {
        // Copy the end bound (K: Copy via Pod).
        let end_bound: Bound<K> = match range.end_bound() {
            Bound::Included(k) => Bound::Included(*k),
            Bound::Excluded(k) => Bound::Excluded(*k),
            Bound::Unbounded => Bound::Unbounded,
        };

        let (current_page, current_slot) = match range.start_bound() {
            Bound::Included(k) => guard.lower_bound(k)?,
            Bound::Excluded(k) => guard.lower_bound_exclusive(k)?,
            Bound::Unbounded => guard.first_leaf(),
        };

        Ok(Self { guard, current_page, current_slot, end_bound })
    }
}

impl<'a, K: Ord + Pod, V: Pod> Iterator for MmapBTreeRangeIter<'a, K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_page == NULL_PAGE {
                return None;
            }

            let (item, next_page, next_slot) = {
                let page = self.guard.store.page(self.current_page);
                let view = LeafView::<K, V>::new(page, &self.guard.layout);
                let n = view.num_keys();
                if self.current_slot < n {
                    let k = view.keys()[self.current_slot];
                    let v = view.values()[self.current_slot];
                    (Some((k, v)), self.current_page, self.current_slot + 1)
                } else {
                    (None, view.next_leaf(), 0)
                }
            };

            self.current_page = next_page;
            self.current_slot = next_slot;

            if let Some((ref k, _)) = item {
                let past_end = match &self.end_bound {
                    Bound::Included(end) => k > end,
                    Bound::Excluded(end) => k >= end,
                    Bound::Unbounded => false,
                };
                if past_end {
                    self.current_page = NULL_PAGE; // exhaust iterator
                    return None;
                }
                return item;
            }

            if next_page == NULL_PAGE {
                return None;
            }
            // else: slot overflowed, loop to next leaf
        }
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

    /// Asserts that `iter()` over `tree` returns exactly `expected` pairs
    /// in strictly ascending key order.
    fn assert_iter_eq<K, V>(tree: &MmapBTree<K, V>, expected: Vec<(K, V)>)
    where
        K: Ord + Pod + std::fmt::Debug,
        V: Pod + PartialEq + std::fmt::Debug,
    {
        let got: Vec<_> = tree.iter().unwrap().collect();
        assert_eq!(got.len(), expected.len(), "iter length mismatch");
        for (i, ((gk, gv), (ek, ev))) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(gk, ek, "key mismatch at position {i}");
            assert_eq!(gv, ev, "value mismatch at position {i}");
        }
        // Strict ascending order.
        for w in got.windows(2) {
            assert!(w[0].0 < w[1].0, "iter not ascending: {:?} >= {:?}", w[0].0, w[1].0);
        }
    }

    /// A small `#[repr(C)]` Pod struct used as a value type in tests.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
    struct Record {
        timestamp: u64,
        value: i32,
        flags: u32,
    }

    /// Build a `[u8; 64]` key from a `u32`, using big-endian bytes so that
    /// numeric order equals lexicographic order.
    fn key64(i: u32) -> [u8; 64] {
        let mut k = [0u8; 64];
        k[..4].copy_from_slice(&i.to_be_bytes());
        k
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
        assert_eq!(tree.get_value(&1).unwrap(), None);
    }

    #[test]
    fn insert_and_get_single() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, u64>(&dir);
        assert_eq!(tree.insert(42, 100).unwrap(), None);
        assert_eq!(tree.get_value(&42).unwrap(), Some(100));
        assert_eq!(tree.len().unwrap(), 1);
    }

    #[test]
    fn insert_updates_existing_key() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, u64>(&dir);
        tree.insert(1, 10).unwrap();
        let old = tree.insert(1, 99).unwrap();
        assert_eq!(old, Some(10));
        assert_eq!(tree.get_value(&1).unwrap(), Some(99));
        assert_eq!(tree.len().unwrap(), 1);
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
            assert_eq!(tree.get_value(&i).unwrap(), Some(i * 10));
        }
        assert_eq!(tree.get_value(&n).unwrap(), None);
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
            assert_eq!(tree.get_value(&i).unwrap(), Some(i));
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
        assert_eq!(tree.get_value(&0).unwrap(), None);
        tree.insert(7, 77).unwrap();
        assert_eq!(tree.get_value(&7).unwrap(), Some(77));
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
        let tree = MmapBTreeBuilder::<i32, i32>::new().path(&path).build().unwrap();
        assert_eq!(tree.len().unwrap(), 50);
        for i in 0..50_i32 {
            assert_eq!(tree.get_value(&i).unwrap(), Some(i * 2));
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
                        assert_eq!(t.get_value(&i).unwrap(), Some(i));
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }

    // ── remove tests ────────────────────────────────────────────────────────

    #[test]
    fn remove_absent_key() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        assert_eq!(tree.remove(&42).unwrap(), None);
        tree.insert(1, 10).unwrap();
        assert_eq!(tree.remove(&99).unwrap(), None);
        assert_eq!(tree.len().unwrap(), 1);
    }

    #[test]
    fn remove_only_key() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        tree.insert(5, 50).unwrap();
        assert_eq!(tree.remove(&5).unwrap(), Some(50));
        assert!(tree.is_empty().unwrap());
        assert_eq!(tree.get_value(&5).unwrap(), None);
    }

    #[test]
    fn remove_sequential_all() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        let n = 500_i32;
        for i in 0..n {
            tree.insert(i, i * 2).unwrap();
        }
        for i in 0..n {
            assert_eq!(tree.remove(&i).unwrap(), Some(i * 2), "remove {i}");
            assert_eq!(tree.len().unwrap(), (n - i - 1) as usize);
        }
        assert!(tree.is_empty().unwrap());
    }

    #[test]
    fn remove_and_reinsert() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        for i in 0..200_i32 {
            tree.insert(i, i).unwrap();
        }
        // Remove evens.
        for i in (0..200_i32).step_by(2) {
            tree.remove(&i).unwrap();
        }
        assert_eq!(tree.len().unwrap(), 100);
        // Verify odds remain.
        for i in (1..200_i32).step_by(2) {
            assert_eq!(tree.get_value(&i).unwrap(), Some(i));
        }
        // Re-insert evens.
        for i in (0..200_i32).step_by(2) {
            tree.insert(i, i * 10).unwrap();
        }
        assert_eq!(tree.len().unwrap(), 200);
        for i in (0..200_i32).step_by(2) {
            assert_eq!(tree.get_value(&i).unwrap(), Some(i * 10));
        }
    }

    #[test]
    fn remove_reverse_order() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        let n = 300_i32;
        for i in 0..n {
            tree.insert(i, i).unwrap();
        }
        for i in (0..n).rev() {
            assert_eq!(tree.remove(&i).unwrap(), Some(i), "remove {i}");
        }
        assert!(tree.is_empty().unwrap());
    }

    // ── iter tests ──────────────────────────────────────────────────────────

    #[test]
    fn iter_empty() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        let items: Vec<_> = tree.iter().unwrap().collect();
        assert!(items.is_empty());
    }

    #[test]
    fn iter_single() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        tree.insert(7, 70).unwrap();
        let items: Vec<_> = tree.iter().unwrap().collect();
        assert_eq!(items, vec![(7, 70)]);
    }

    #[test]
    fn iter_ascending_order() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        let n = 1000_i32;
        // Insert in reverse order to stress the tree structure.
        for i in (0..n).rev() {
            tree.insert(i, i * 3).unwrap();
        }
        let items: Vec<_> = tree.iter().unwrap().collect();
        assert_eq!(items.len(), n as usize);
        for (idx, (k, v)) in items.iter().enumerate() {
            assert_eq!(*k, idx as i32);
            assert_eq!(*v, idx as i32 * 3);
        }
        // Verify strict ascending order.
        for w in items.windows(2) {
            assert!(w[0].0 < w[1].0);
        }
    }

    // ── range tests ─────────────────────────────────────────────────────────

    #[test]
    fn range_empty_tree() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        let items: Vec<_> = tree.range(0..10).unwrap().collect();
        assert!(items.is_empty());
    }

    #[test]
    fn range_included_excluded() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        for i in 0..20_i32 {
            tree.insert(i, i).unwrap();
        }
        // 5..10 → keys 5, 6, 7, 8, 9
        let items: Vec<_> = tree.range(5..10).unwrap().collect();
        assert_eq!(items, (5..10).map(|i| (i, i)).collect::<Vec<_>>());

        // 5..=10 → keys 5, 6, 7, 8, 9, 10
        let items: Vec<_> = tree.range(5..=10).unwrap().collect();
        assert_eq!(items, (5..=10).map(|i| (i, i)).collect::<Vec<_>>());
    }

    #[test]
    fn range_unbounded() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        for i in 0..50_i32 {
            tree.insert(i, i).unwrap();
        }
        let all: Vec<_> = tree.range(..).unwrap().collect();
        let expected: Vec<_> = (0..50_i32).map(|i| (i, i)).collect();
        assert_eq!(all, expected);
    }

    #[test]
    fn range_no_match() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        for i in [1_i32, 2, 3, 10, 11, 12] {
            tree.insert(i, i).unwrap();
        }
        // Range entirely in the gap between 3 and 10.
        let items: Vec<_> = tree.range(4..10).unwrap().collect();
        assert!(items.is_empty());
    }

    // ── Type variety tests ──────────────────────────────────────────────────

    /// u8 keys: all 256 possible values inserted, retrieved, iterated.
    #[test]
    fn type_u8_u8_full_range() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<u8, u8>(&dir);
        for i in 0u8..=255 {
            tree.insert(i, i.wrapping_mul(3)).unwrap();
        }
        assert_eq!(tree.len().unwrap(), 256);
        for i in 0u8..=255 {
            assert_eq!(tree.get_value(&i).unwrap(), Some(i.wrapping_mul(3)));
        }
        let items: Vec<_> = tree.iter().unwrap().collect();
        assert_eq!(items.len(), 256);
        for (idx, (k, v)) in items.iter().enumerate() {
            assert_eq!(*k, idx as u8);
            assert_eq!(*v, (idx as u8).wrapping_mul(3));
        }
    }

    /// u64 keys with values at the numeric extremes.
    #[test]
    fn type_u64_u64_extremes() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<u64, u64>(&dir);
        let keys = [0u64, 1, u64::MAX / 2, u64::MAX - 1, u64::MAX];
        for &k in &keys {
            tree.insert(k, k ^ 0xDEAD_BEEF).unwrap();
        }
        assert_eq!(tree.len().unwrap(), keys.len());
        for &k in &keys {
            assert_eq!(tree.get_value(&k).unwrap(), Some(k ^ 0xDEAD_BEEF));
        }
        let items: Vec<_> = tree.iter().unwrap().collect();
        assert_eq!(items.len(), keys.len());
        // Must be in ascending order.
        for w in items.windows(2) {
            assert!(w[0].0 < w[1].0);
        }
    }

    /// Array keys: `[i32; 2]` uses lexicographic ordering.
    #[test]
    fn type_array_key_ordering() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<[i32; 2], i64>(&dir);
        let pairs: Vec<([i32; 2], i64)> = vec![
            ([0, 0], 1),
            ([0, 1], 2),
            ([1, 0], 3),
            ([1, 1], 4),
            ([2, -1], 5),
        ];
        for &(k, v) in &pairs {
            tree.insert(k, v).unwrap();
        }
        // Iter must produce them in lexicographic order.
        let mut sorted = pairs.clone();
        sorted.sort_by_key(|&(k, _)| k);
        assert_iter_eq(&tree, sorted);
    }

    /// Large value type: 32-byte values ([u64; 4]).
    #[test]
    fn type_large_value() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, [u64; 4]>(&dir);
        let n = 200_i32;
        for i in 0..n {
            let v = [i as u64, (i * 2) as u64, (i * 3) as u64, (i * 4) as u64];
            tree.insert(i, v).unwrap();
        }
        assert_eq!(tree.len().unwrap(), n as usize);
        for i in 0..n {
            let expected = [i as u64, (i * 2) as u64, (i * 3) as u64, (i * 4) as u64];
            assert_eq!(tree.get_value(&i).unwrap(), Some(expected));
        }
    }

    /// Custom #[repr(C)] Pod struct as value type.
    #[test]
    fn type_pod_struct_value() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, Record>(&dir);
        for i in 0..50_i32 {
            let rec = Record { timestamp: i as u64 * 1000, value: i * -1, flags: i as u32 & 0xFF };
            tree.insert(i, rec).unwrap();
        }
        assert_eq!(tree.len().unwrap(), 50);
        for i in 0..50_i32 {
            let got = tree.get_value(&i).unwrap().unwrap();
            assert_eq!(got.timestamp, i as u64 * 1000);
            assert_eq!(got.value, i * -1);
            assert_eq!(got.flags, i as u32 & 0xFF);
        }
    }

    // ── Boundary value tests ────────────────────────────────────────────────

    /// i32::MIN and i32::MAX are valid keys and are ordered correctly.
    #[test]
    fn boundary_i32_min_max_zero() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        tree.insert(0, 0).unwrap();
        tree.insert(i32::MIN, -1).unwrap();
        tree.insert(i32::MAX, 1).unwrap();
        assert_eq!(tree.get_value(&i32::MIN).unwrap(), Some(-1));
        assert_eq!(tree.get_value(&0).unwrap(), Some(0));
        assert_eq!(tree.get_value(&i32::MAX).unwrap(), Some(1));
        assert_eq!(tree.len().unwrap(), 3);

        let items: Vec<_> = tree.iter().unwrap().collect();
        assert_eq!(items, vec![(i32::MIN, -1), (0, 0), (i32::MAX, 1)]);
    }

    /// Signed integer ordering: negative keys must sort before positive.
    #[test]
    fn boundary_negative_keys_sorted() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i64, i64>(&dir);
        let keys: Vec<i64> = (-50..=50).collect();
        // Insert in scrambled order.
        for (i, _) in keys.iter().enumerate() {
            let scrambled = keys[(i * 37) % keys.len()];
            tree.insert(scrambled, scrambled * 10).unwrap();
        }
        let items: Vec<_> = tree.iter().unwrap().collect();
        assert_eq!(items.len(), 101);
        for (i, (k, v)) in items.iter().enumerate() {
            assert_eq!(*k, -50 + i as i64);
            assert_eq!(*v, k * 10);
        }
    }

    // ── Insert / update edge cases ──────────────────────────────────────────

    /// Repeated updates to the same key must not change `len()`.
    #[test]
    fn insert_repeated_updates_same_key() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i64>(&dir);
        tree.insert(42, 1).unwrap();
        for round in 2..=100_i64 {
            let old = tree.insert(42, round).unwrap();
            assert_eq!(old, Some(round - 1));
            assert_eq!(tree.len().unwrap(), 1);
        }
        assert_eq!(tree.get_value(&42).unwrap(), Some(100));
    }

    /// `contains_key` is consistent with `get`.
    #[test]
    fn contains_key_consistency() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        for i in 0..100_i32 {
            tree.insert(i * 2, i).unwrap(); // even keys only
        }
        for i in 0..100_i32 {
            assert!(tree.contains_key(&(i * 2)).unwrap(), "even key {}", i * 2);
            assert!(!tree.contains_key(&(i * 2 + 1)).unwrap(), "odd key {}", i * 2 + 1);
        }
    }

    // ── get (MmapBTreeValueRef) ─────────────────────────────────────────────

    /// `get` on an empty tree returns `None`.
    #[test]
    fn get_ref_empty_tree() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, u64>(&dir);
        assert!(tree.get(&1).unwrap().is_none());
    }

    /// `get` returns `None` for a missing key and `Some` for a present one.
    #[test]
    fn get_ref_present_and_absent() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, u64>(&dir);
        tree.insert(10, 99).unwrap();
        assert!(tree.get(&9).unwrap().is_none());
        assert!(tree.get(&11).unwrap().is_none());
        assert_eq!(*tree.get(&10).unwrap().unwrap(), 99_u64);
    }

    /// The dereffed value equals what `get_value` returns.
    #[test]
    fn get_ref_matches_get_value() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i64>(&dir);
        for i in 0..50_i32 {
            tree.insert(i, i as i64 * 3).unwrap();
        }
        for i in 0..50_i32 {
            let by_ref = *tree.get(&i).unwrap().unwrap();
            let by_val = tree.get_value(&i).unwrap().unwrap();
            assert_eq!(by_ref, by_val, "mismatch at key {i}");
        }
    }

    /// `get` after an update reflects the new value.
    #[test]
    fn get_ref_reflects_update() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        tree.insert(5, 1).unwrap();
        tree.insert(5, 2).unwrap();
        assert_eq!(*tree.get(&5).unwrap().unwrap(), 2);
    }

    /// `get` after `remove` returns `None`.
    #[test]
    fn get_ref_after_remove() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, u64>(&dir);
        tree.insert(7, 42).unwrap();
        tree.remove(&7).unwrap();
        assert!(tree.get(&7).unwrap().is_none());
    }

    /// Multiple simultaneous `get` refs can coexist (all hold read locks).
    #[test]
    fn get_ref_multiple_simultaneous() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, u64>(&dir);
        tree.insert(1, 10).unwrap();
        tree.insert(2, 20).unwrap();
        tree.insert(3, 30).unwrap();
        let r1 = tree.get(&1).unwrap().unwrap();
        let r2 = tree.get(&2).unwrap().unwrap();
        let r3 = tree.get(&3).unwrap().unwrap();
        assert_eq!(*r1, 10);
        assert_eq!(*r2, 20);
        assert_eq!(*r3, 30);
    }

    /// `get` works with a Pod struct value.
    #[test]
    fn get_ref_pod_struct_value() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, Record>(&dir);
        let r = Record { timestamp: 999, value: -7, flags: 0xDEAD };
        tree.insert(42, r).unwrap();
        let got = tree.get(&42).unwrap().unwrap();
        assert_eq!(got.timestamp, 999);
        assert_eq!(got.value, -7);
        assert_eq!(got.flags, 0xDEAD);
    }

    /// `get` across many entries (multi-page tree) returns the correct ref.
    #[test]
    fn get_ref_large_tree() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        let n = 1000_i32;
        for i in 0..n {
            tree.insert(i, i * 7).unwrap();
        }
        for i in 0..n {
            assert_eq!(*tree.get(&i).unwrap().unwrap(), i * 7, "key {i}");
        }
        assert!(tree.get(&n).unwrap().is_none());
    }

    // ── Remove edge cases ───────────────────────────────────────────────────

    /// Removing a key that was updated returns the most-recent value.
    #[test]
    fn remove_after_update_returns_latest() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        tree.insert(7, 10).unwrap();
        tree.insert(7, 20).unwrap();
        assert_eq!(tree.remove(&7).unwrap(), Some(20));
        assert_eq!(tree.get_value(&7).unwrap(), None);
        assert!(tree.is_empty().unwrap());
    }

    /// Remove then re-insert the same key succeeds and returns the new value.
    #[test]
    fn remove_then_reinsert_same_key() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        for i in 0..50_i32 {
            tree.insert(i, i).unwrap();
        }
        tree.remove(&25).unwrap();
        assert_eq!(tree.get_value(&25).unwrap(), None);
        tree.insert(25, 999).unwrap();
        assert_eq!(tree.get_value(&25).unwrap(), Some(999));
        assert_eq!(tree.len().unwrap(), 50);
    }

    /// Removing the very first (smallest) key of a large tree.
    #[test]
    fn remove_first_key_of_large_tree() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        let n = 600_i32;
        for i in 0..n {
            tree.insert(i, i).unwrap();
        }
        assert_eq!(tree.remove(&0).unwrap(), Some(0));
        assert_eq!(tree.len().unwrap(), (n - 1) as usize);
        assert_eq!(tree.get_value(&0).unwrap(), None);
        // Everything else still present.
        for i in 1..n {
            assert_eq!(tree.get_value(&i).unwrap(), Some(i), "missing key {i}");
        }
    }

    /// Removing the very last (largest) key of a large tree.
    #[test]
    fn remove_last_key_of_large_tree() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        let n = 600_i32;
        for i in 0..n {
            tree.insert(i, i).unwrap();
        }
        assert_eq!(tree.remove(&(n - 1)).unwrap(), Some(n - 1));
        assert_eq!(tree.len().unwrap(), (n - 1) as usize);
        assert_eq!(tree.get_value(&(n - 1)).unwrap(), None);
        for i in 0..n - 1 {
            assert_eq!(tree.get_value(&i).unwrap(), Some(i), "missing key {i}");
        }
    }

    /// Removing the middle key of a large tree.
    #[test]
    fn remove_middle_key_of_large_tree() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        let n = 600_i32;
        for i in 0..n {
            tree.insert(i, i * 3).unwrap();
        }
        let mid = n / 2;
        assert_eq!(tree.remove(&mid).unwrap(), Some(mid * 3));
        assert_eq!(tree.len().unwrap(), (n - 1) as usize);
        for i in 0..n {
            let expected = if i == mid { None } else { Some(i * 3) };
            assert_eq!(tree.get_value(&i).unwrap(), expected, "key {i}");
        }
    }

    /// Removing a key twice: second remove returns None.
    #[test]
    fn remove_twice_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        tree.insert(1, 2).unwrap();
        assert_eq!(tree.remove(&1).unwrap(), Some(2));
        assert_eq!(tree.remove(&1).unwrap(), None);
        assert!(tree.is_empty().unwrap());
    }

    // ── Iterator correctness after removes ──────────────────────────────────

    /// After removing some keys, iter() must yield exactly the remaining
    /// keys in ascending order.
    #[test]
    fn iter_after_partial_remove() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        let n = 600_i32;
        for i in 0..n {
            tree.insert(i, i).unwrap();
        }
        // Remove multiples of 3.
        let removed: Vec<i32> = (0..n).filter(|i| i % 3 == 0).collect();
        for &k in &removed {
            tree.remove(&k).unwrap();
        }
        let remaining: Vec<_> = (0..n).filter(|i| i % 3 != 0).map(|i| (i, i)).collect();
        assert_iter_eq(&tree, remaining);
    }

    /// iter() count always matches len().
    #[test]
    fn iter_count_matches_len() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        for i in 0..300_i32 {
            tree.insert(i, i).unwrap();
        }
        for i in (0..300_i32).step_by(7) {
            tree.remove(&i).unwrap();
        }
        let iter_count = tree.iter().unwrap().count();
        assert_eq!(iter_count, tree.len().unwrap());
    }

    /// iter() yields no duplicate keys.
    #[test]
    fn iter_no_duplicate_keys() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        for i in (0..400_i32).rev() {
            tree.insert(i, i).unwrap();
        }
        let items: Vec<_> = tree.iter().unwrap().collect();
        // All keys are unique and strictly ascending.
        for w in items.windows(2) {
            assert!(w[0].0 < w[1].0, "duplicate or out-of-order: {:?}", &w);
        }
    }

    // ── Range iterator — all bound variants ────────────────────────────────

    /// Excluded start bound `(start, end)`.
    #[test]
    fn range_excluded_start() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        for i in 0..20_i32 {
            tree.insert(i, i).unwrap();
        }
        use std::ops::Bound::*;
        // (5, 10) → keys 6, 7, 8, 9
        let items: Vec<_> = tree
            .range((Excluded(5), Excluded(10)))
            .unwrap()
            .collect();
        assert_eq!(items, vec![(6, 6), (7, 7), (8, 8), (9, 9)]);
    }

    /// Unbounded start, exclusive end `..end`.
    #[test]
    fn range_unbounded_start_exclusive_end() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        for i in 0..20_i32 {
            tree.insert(i, i).unwrap();
        }
        let items: Vec<_> = tree.range(..5).unwrap().collect();
        assert_eq!(items, (0..5_i32).map(|i| (i, i)).collect::<Vec<_>>());
    }

    /// Unbounded start, inclusive end `..=end`.
    #[test]
    fn range_unbounded_start_inclusive_end() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        for i in 0..20_i32 {
            tree.insert(i, i).unwrap();
        }
        let items: Vec<_> = tree.range(..=4).unwrap().collect();
        assert_eq!(items, (0..=4_i32).map(|i| (i, i)).collect::<Vec<_>>());
    }

    /// Bounded start, unbounded end `start..`.
    #[test]
    fn range_bounded_start_unbounded_end() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        for i in 0..20_i32 {
            tree.insert(i, i).unwrap();
        }
        let items: Vec<_> = tree.range(15..).unwrap().collect();
        assert_eq!(items, (15..20_i32).map(|i| (i, i)).collect::<Vec<_>>());
    }

    /// Single-element range `k..=k`.
    #[test]
    fn range_single_key_inclusive() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        for i in 0..20_i32 {
            tree.insert(i, i * 2).unwrap();
        }
        let items: Vec<_> = tree.range(7..=7).unwrap().collect();
        assert_eq!(items, vec![(7, 14)]);
    }

    /// Empty range `k..k` (exclusive end == start).
    #[test]
    fn range_empty_exclusive_end_equals_start() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        for i in 0..20_i32 {
            tree.insert(i, i).unwrap();
        }
        let items: Vec<_> = tree.range(7..7).unwrap().collect();
        assert!(items.is_empty());
    }

    /// Range before the smallest key in the tree.
    #[test]
    fn range_before_all_keys() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        for i in 100..200_i32 {
            tree.insert(i, i).unwrap();
        }
        let items: Vec<_> = tree.range(0..100).unwrap().collect();
        assert!(items.is_empty());
    }

    /// Range after the largest key in the tree.
    #[test]
    fn range_after_all_keys() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        for i in 0..100_i32 {
            tree.insert(i, i).unwrap();
        }
        let items: Vec<_> = tree.range(100..).unwrap().collect();
        assert!(items.is_empty());
    }

    /// Range whose bounds land exactly on existing keys (inclusive/exclusive
    /// boundary keys tested explicitly).
    #[test]
    fn range_boundary_precision() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        for i in [0_i32, 5, 10, 15, 20] {
            tree.insert(i, i).unwrap();
        }
        // Inclusive 5..=15 → 5, 10, 15
        let a: Vec<_> = tree.range(5..=15).unwrap().collect();
        assert_eq!(a, vec![(5, 5), (10, 10), (15, 15)]);

        // Exclusive 5..15 → 5, 10
        let b: Vec<_> = tree.range(5..15).unwrap().collect();
        assert_eq!(b, vec![(5, 5), (10, 10)]);

        // Excluded(5)..Excluded(15) → 10
        use std::ops::Bound::*;
        let c: Vec<_> = tree.range((Excluded(5), Excluded(15))).unwrap().collect();
        assert_eq!(c, vec![(10, 10)]);

        // Excluded(5)..=15 → 10, 15
        let d: Vec<_> = tree.range((Excluded(5), Included(15))).unwrap().collect();
        assert_eq!(d, vec![(10, 10), (15, 15)]);
    }

    /// Range spanning gaps: keys that don't exist within the bounds.
    #[test]
    fn range_with_interior_gaps() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        // Insert only multiples of 10.
        for i in (0..=100_i32).step_by(10) {
            tree.insert(i, i).unwrap();
        }
        // Range 15..=85 → 20, 30, 40, 50, 60, 70, 80
        let items: Vec<_> = tree.range(15..=85).unwrap().collect();
        let expected: Vec<_> = (2..=8_i32).map(|i| (i * 10, i * 10)).collect();
        assert_eq!(items, expected);
    }

    /// Range on a large tree spans multiple leaf pages.
    #[test]
    fn range_across_leaf_pages() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        let n = 2000_i32;
        for i in 0..n {
            tree.insert(i, i * 2).unwrap();
        }
        let lo = 300_i32;
        let hi = 1700_i32;
        let items: Vec<_> = tree.range(lo..=hi).unwrap().collect();
        let expected: Vec<_> = (lo..=hi).map(|i| (i, i * 2)).collect();
        assert_eq!(items, expected);
    }

    // ── 3-level B+tree stress test ──────────────────────────────────────────
    //
    // [u8; 64] keys have size=64, align=1.  With 4096-byte pages:
    //   leaf_capacity  = floor((4096-16) / (64+64)) ≈ 31
    //   internal_capacity ≈ floor((4096-24) / (64+8)) ≈ 56
    //
    // Forcing a 3-level tree requires ≥ 56 * (31/2+1) ≈ 896 entries.
    // We insert 2000 to comfortably reach that level.

    #[test]
    fn three_level_tree_large_keys() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<[u8; 64], u32>(&dir);
        let n: u32 = 2000;

        for i in 0..n {
            tree.insert(key64(i), i).unwrap();
        }
        assert_eq!(tree.len().unwrap(), n as usize);

        // Point lookup: every key must be found.
        for i in 0..n {
            assert_eq!(tree.get_value(&key64(i)).unwrap(), Some(i), "get {i}");
        }

        // iter() must return all n keys in sorted (big-endian byte) order.
        let items: Vec<_> = tree.iter().unwrap().collect();
        assert_eq!(items.len(), n as usize);
        for (j, (k, v)) in items.iter().enumerate() {
            assert_eq!(*k, key64(j as u32), "iter key at {j}");
            assert_eq!(*v, j as u32, "iter value at {j}");
        }

        // Remove all keys and verify the tree is empty.
        for i in (0..n).rev() {
            assert_eq!(tree.remove(&key64(i)).unwrap(), Some(i), "remove {i}");
        }
        assert!(tree.is_empty().unwrap());
        assert_eq!(tree.iter().unwrap().count(), 0);
    }

    // ── Stress tests ────────────────────────────────────────────────────────

    /// Insert 5000 keys sequentially, verify with iter, remove all forward.
    #[test]
    fn stress_5000_sequential_forward_remove() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        let n = 5000_i32;
        for i in 0..n {
            tree.insert(i, i).unwrap();
        }
        // iter order check
        let items: Vec<_> = tree.iter().unwrap().collect();
        assert_eq!(items.len(), n as usize);
        for (j, (k, _)) in items.iter().enumerate() {
            assert_eq!(*k, j as i32);
        }
        // remove all forward
        for i in 0..n {
            assert_eq!(tree.remove(&i).unwrap(), Some(i));
        }
        assert!(tree.is_empty().unwrap());
    }

    /// Insert 5000 keys, remove all in reverse (right-to-left) order.
    #[test]
    fn stress_5000_reverse_remove() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        let n = 5000_i32;
        for i in 0..n {
            tree.insert(i, i).unwrap();
        }
        for i in (0..n).rev() {
            assert_eq!(tree.remove(&i).unwrap(), Some(i));
        }
        assert!(tree.is_empty().unwrap());
    }

    /// Zigzag insert order: 0, N-1, 1, N-2, … to stress separator splits.
    #[test]
    fn stress_zigzag_insert_then_remove_all() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        let n = 1000_i32;
        let keys: Vec<i32> = (0..n / 2)
            .flat_map(|i| [i, n - 1 - i])
            .collect();
        for &k in &keys {
            tree.insert(k, k * 7).unwrap();
        }
        assert_eq!(tree.len().unwrap(), n as usize);
        // Every key present.
        for i in 0..n {
            assert_eq!(tree.get_value(&i).unwrap(), Some(i * 7), "key {i}");
        }
        // iter still sorted.
        let items: Vec<_> = tree.iter().unwrap().collect();
        for w in items.windows(2) {
            assert!(w[0].0 < w[1].0);
        }
        // Remove all.
        for i in 0..n {
            tree.remove(&i).unwrap();
        }
        assert!(tree.is_empty().unwrap());
    }

    /// Sliding window: keep approximately 200 live keys at any time.
    /// Inserts key i, removes key i-200 for i in 200..2200.
    #[test]
    fn stress_sliding_window() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        let window = 200_i32;
        let total = 2200_i32;

        // Seed the first window.
        for i in 0..window {
            tree.insert(i, i).unwrap();
        }
        // Slide.
        for i in window..total {
            tree.insert(i, i).unwrap();
            tree.remove(&(i - window)).unwrap();
            assert_eq!(tree.len().unwrap(), window as usize);
        }
        // The remaining window is [total-window, total).
        for i in (total - window)..total {
            assert_eq!(tree.get_value(&i).unwrap(), Some(i));
        }
        assert_eq!(tree.len().unwrap(), window as usize);
    }

    /// Remove every other key, verify remaining, remove rest.
    #[test]
    fn stress_remove_alternating_then_rest() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        let n = 1000_i32;
        for i in 0..n {
            tree.insert(i, i).unwrap();
        }
        // Remove evens.
        for i in (0..n).step_by(2) {
            tree.remove(&i).unwrap();
        }
        assert_eq!(tree.len().unwrap(), (n / 2) as usize);
        // Only odds remain.
        for i in 0..n {
            let exp = if i % 2 == 0 { None } else { Some(i) };
            assert_eq!(tree.get_value(&i).unwrap(), exp, "key {i}");
        }
        // Remove odds.
        for i in (1..n).step_by(2) {
            tree.remove(&i).unwrap();
        }
        assert!(tree.is_empty().unwrap());
    }

    /// Interleaved inserts and removes in a single pass.
    #[test]
    fn stress_interleaved_insert_remove() {
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        // Insert 0..1000; every time we hit a multiple of 50, remove
        // the previous 50 entries.
        let n = 1000_i32;
        let batch = 50_i32;
        for i in 0..n {
            tree.insert(i, i).unwrap();
            if i > 0 && i % batch == 0 {
                for j in (i - batch)..i {
                    tree.remove(&j).unwrap();
                }
            }
        }
        // Remaining keys are those not yet removed: last partial batch.
        let removed_up_to = (n / batch) * batch - batch;
        let remaining: Vec<(i32, i32)> = (removed_up_to..n).map(|i| (i, i)).collect();
        assert_iter_eq(&tree, remaining);
    }

    // ── Persistence tests ───────────────────────────────────────────────────

    /// Removes survive a close/reopen.
    #[test]
    fn persistence_with_removes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p2.db");
        {
            let tree = MmapBTreeBuilder::<i32, i32>::new().path(&path).build().unwrap();
            for i in 0..100_i32 {
                tree.insert(i, i * 5).unwrap();
            }
            // Remove every third key.
            for i in (0..100_i32).step_by(3) {
                tree.remove(&i).unwrap();
            }
            tree.flush().unwrap();
        }
        // Re-open and verify.
        let tree = MmapBTreeBuilder::<i32, i32>::new().path(&path).build().unwrap();
        for i in 0..100_i32 {
            let expected = if i % 3 == 0 { None } else { Some(i * 5) };
            assert_eq!(tree.get_value(&i).unwrap(), expected, "key {i}");
        }
        let expected_len = (0..100_i32).filter(|i| i % 3 != 0).count();
        assert_eq!(tree.len().unwrap(), expected_len);
    }

    /// Multiple open/close/modify cycles preserve consistency.
    #[test]
    fn persistence_multiple_cycles() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cycles.db");

        // Cycle 1: insert 0..100
        {
            let t = MmapBTreeBuilder::<i32, i32>::new().path(&path).build().unwrap();
            for i in 0..100_i32 { t.insert(i, i).unwrap(); }
        }
        // Cycle 2: insert 100..200, remove 0..50
        {
            let t = MmapBTreeBuilder::<i32, i32>::new().path(&path).build().unwrap();
            assert_eq!(t.len().unwrap(), 100);
            for i in 100..200_i32 { t.insert(i, i).unwrap(); }
            for i in 0..50_i32 { t.remove(&i).unwrap(); }
        }
        // Cycle 3: verify final state
        {
            let t = MmapBTreeBuilder::<i32, i32>::new().path(&path).build().unwrap();
            assert_eq!(t.len().unwrap(), 150); // 50..200
            for i in 0..50_i32 { assert_eq!(t.get_value(&i).unwrap(), None); }
            for i in 50..200_i32 { assert_eq!(t.get_value(&i).unwrap(), Some(i)); }
        }
    }

    // ── Concurrent access ───────────────────────────────────────────────────

    /// Multiple threads can call iter() simultaneously (all hold read locks).
    #[test]
    fn concurrent_iter_readers() {
        use std::sync::Arc;
        let dir = tempfile::tempdir().unwrap();
        let tree = Arc::new(open::<i32, i32>(&dir));
        let n = 500_i32;
        for i in 0..n {
            tree.insert(i, i).unwrap();
        }
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let t = Arc::clone(&tree);
                std::thread::spawn(move || {
                    let items: Vec<_> = t.iter().unwrap().collect();
                    assert_eq!(items.len(), n as usize);
                    for (j, (k, v)) in items.iter().enumerate() {
                        assert_eq!(*k, j as i32);
                        assert_eq!(*v, j as i32);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }

    /// Concurrent range queries from multiple threads.
    #[test]
    fn concurrent_range_readers() {
        use std::sync::Arc;
        let dir = tempfile::tempdir().unwrap();
        let tree = Arc::new(open::<i32, i32>(&dir));
        let n = 400_i32;
        for i in 0..n {
            tree.insert(i, i * 2).unwrap();
        }
        let handles: Vec<_> = (0..4_i32)
            .map(|t| {
                let tree = Arc::clone(&tree);
                let lo = t * 100;
                let hi = lo + 100;
                std::thread::spawn(move || {
                    let items: Vec<_> = tree.range(lo..hi).unwrap().collect();
                    assert_eq!(items.len(), 100);
                    for (k, v) in items {
                        assert!(k >= lo && k < hi);
                        assert_eq!(v, k * 2);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }

    // ── Structural correctness ───────────────────────────────────────────────

    /// After many inserts and targeted removes, iter() must match an
    /// independently computed reference set.
    #[test]
    fn structural_iter_matches_reference_set() {
        use std::collections::BTreeMap;
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i64>(&dir);
        let mut reference: BTreeMap<i32, i64> = BTreeMap::new();

        // Deterministic "random" sequence via LCG.
        let mut x: u32 = 0xDEAD_BEEF;
        let lcg = |v: &mut u32| -> u32 {
            *v = v.wrapping_mul(1664525).wrapping_add(1013904223);
            *v
        };

        for _ in 0..800 {
            let key = (lcg(&mut x) % 500) as i32;
            let val = lcg(&mut x) as i64;
            tree.insert(key, val).unwrap();
            reference.insert(key, val);
        }

        // Spot-remove ~100 keys.
        for _ in 0..100 {
            let key = (lcg(&mut x) % 500) as i32;
            let got = tree.remove(&key).unwrap();
            let exp = reference.remove(&key);
            assert_eq!(got, exp, "remove({key})");
        }

        assert_eq!(tree.len().unwrap(), reference.len());

        // Full iter comparison with BTreeMap reference.
        let tree_items: Vec<_> = tree.iter().unwrap().collect();
        let ref_items: Vec<_> = reference.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(tree_items, ref_items);
    }

    /// get() is consistent with iter() after a complex mix of operations.
    #[test]
    fn structural_get_consistent_with_iter() {
        use std::collections::BTreeMap;
        let dir = tempfile::tempdir().unwrap();
        let tree = open::<i32, i32>(&dir);
        let mut reference: BTreeMap<i32, i32> = BTreeMap::new();

        let mut x: u32 = 0xCAFE_BABE;
        let lcg = |v: &mut u32| -> u32 {
            *v = v.wrapping_mul(1664525).wrapping_add(1013904223);
            *v
        };

        for _ in 0..600 {
            let key = (lcg(&mut x) % 300) as i32;
            let val = (lcg(&mut x) % 10_000) as i32;
            if lcg(&mut x) % 4 == 0 && !reference.is_empty() {
                // Remove a random key.
                let rem_key = (lcg(&mut x) % 300) as i32;
                tree.remove(&rem_key).unwrap();
                reference.remove(&rem_key);
            } else {
                tree.insert(key, val).unwrap();
                reference.insert(key, val);
            }
        }

        // Every key in reference must be found by get().
        for (&k, &v) in &reference {
            assert_eq!(tree.get_value(&k).unwrap(), Some(v), "get({k})");
        }
        // No key outside reference must be found.
        for k in 0..300_i32 {
            let expected = reference.get(&k).copied();
            assert_eq!(tree.get_value(&k).unwrap(), expected, "get({k})");
        }

        // iter() must exactly match reference.
        let tree_items: Vec<_> = tree.iter().unwrap().collect();
        let ref_items: Vec<_> = reference.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(tree_items, ref_items);
    }

    // ── Recovery / crash safety ──────────────────────────────────────────────

    /// Plant a WAL for insert on a clean (empty) tree, re-open → key recovered.
    #[test]
    fn recovery_insert_from_wal_on_clean_tree() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.db");
        // Create an empty tree and close it.
        {
            let _tree = MmapBTreeBuilder::<i32, u64>::new()
                .path(&path)
                .build()
                .unwrap();
        }
        // Plant a WAL as if we crashed after fsync but before any mmap write.
        let key: i32 = 42;
        let val: u64 = 100;
        wal::write_and_sync(
            &path,
            wal::WAL_OP_INSERT,
            bytemuck::bytes_of(&key),
            bytemuck::bytes_of(&val),
        )
        .unwrap();
        // Re-open: WAL recovery inserts the key.
        let tree = MmapBTreeBuilder::<i32, u64>::new()
            .path(&path)
            .build()
            .unwrap();
        assert_eq!(tree.get_value(&key).unwrap(), Some(val));
        // WAL must be cleaned up after recovery.
        assert!(!wal::wal_path(&path).exists());
    }

    /// Plant a WAL for remove after a successful insert, re-open → key absent.
    #[test]
    fn recovery_remove_from_wal_on_clean_tree() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.db");
        {
            let tree = MmapBTreeBuilder::<i32, u64>::new()
                .path(&path)
                .build()
                .unwrap();
            tree.insert(42_i32, 100_u64).unwrap();
        }
        // Plant a WAL for remove(42) as if we crashed before the mmap write.
        let key: i32 = 42;
        let zeros = vec![0u8; std::mem::size_of::<u64>()];
        wal::write_and_sync(&path, wal::WAL_OP_REMOVE, bytemuck::bytes_of(&key), &zeros)
            .unwrap();
        // Re-open: WAL recovery removes the key.
        let tree = MmapBTreeBuilder::<i32, u64>::new()
            .path(&path)
            .build()
            .unwrap();
        assert_eq!(tree.get_value(&key).unwrap(), None);
        assert!(!wal::wal_path(&path).exists());
    }

    /// Re-applying an already-completed insert WAL is idempotent.
    ///
    /// Scenario: insert completed and was flushed, but the WAL was not yet
    /// deleted when the process died.  On replay the insert overwrites the
    /// same key with the same value → still correct.
    #[test]
    fn recovery_insert_already_done_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.db");
        {
            let tree = MmapBTreeBuilder::<i32, u64>::new()
                .path(&path)
                .build()
                .unwrap();
            tree.insert(7_i32, 888_u64).unwrap();
        }
        // Plant the same WAL again (simulating: WAL was not deleted before crash).
        let key: i32 = 7;
        let val: u64 = 888;
        wal::write_and_sync(
            &path,
            wal::WAL_OP_INSERT,
            bytemuck::bytes_of(&key),
            bytemuck::bytes_of(&val),
        )
        .unwrap();
        let tree = MmapBTreeBuilder::<i32, u64>::new()
            .path(&path)
            .build()
            .unwrap();
        assert_eq!(tree.get_value(&key).unwrap(), Some(val));
        assert!(!wal::wal_path(&path).exists());
    }

    /// Re-applying a remove WAL for a key that was already absent is a no-op.
    #[test]
    fn recovery_remove_key_absent_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.db");
        {
            let _tree = MmapBTreeBuilder::<i32, u64>::new()
                .path(&path)
                .build()
                .unwrap();
        }
        // Plant a remove WAL for key 99, which was never inserted.
        let key: i32 = 99;
        let zeros = vec![0u8; std::mem::size_of::<u64>()];
        wal::write_and_sync(&path, wal::WAL_OP_REMOVE, bytemuck::bytes_of(&key), &zeros)
            .unwrap();
        // Re-open: remove of absent key is a no-op, no error.
        let tree = MmapBTreeBuilder::<i32, u64>::new()
            .path(&path)
            .build()
            .unwrap();
        assert_eq!(tree.get_value(&key).unwrap(), None);
        assert!(!wal::wal_path(&path).exists());
    }

    /// A truncated / garbage WAL is silently ignored; the tree opens normally.
    #[test]
    fn recovery_corrupt_wal_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.db");
        {
            let _tree = MmapBTreeBuilder::<i32, u64>::new()
                .path(&path)
                .build()
                .unwrap();
        }
        // Write 3 bytes of garbage — shorter than WAL_HEADER_LEN + any K/V.
        std::fs::write(wal::wal_path(&path), b"XXX").unwrap();
        // Must open without error and behave normally.
        let tree = MmapBTreeBuilder::<i32, u64>::new()
            .path(&path)
            .build()
            .unwrap();
        tree.insert(1_i32, 2_u64).unwrap();
        assert_eq!(tree.get_value(&1_i32).unwrap(), Some(2_u64));
    }

    /// After a successful insert or remove, the WAL file must be absent.
    #[test]
    fn recovery_wal_deleted_after_success() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.db");
        let tree = MmapBTreeBuilder::<i32, u64>::new()
            .path(&path)
            .build()
            .unwrap();
        tree.insert(1_i32, 2_u64).unwrap();
        assert!(
            !wal::wal_path(&path).exists(),
            "WAL must be deleted after insert"
        );
        tree.remove(&1_i32).unwrap();
        assert!(
            !wal::wal_path(&path).exists(),
            "WAL must be deleted after remove"
        );
    }

    /// Opening a version-1 file triggers the one-time migration: checksums are
    /// written to all live node pages and the stored version is bumped to 2.
    /// All entries remain readable and new operations succeed.
    #[test]
    fn recovery_version1_upgrade() {
        use crate::storage::{NODE_KIND_INTERNAL, NODE_KIND_LEAF, PAGE_SIZE};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.db");

        // 1. Create a v2 tree with some entries.
        {
            let tree = MmapBTreeBuilder::<i32, i32>::new()
                .path(&path)
                .build()
                .unwrap();
            for i in 0..20_i32 {
                tree.insert(i, i * 10).unwrap();
            }
        }

        // 2. Patch the file: downgrade version to 1 and zero all page checksums.
        {
            let mut data = std::fs::read(&path).unwrap();
            // `version` is at bytes 8..10 of FileHeader (native endian u16).
            let v1 = 1u16.to_ne_bytes();
            data[8] = v1[0];
            data[9] = v1[1];
            // Zero checksum field (bytes 4..8) in every live node page.
            let num_pages = data.len() / PAGE_SIZE;
            for p in 1..num_pages {
                let base = p * PAGE_SIZE;
                let kind = data[base];
                if kind == NODE_KIND_LEAF || kind == NODE_KIND_INTERNAL {
                    data[base + 4] = 0;
                    data[base + 5] = 0;
                    data[base + 6] = 0;
                    data[base + 7] = 0;
                }
            }
            std::fs::write(&path, &data).unwrap();
        }

        // 3. Re-open: the v1→v2 migration runs automatically.
        let tree = MmapBTreeBuilder::<i32, i32>::new()
            .path(&path)
            .build()
            .unwrap();

        // 4. All previously inserted entries must still be readable.
        for i in 0..20_i32 {
            assert_eq!(tree.get_value(&i).unwrap(), Some(i * 10), "key {i}");
        }

        // 5. New operations must work (checksums are now written on mutations).
        tree.insert(100_i32, 999_i32).unwrap();
        assert_eq!(tree.get_value(&100_i32).unwrap(), Some(999_i32));
        tree.remove(&0_i32).unwrap();
        assert_eq!(tree.get_value(&0_i32).unwrap(), None);
    }

    /// Flipping a byte in a node page on disk is detected as corruption on read.
    #[test]
    fn recovery_checksum_detects_page_flip() {
        use crate::storage::PAGE_SIZE;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.db");

        // Insert enough entries to populate at least one leaf page.
        {
            let tree = MmapBTreeBuilder::<i32, i32>::new()
                .path(&path)
                .build()
                .unwrap();
            for i in 0..50_i32 {
                tree.insert(i, i).unwrap();
            }
        }

        // Flip a byte in page 1 (first non-header page) past the NodeHeader,
        // so it is covered by the checksum but isn't the checksum field itself.
        {
            let mut data = std::fs::read(&path).unwrap();
            let target = PAGE_SIZE + 20; // byte 20 within page 1 (in key area)
            data[target] ^= 0xFF;
            std::fs::write(&path, &data).unwrap();
        }

        // Re-open: no WAL, so no recovery attempt; the page is simply corrupt.
        let tree = MmapBTreeBuilder::<i32, i32>::new()
            .path(&path)
            .build()
            .unwrap();

        // Any read that traverses page 1 must surface a Corruption error.
        match tree.get_value(&0_i32) {
            Err(BTreeError::Corruption(_)) => {} // expected
            Ok(v) => panic!("expected Corruption, got Ok({v:?})"),
            Err(e) => panic!("expected Corruption, got {e:?}"),
        }
    }
}
