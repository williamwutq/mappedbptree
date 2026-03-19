//! On-disk storage format and memory-mapped page management.
//!
//! # File layout
//!
//! ```text
//! Page 0:  FileHeader  (PAGE_SIZE bytes, only first 64 bytes are meaningful)
//! Page 1:  node page
//! Page 2:  node page
//! ...
//! ```
//!
//! Page index `0` is reserved for the file header and is never used as a
//! node page. It therefore serves as a natural null/sentinel value:
//! `NULL_PAGE = 0`.
//!
//! # Node page layout
//!
//! Every node page starts with a [`NodeHeader`] (16 bytes), followed by
//! keys, then values (leaf) or child page indices (internal). Offsets are
//! computed by [`NodeLayout`] to satisfy alignment requirements of `K` and
//! `V`.

use std::fs::{File, OpenOptions};
use std::path::Path;

use bytemuck::{Pod, Zeroable};
use memmap2::MmapMut;

use crate::tree::{BTreeError, Result};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// ASCII magic bytes written into every file header.
pub const MAGIC: [u8; 8] = *b"MMBPTREE";

/// Current on-disk format version.
pub const VERSION: u16 = 1;

/// Sentinel page index meaning "no page" (null pointer equivalent).
/// Page 0 is the file header, so it is never a valid node page.
pub const NULL_PAGE: u64 = 0;

/// Default (and currently fixed) page size in bytes.
pub const PAGE_SIZE: usize = 4096;

/// Byte size of [`NodeHeader`].
pub const NODE_HEADER_SIZE: usize = 16;

// Node kind tags stored in `NodeHeader::node_kind`.
pub const NODE_KIND_INTERNAL: u8 = 0;
pub const NODE_KIND_LEAF: u8 = 1;
pub const NODE_KIND_FREE: u8 = 0xFF;

// ---------------------------------------------------------------------------
// On-disk structs  (repr(C), Pod — stable layout, safe byte reinterpretation)
// ---------------------------------------------------------------------------

/// Occupies the first [`PAGE_SIZE`] bytes of the file (only the first 64
/// bytes contain meaningful data; the rest are zeroed padding).
///
/// All multi-byte fields are stored in native byte order. This library does
/// not support cross-endian file sharing; files are tied to the host endianness.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FileHeader {
    /// ASCII magic: `b"MMBPTREE"`.
    pub magic: [u8; 8],
    /// Format version, currently [`VERSION`].
    pub version: u16,
    pub _pad0: u16,
    /// Page size in bytes; must equal [`PAGE_SIZE`] when opening an existing file.
    pub page_size: u32,
    /// `size_of::<K>()` at creation time.  Mismatch is a hard error on open.
    pub key_size: u32,
    /// `size_of::<V>()` at creation time.  Mismatch is a hard error on open.
    pub value_size: u32,
    /// Page index of the root node, or [`NULL_PAGE`] for an empty tree.
    pub root_page: u64,
    /// Head of the free-page singly-linked list, or [`NULL_PAGE`] if empty.
    pub free_list_head: u64,
    /// Total number of pages in the file (including page 0).
    pub num_pages: u64,
    /// Total number of key-value pairs stored in the tree.
    pub num_entries: u64,
    /// Reserved for future use (checksum, flags, etc.).
    pub _pad1: u64,
}

// Compile-time size assertion.
const _FILE_HEADER_SIZE: () = assert!(std::mem::size_of::<FileHeader>() == 64);

/// Sits at byte offset 0 of every non-header page.
///
/// For freed pages, this is reinterpreted as a [`FreePageHeader`] overlay
/// (same size and alignment).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct NodeHeader {
    /// [`NODE_KIND_INTERNAL`], [`NODE_KIND_LEAF`], or [`NODE_KIND_FREE`].
    pub node_kind: u8,
    pub _pad: u8,
    /// Number of live keys currently stored in this node.
    pub num_keys: u16,
    pub _pad2: u32,
    /// **Leaf**: page index of the next leaf sibling ([`NULL_PAGE`] = last leaf).
    /// **Internal**: always [`NULL_PAGE`].
    /// **Freed**: repurposed as `next_free` pointer by [`FreePageHeader`].
    pub next_leaf: u64,
}

const _NODE_HEADER_SIZE: () = assert!(std::mem::size_of::<NodeHeader>() == NODE_HEADER_SIZE);
const _NODE_HEADER_ALIGN: () = assert!(std::mem::align_of::<NodeHeader>() == 8);

/// Overlay written into the first [`NODE_HEADER_SIZE`] bytes of a freed page
/// so that free pages can be chained into a singly-linked list.
///
/// Has the same size and alignment as [`NodeHeader`]; the two are
/// layout-compatible.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FreePageHeader {
    /// Always [`NODE_KIND_FREE`] (0xFF) — distinguishes from live node kinds.
    pub kind: u8,
    pub _pad: u8,
    pub _pad2: u16,
    pub _pad3: u32,
    /// Next page in the free list, or [`NULL_PAGE`] if this is the tail.
    pub next_free: u64,
}

const _FREE_HEADER_SIZE: () = assert!(std::mem::size_of::<FreePageHeader>() == NODE_HEADER_SIZE);

// ---------------------------------------------------------------------------
// NodeLayout — pre-computed byte offsets for a given (K, V) pair
// ---------------------------------------------------------------------------

/// Pre-computed byte offsets and capacities for node pages.
///
/// Created once at open time via [`NodeLayout::new`] and then shared
/// (read-only) across all node accesses.  `NodeLayout` is not generic: it
/// stores the raw sizes and offsets derived from `size_of::<K>()` etc., so
/// the rest of the code can use it without carrying type parameters everywhere.
#[derive(Clone, Copy, Debug)]
pub struct NodeLayout {
    pub page_size: usize,

    /// Byte offset within a page where the key array begins.
    /// Equal to `align_up(NODE_HEADER_SIZE, align_of::<K>())`.
    pub key_start: usize,

    /// Maximum number of keys in a leaf node.
    pub leaf_capacity: usize,
    /// Byte offset within a leaf page where the value array begins.
    pub leaf_values_start: usize,

    /// Maximum number of keys in an internal node.
    /// An internal node with `n` keys has `n + 1` child pointers.
    pub internal_capacity: usize,
    /// Byte offset within an internal page where the child-pointer array begins.
    pub internal_children_start: usize,
}

impl NodeLayout {
    /// Computes the layout for a tree whose keys have `key_size` bytes and
    /// `key_align` alignment, and whose values have `value_size` bytes and
    /// `value_align` alignment, using pages of `page_size` bytes.
    ///
    /// # Panics
    ///
    /// Panics if the page is too small to hold even one key-value pair, or
    /// one key-child pair.
    pub fn new(
        page_size: usize,
        key_size: usize,
        key_align: usize,
        value_size: usize,
        value_align: usize,
    ) -> Self {
        let key_start = align_up(NODE_HEADER_SIZE, key_align);

        let leaf_capacity =
            compute_leaf_capacity(page_size, key_start, key_size, value_size, value_align);
        let leaf_values_start =
            align_up(key_start + leaf_capacity * key_size, value_align);

        let internal_capacity =
            compute_internal_capacity(page_size, key_start, key_size);
        let internal_children_start =
            align_up(key_start + internal_capacity * key_size, 8);

        NodeLayout {
            page_size,
            key_start,
            leaf_capacity,
            leaf_values_start,
            internal_capacity,
            internal_children_start,
        }
    }
}

/// Rounds `offset` up to the nearest multiple of `align`.
/// `align` must be a power of two (guaranteed for all Rust alignments).
#[inline]
pub const fn align_up(offset: usize, align: usize) -> usize {
    (offset + align - 1) & !(align - 1)
}

/// Largest `n` such that a leaf page with `n` key-value pairs fits in
/// `page_size` bytes given the computed alignment padding.
///
/// Uses a linear probe because the alignment padding between the key and
/// value arrays depends on `n * key_size`, making a closed-form expression
/// impractical.
fn compute_leaf_capacity(
    page_size: usize,
    key_start: usize,
    key_size: usize,
    value_size: usize,
    value_align: usize,
) -> usize {
    let mut n = 0usize;
    loop {
        let next = n + 1;
        let val_start = align_up(key_start + next * key_size, value_align);
        let end = val_start + next * value_size;
        if end > page_size {
            break;
        }
        n = next;
    }
    assert!(n > 0, "page_size too small for even one key-value pair");
    n
}

/// Largest `n` such that an internal page with `n` keys and `n + 1` child
/// pointers (`u64`) fits in `page_size` bytes.
fn compute_internal_capacity(
    page_size: usize,
    key_start: usize,
    key_size: usize,
) -> usize {
    let mut n = 0usize;
    loop {
        let next = n + 1;
        let children_start = align_up(key_start + next * key_size, 8); // u64 align
        let end = children_start + (next + 1) * 8; // n+1 children → n+2... no: n keys → n+1 children
        if end > page_size {
            break;
        }
        n = next;
    }
    assert!(n > 0, "page_size too small for even one key-child pair");
    n
}

// ---------------------------------------------------------------------------
// MmapStore — non-generic file and mapping manager
// ---------------------------------------------------------------------------

/// Manages the memory-mapped file backing the B+tree.
///
/// `MmapStore` is not generic: it deals only in raw byte pages.  The typed
/// node views ([`crate::node`]) are constructed by callers that know `K` and
/// `V`.
pub struct MmapStore {
    file: File,
    mmap: MmapMut,
    pub page_size: usize,
    /// Cached copy of `header().num_pages`.  Kept in sync by every operation
    /// that changes the page count so that page-index arithmetic never needs
    /// a `from_bytes` round-trip through the header.
    pub total_pages: u64,
}

impl std::fmt::Debug for MmapStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapStore")
            .field("page_size", &self.page_size)
            .field("total_pages", &self.total_pages)
            .finish()
    }
}

impl MmapStore {
    /// Opens the file at `path`, creating it if it does not exist.
    ///
    /// On creation, writes an initial [`FileHeader`] but does **not**
    /// allocate a root page; the tree starts empty (`root_page = NULL_PAGE`).
    ///
    /// On open, validates the magic, version, key size, and value size.
    ///
    /// # Errors
    ///
    /// Returns [`BTreeError::Io`] on any file-system failure, or
    /// [`BTreeError::Corruption`] if the existing file is malformed.
    pub fn open(path: &Path, key_size: usize, value_size: usize) -> Result<Self> {
        let file_exists = path.exists() && {
            let meta = std::fs::metadata(path).map_err(BTreeError::from)?;
            meta.len() > 0
        };

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)
            .map_err(BTreeError::from)?;

        if file_exists {
            Self::open_existing(file, key_size, value_size)
        } else {
            Self::create_new(file, key_size, value_size)
        }
    }

    fn open_existing(file: File, key_size: usize, value_size: usize) -> Result<Self> {
        // SAFETY: We own the file and hold it open for the lifetime of the
        // mapping.  No other process is assumed to write to it concurrently
        // (enforced by the caller via RwLock).
        let mmap = unsafe { MmapMut::map_mut(&file) }.map_err(BTreeError::from)?;

        // Validate the header.
        if mmap.len() < std::mem::size_of::<FileHeader>() {
            return Err(BTreeError::Corruption("file too small for a header".into()));
        }
        let hdr = bytemuck::from_bytes::<FileHeader>(&mmap[..64]);

        if hdr.magic != MAGIC {
            return Err(BTreeError::Corruption("bad magic bytes".into()));
        }
        if hdr.version != VERSION {
            return Err(BTreeError::Corruption(format!(
                "unsupported version {} (expected {})",
                hdr.version, VERSION
            )));
        }
        if hdr.page_size as usize != PAGE_SIZE {
            return Err(BTreeError::Corruption(format!(
                "page_size mismatch: file has {}, expected {}",
                hdr.page_size, PAGE_SIZE
            )));
        }
        if hdr.key_size as usize != key_size {
            return Err(BTreeError::Corruption(format!(
                "key_size mismatch: file has {}, K is {} bytes",
                hdr.key_size, key_size
            )));
        }
        if hdr.value_size as usize != value_size {
            return Err(BTreeError::Corruption(format!(
                "value_size mismatch: file has {}, V is {} bytes",
                hdr.value_size, value_size
            )));
        }

        let total_pages = hdr.num_pages;
        Ok(Self { file, mmap, page_size: PAGE_SIZE, total_pages })
    }

    fn create_new(file: File, key_size: usize, value_size: usize) -> Result<Self> {
        // Allocate exactly one page: the header page.
        file.set_len(PAGE_SIZE as u64).map_err(BTreeError::from)?;

        // SAFETY: file was just created/truncated; we own it exclusively.
        let mut mmap = unsafe { MmapMut::map_mut(&file) }.map_err(BTreeError::from)?;

        // Write the initial header.
        let hdr = bytemuck::from_bytes_mut::<FileHeader>(&mut mmap[..64]);
        *hdr = FileHeader {
            magic: MAGIC,
            version: VERSION,
            _pad0: 0,
            page_size: PAGE_SIZE as u32,
            key_size: key_size as u32,
            value_size: value_size as u32,
            root_page: NULL_PAGE,
            free_list_head: NULL_PAGE,
            num_pages: 1,
            num_entries: 0,
            _pad1: 0,
        };

        mmap.flush().map_err(BTreeError::from)?;

        Ok(Self { file, mmap, page_size: PAGE_SIZE, total_pages: 1 })
    }

    // -----------------------------------------------------------------------
    // Raw page access
    // -----------------------------------------------------------------------

    /// Returns a read-only byte slice for page `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= total_pages` (internal bug — callers must validate).
    #[inline]
    pub fn page(&self, index: u64) -> &[u8] {
        let start = index as usize * self.page_size;
        &self.mmap[start..start + self.page_size]
    }

    /// Returns a mutable byte slice for page `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= total_pages`.
    #[inline]
    pub fn page_mut(&mut self, index: u64) -> &mut [u8] {
        let start = index as usize * self.page_size;
        &mut self.mmap[start..start + self.page_size]
    }

    // -----------------------------------------------------------------------
    // Header access
    // -----------------------------------------------------------------------

    /// Returns a reference to the [`FileHeader`].
    ///
    /// # Safety (soundness argument)
    ///
    /// `self.mmap` starts at a page-aligned (≥ 4096-byte) address.
    /// `FileHeader` has `align = 8`; `8 | 4096`.  `bytemuck::from_bytes`
    /// trusts the caller to provide sufficient alignment — our invariant
    /// guarantees it.
    #[inline]
    pub fn header(&self) -> &FileHeader {
        bytemuck::from_bytes(&self.mmap[..std::mem::size_of::<FileHeader>()])
    }

    /// Returns a mutable reference to the [`FileHeader`].
    #[inline]
    pub fn header_mut(&mut self) -> &mut FileHeader {
        bytemuck::from_bytes_mut(&mut self.mmap[..std::mem::size_of::<FileHeader>()])
    }

    // -----------------------------------------------------------------------
    // Page allocation / deallocation
    // -----------------------------------------------------------------------

    /// Allocates a page, returning its index.
    ///
    /// Pops from the free list if non-empty; otherwise grows the file by one
    /// page.  The returned page contains whatever bytes were last written to
    /// it — callers **must** call the appropriate `init()` method on the node
    /// view before treating it as a valid node.
    ///
    /// # Errors
    ///
    /// Returns [`BTreeError::Io`] if file growth fails.
    pub fn alloc_page(&mut self) -> Result<u64> {
        // Read the current free-list head before any mutation.
        let free_head = self.header().free_list_head;

        if free_head != NULL_PAGE {
            // Pop from the free list.
            // Borrow of `page(free_head)` ends before `header_mut()` starts.
            let next_free = {
                let page = self.page(free_head);
                bytemuck::from_bytes::<FreePageHeader>(&page[..NODE_HEADER_SIZE]).next_free
            };
            self.header_mut().free_list_head = next_free;
            Ok(free_head)
        } else {
            // No free pages — grow the file by one page.
            // TODO(perf): grow in batches (e.g. 64 pages) to amortise the
            // cost of file extension and mmap recreation.
            self.grow()?;
            Ok(self.total_pages - 1)
        }
    }

    /// Returns page `index` to the free list.
    ///
    /// Overwrites the first [`NODE_HEADER_SIZE`] bytes of the page with a
    /// [`FreePageHeader`] so the page can be reused.  Does **not** zero the
    /// rest of the page — callers must call `init()` after re-allocating.
    pub fn free_page(&mut self, index: u64) {
        debug_assert_ne!(index, NULL_PAGE, "cannot free the header page");

        // Step 1: snapshot the current free-list head (immutable borrow drops
        //         at the semicolon).
        let old_head = self.header().free_list_head;

        // Step 2: write the FreePageHeader into the freed page.
        //         The mutable borrow of the *page region* drops at the `}`.
        {
            let page = self.page_mut(index);
            let fph = bytemuck::from_bytes_mut::<FreePageHeader>(&mut page[..NODE_HEADER_SIZE]);
            *fph = FreePageHeader {
                kind: NODE_KIND_FREE,
                _pad: 0,
                _pad2: 0,
                _pad3: 0,
                next_free: old_head,
            };
        }

        // Step 3: update the header.  The previous &mut borrow has already
        //         been dropped, so this &mut borrow is legal.
        self.header_mut().free_list_head = index;
    }

    // -----------------------------------------------------------------------
    // File growth
    // -----------------------------------------------------------------------

    /// Extends the file by one page and recreates the memory mapping.
    ///
    /// `MmapMut` cannot be resized in place, so the existing mapping is
    /// replaced with a 1-byte anonymous mapping as a placeholder while the
    /// file is extended, then a new full mapping is installed.
    ///
    /// # Safety (soundness argument)
    ///
    /// `grow` takes `&mut self`, which means no other code holds a live
    /// reference into `self.mmap`.  The swap through `map_anon` keeps
    /// `self.mmap` in a valid (non-dangling) state at all times.
    fn grow(&mut self) -> Result<()> {
        let new_total = self.total_pages + 1;
        let new_size = new_total as u64 * self.page_size as u64;

        // Replace the existing mapping with a 1-byte anonymous placeholder so
        // that `self.mmap` is never in an invalid/dangling state during the
        // file extension.
        let old_mmap = std::mem::replace(
            &mut self.mmap,
            // map_anon allocates anonymous memory — no file, no aliasing.
            MmapMut::map_anon(1).map_err(BTreeError::from)?,
        );
        drop(old_mmap); // explicit: old mapping is fully released before set_len

        self.file.set_len(new_size).map_err(BTreeError::from)?;

        // SAFETY: The file has been extended; we own it exclusively.
        self.mmap = unsafe { MmapMut::map_mut(&self.file) }.map_err(BTreeError::from)?;

        self.total_pages = new_total;
        self.header_mut().num_pages = new_total;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Persistence
    // -----------------------------------------------------------------------

    /// Flushes all dirty pages to disk via `msync`.
    ///
    /// # Errors
    ///
    /// Returns [`BTreeError::Io`] if the flush fails.
    pub fn flush(&self) -> Result<()> {
        self.mmap.flush().map_err(BTreeError::from)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_header_size() {
        assert_eq!(std::mem::size_of::<FileHeader>(), 64);
    }

    #[test]
    fn node_header_size_and_align() {
        assert_eq!(std::mem::size_of::<NodeHeader>(), 16);
        assert_eq!(std::mem::align_of::<NodeHeader>(), 8);
    }

    #[test]
    fn free_header_size() {
        assert_eq!(std::mem::size_of::<FreePageHeader>(), 16);
    }

    #[test]
    fn leaf_capacity_i32_i32() {
        // K = i32 (4 bytes, align 4), V = i32 (4 bytes, align 4)
        let layout = NodeLayout::new(4096, 4, 4, 4, 4);
        // key_start = align_up(16, 4) = 16
        // layout: 16 + n*4 (keys) + n*4 (values) ≤ 4096
        // n ≤ (4096 - 16) / 8 = 510
        assert_eq!(layout.key_start, 16);
        assert_eq!(layout.leaf_capacity, 510);
        assert_eq!(layout.leaf_values_start, 16 + 510 * 4); // 2056, already 8-aligned
    }

    #[test]
    fn leaf_capacity_u64_u64() {
        // K = u64 (8 bytes, align 8), V = u64 (8 bytes, align 8)
        let layout = NodeLayout::new(4096, 8, 8, 8, 8);
        // key_start = align_up(16, 8) = 16
        // 16 + n*8 + n*8 ≤ 4096  →  n ≤ (4096 - 16) / 16 = 255
        assert_eq!(layout.leaf_capacity, 255);
    }

    #[test]
    fn internal_capacity_i32() {
        // K = i32: each key 4 bytes, each child 8 bytes
        let layout = NodeLayout::new(4096, 4, 4, 4, 4);
        // key_start=16, n keys: children_start=align_up(16+4n, 8), end=children_start+(n+1)*8
        // rough: 16 + 4n + 8 + (n+1)*8 ≤ 4096  →  12n ≤ 4064 → n ≤ 338
        assert!(layout.internal_capacity >= 300);
    }

    #[test]
    fn align_up_power_of_two() {
        assert_eq!(align_up(0, 8), 0);
        assert_eq!(align_up(1, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(9, 8), 16);
        assert_eq!(align_up(16, 4), 16);
        assert_eq!(align_up(17, 4), 20);
    }

    #[test]
    fn create_and_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");

        {
            let store = MmapStore::open(&path, 4, 4).unwrap();
            assert_eq!(store.total_pages, 1);
            assert_eq!(store.header().root_page, NULL_PAGE);
        }

        // Re-open the existing file.
        let store = MmapStore::open(&path, 4, 4).unwrap();
        assert_eq!(store.header().magic, MAGIC);
        assert_eq!(store.header().version, VERSION);
    }

    #[test]
    fn open_rejects_wrong_key_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        MmapStore::open(&path, 4, 4).unwrap();
        let err = MmapStore::open(&path, 8, 4).unwrap_err();
        assert!(matches!(err, BTreeError::Corruption(_)));
    }

    #[test]
    fn alloc_and_free_page() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let mut store = MmapStore::open(&path, 4, 4).unwrap();

        let p1 = store.alloc_page().unwrap();
        assert_eq!(p1, 1); // first real page
        assert_eq!(store.total_pages, 2);

        let p2 = store.alloc_page().unwrap();
        assert_eq!(p2, 2);

        // Free p1, then re-allocate — should get p1 back.
        store.free_page(p1);
        let p3 = store.alloc_page().unwrap();
        assert_eq!(p3, p1);
    }
}
