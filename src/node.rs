//! Typed zero-cost views into raw node pages.
//!
//! Each view wraps a `&[u8]` (or `&mut [u8]`) page slice and a reference to
//! the pre-computed [`NodeLayout`].  No heap allocation occurs.
//!
//! # Alignment and safety
//!
//! All slice-from-raw-parts calls within this module are sound because:
//!
//! 1. `memmap2` guarantees that the base address of the mapping is aligned to
//!    at least the system page size (4096 bytes).
//! 2. Every page starts at an offset that is a multiple of `PAGE_SIZE` from
//!    the base, so each page pointer is also 4096-aligned.
//! 3. `NodeLayout::key_start = align_up(NODE_HEADER_SIZE, align_of::<K>())`,
//!    so `page_ptr + key_start` satisfies `align_of::<K>()` for any K whose
//!    alignment divides 4096 (i.e. any K with `align_of::<K>() ≤ 4096`).
//!    `bytemuck::Pod` types satisfy this because Rust's allocator guarantees
//!    alignment ≤ `isize::MAX` and all real-world Pod types have alignment
//!    ≤ a few hundred bytes.
//! 4. `NodeLayout::leaf_values_start = align_up(..., align_of::<V>())` and
//!    `NodeLayout::internal_children_start = align_up(..., 8)` carry the same
//!    guarantee for V and `u64` respectively.
//! 5. `K: bytemuck::Pod` guarantees that any bit pattern is a valid K, so
//!    reading uninitialised-looking bytes (e.g. past `num_keys` in the capacity
//!    array) cannot cause undefined behaviour.
//!
//! `bytemuck::cast_slice::<u8, K>` cannot be used here because it checks
//! `align_of::<K>() ≤ align_of::<u8>()` (i.e. ≤ 1), which fails for every
//! type wider than a byte.  The raw-pointer approach with the documented
//! invariants above is the correct alternative.

use std::marker::PhantomData;
use std::slice;

use bytemuck::Pod;

use crate::storage::{
    NODE_HEADER_SIZE, NODE_KIND_INTERNAL, NODE_KIND_LEAF, NodeHeader, NodeLayout,
};

// ---------------------------------------------------------------------------
// Leaf node — read-only view
// ---------------------------------------------------------------------------

/// Read-only typed view into a leaf node page.
///
/// Holds a shared borrow of the page slice and of the [`NodeLayout`].
/// Dropping this view releases the borrow — which, in practice, releases the
/// `RwLockReadGuard` that was keeping the mapping stable.
pub struct LeafView<'a, K, V> {
    data: &'a [u8],
    layout: &'a NodeLayout,
    _marker: PhantomData<(K, V)>,
}

impl<'a, K: Pod, V: Pod> LeafView<'a, K, V> {
    /// Constructs a view from a page-sized byte slice.
    ///
    /// # Panics (debug only)
    ///
    /// Panics in debug builds if `data.len() != layout.page_size` or if the
    /// page's `node_kind` is not [`NODE_KIND_LEAF`].
    #[inline]
    pub fn new(data: &'a [u8], layout: &'a NodeLayout) -> Self {
        debug_assert_eq!(data.len(), layout.page_size, "page slice has wrong length");
        debug_assert_eq!(
            data[0], NODE_KIND_LEAF,
            "LeafView constructed on a non-leaf page (kind={})",
            data[0]
        );
        Self {
            data,
            layout,
            _marker: PhantomData,
        }
    }

    /// Returns a reference to the raw [`NodeHeader`].
    #[inline]
    pub fn header(&self) -> &NodeHeader {
        // SAFETY: page starts at ≥4096-byte boundary; NodeHeader has align=8;
        // 8 divides 4096.  bytemuck trusts caller for alignment.
        bytemuck::from_bytes(&self.data[..NODE_HEADER_SIZE])
    }

    /// Number of live key-value pairs in this leaf.
    #[inline]
    pub fn num_keys(&self) -> usize {
        self.header().num_keys as usize
    }

    /// Slice of live keys (length = `num_keys`).
    #[inline]
    pub fn keys(&self) -> &[K] {
        // SAFETY: See module-level alignment argument.
        // `num_keys ≤ leaf_capacity` is an invariant maintained by all
        // mutating operations (checked in debug builds by write-path asserts).
        unsafe {
            slice::from_raw_parts(
                self.data.as_ptr().add(self.layout.key_start) as *const K,
                self.num_keys(),
            )
        }
    }

    /// Slice of live values (length = `num_keys`).
    #[inline]
    pub fn values(&self) -> &[V] {
        // SAFETY: Same invariants as `keys()`. leaf_values_start is
        // align_up(..., align_of::<V>()) so the pointer is V-aligned.
        unsafe {
            slice::from_raw_parts(
                self.data.as_ptr().add(self.layout.leaf_values_start) as *const V,
                self.num_keys(),
            )
        }
    }

    /// Page index of the next leaf sibling, or [`crate::storage::NULL_PAGE`].
    #[inline]
    pub fn next_leaf(&self) -> u64 {
        self.header().next_leaf
    }
}

// ---------------------------------------------------------------------------
// Leaf node — mutable view
// ---------------------------------------------------------------------------

/// Mutable typed view into a leaf node page.
///
/// `keys_mut()` and `values_mut()` return slices of the **full capacity**
/// (not just the live `num_keys` elements) so that the tree algorithm can
/// shift elements during insertion and removal without a second accessor.
/// After modifying the arrays, call [`Self::set_num_keys`] to commit the new
/// live count.
pub struct LeafViewMut<'a, K, V> {
    data: &'a mut [u8],
    layout: &'a NodeLayout,
    _marker: PhantomData<(K, V)>,
}

impl<'a, K: Pod, V: Pod> LeafViewMut<'a, K, V> {
    /// Constructs a mutable view from a page-sized byte slice.
    #[inline]
    pub fn new(data: &'a mut [u8], layout: &'a NodeLayout) -> Self {
        debug_assert_eq!(data.len(), layout.page_size, "page slice has wrong length");
        Self {
            data,
            layout,
            _marker: PhantomData,
        }
    }

    /// Initialises this page as a fresh, empty leaf node.
    ///
    /// Zeroes the whole page then stamps `node_kind = NODE_KIND_LEAF`.
    /// Must be called on every freshly allocated page before it is used.
    pub fn init(&mut self) {
        self.data.fill(0);
        let hdr = bytemuck::from_bytes_mut::<NodeHeader>(&mut self.data[..NODE_HEADER_SIZE]);
        hdr.node_kind = NODE_KIND_LEAF;
        // num_keys, next_leaf, and _pad fields are already 0 from fill(0).
    }

    /// Returns a reference to the raw [`NodeHeader`].
    #[inline]
    pub fn header(&self) -> &NodeHeader {
        bytemuck::from_bytes(&self.data[..NODE_HEADER_SIZE])
    }

    /// Returns a mutable reference to the raw [`NodeHeader`].
    #[inline]
    pub fn header_mut(&mut self) -> &mut NodeHeader {
        bytemuck::from_bytes_mut(&mut self.data[..NODE_HEADER_SIZE])
    }

    /// Number of live key-value pairs in this leaf.
    #[inline]
    pub fn num_keys(&self) -> usize {
        self.header().num_keys as usize
    }

    /// Sets the live key count.  Must be called after any insertion or removal.
    #[inline]
    pub fn set_num_keys(&mut self, n: usize) {
        debug_assert!(n <= self.layout.leaf_capacity);
        self.header_mut().num_keys = n as u16;
    }

    /// Sets the next-sibling leaf pointer.
    #[inline]
    pub fn set_next_leaf(&mut self, page: u64) {
        self.header_mut().next_leaf = page;
    }

    /// Mutable slice over the **full key capacity** (length = `leaf_capacity`).
    ///
    /// Only indices `0..num_keys` contain live data; the rest may hold stale
    /// bytes.  The tree algorithm must update `num_keys` via [`Self::set_num_keys`]
    /// after writing.
    #[inline]
    pub fn keys_mut(&mut self) -> &mut [K] {
        // SAFETY: Same alignment invariants as LeafView::keys(), plus
        // exclusive access is guaranteed because we hold &mut [u8].
        unsafe {
            slice::from_raw_parts_mut(
                self.data.as_mut_ptr().add(self.layout.key_start) as *mut K,
                self.layout.leaf_capacity,
            )
        }
    }

    /// Mutable slice over the **full value capacity** (length = `leaf_capacity`).
    #[inline]
    pub fn values_mut(&mut self) -> &mut [V] {
        // SAFETY: leaf_values_start is V-aligned; exclusive access via &mut.
        unsafe {
            slice::from_raw_parts_mut(
                self.data.as_mut_ptr().add(self.layout.leaf_values_start) as *mut V,
                self.layout.leaf_capacity,
            )
        }
    }

    /// Read-only slice of live keys (length = `num_keys`).
    #[inline]
    #[allow(dead_code)]
    pub fn keys(&self) -> &[K] {
        unsafe {
            slice::from_raw_parts(
                self.data.as_ptr().add(self.layout.key_start) as *const K,
                self.num_keys(),
            )
        }
    }

    /// Read-only slice of live values (length = `num_keys`).
    #[inline]
    #[allow(dead_code)]
    pub fn values(&self) -> &[V] {
        unsafe {
            slice::from_raw_parts(
                self.data.as_ptr().add(self.layout.leaf_values_start) as *const V,
                self.num_keys(),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Internal node — read-only view
// ---------------------------------------------------------------------------

/// Read-only typed view into an internal node page.
///
/// An internal node with `n` keys has exactly `n + 1` child page pointers.
pub struct InternalView<'a, K> {
    data: &'a [u8],
    layout: &'a NodeLayout,
    _marker: PhantomData<K>,
}

impl<'a, K: Pod> InternalView<'a, K> {
    /// Constructs a view from a page-sized byte slice.
    #[inline]
    pub fn new(data: &'a [u8], layout: &'a NodeLayout) -> Self {
        debug_assert_eq!(data.len(), layout.page_size);
        debug_assert_eq!(
            data[0], NODE_KIND_INTERNAL,
            "InternalView constructed on a non-internal page (kind={})",
            data[0]
        );
        Self {
            data,
            layout,
            _marker: PhantomData,
        }
    }

    /// Returns a reference to the raw [`NodeHeader`].
    #[inline]
    pub fn header(&self) -> &NodeHeader {
        bytemuck::from_bytes(&self.data[..NODE_HEADER_SIZE])
    }

    /// Number of live keys in this node.
    #[inline]
    pub fn num_keys(&self) -> usize {
        self.header().num_keys as usize
    }

    /// Slice of live keys (length = `num_keys`).
    #[inline]
    pub fn keys(&self) -> &[K] {
        // SAFETY: See module-level alignment argument.
        unsafe {
            slice::from_raw_parts(
                self.data.as_ptr().add(self.layout.key_start) as *const K,
                self.num_keys(),
            )
        }
    }

    /// Slice of live child page indices (length = `num_keys + 1`).
    #[inline]
    pub fn children(&self) -> &[u64] {
        // SAFETY: internal_children_start = align_up(..., 8), so the pointer
        // is 8-byte aligned (= align_of::<u64>()).  num_keys+1 u64s fit
        // within the page by construction of internal_capacity.
        unsafe {
            slice::from_raw_parts(
                self.data.as_ptr().add(self.layout.internal_children_start) as *const u64,
                self.num_keys() + 1,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Internal node — mutable view
// ---------------------------------------------------------------------------

/// Mutable typed view into an internal node page.
///
/// As with [`LeafViewMut`], the mutable key and child slices expose the full
/// capacity so the tree algorithm can shift entries freely.
pub struct InternalViewMut<'a, K> {
    data: &'a mut [u8],
    layout: &'a NodeLayout,
    _marker: PhantomData<K>,
}

impl<'a, K: Pod> InternalViewMut<'a, K> {
    /// Constructs a mutable view from a page-sized byte slice.
    #[inline]
    pub fn new(data: &'a mut [u8], layout: &'a NodeLayout) -> Self {
        debug_assert_eq!(data.len(), layout.page_size);
        Self {
            data,
            layout,
            _marker: PhantomData,
        }
    }

    /// Initialises this page as a fresh, empty internal node.
    pub fn init(&mut self) {
        self.data.fill(0);
        let hdr = bytemuck::from_bytes_mut::<NodeHeader>(&mut self.data[..NODE_HEADER_SIZE]);
        hdr.node_kind = NODE_KIND_INTERNAL;
    }

    /// Returns a reference to the raw [`NodeHeader`].
    #[inline]
    pub fn header(&self) -> &NodeHeader {
        bytemuck::from_bytes(&self.data[..NODE_HEADER_SIZE])
    }

    /// Returns a mutable reference to the raw [`NodeHeader`].
    #[inline]
    pub fn header_mut(&mut self) -> &mut NodeHeader {
        bytemuck::from_bytes_mut(&mut self.data[..NODE_HEADER_SIZE])
    }

    /// Number of live keys.
    #[inline]
    pub fn num_keys(&self) -> usize {
        self.header().num_keys as usize
    }

    /// Sets the live key count.
    #[inline]
    pub fn set_num_keys(&mut self, n: usize) {
        debug_assert!(n <= self.layout.internal_capacity);
        self.header_mut().num_keys = n as u16;
    }

    /// Mutable slice over the **full key capacity** (length = `internal_capacity`).
    #[inline]
    pub fn keys_mut(&mut self) -> &mut [K] {
        // SAFETY: See module-level alignment argument.
        unsafe {
            slice::from_raw_parts_mut(
                self.data.as_mut_ptr().add(self.layout.key_start) as *mut K,
                self.layout.internal_capacity,
            )
        }
    }

    /// Mutable slice over the **full child array** (length = `internal_capacity + 1`).
    #[inline]
    pub fn children_mut(&mut self) -> &mut [u64] {
        // SAFETY: internal_children_start is 8-aligned; exclusive &mut access.
        unsafe {
            slice::from_raw_parts_mut(
                self.data
                    .as_mut_ptr()
                    .add(self.layout.internal_children_start) as *mut u64,
                self.layout.internal_capacity + 1,
            )
        }
    }

    /// Read-only slice of live keys (length = `num_keys`).
    #[inline]
    #[allow(dead_code)]
    pub fn keys(&self) -> &[K] {
        unsafe {
            slice::from_raw_parts(
                self.data.as_ptr().add(self.layout.key_start) as *const K,
                self.num_keys(),
            )
        }
    }

    /// Read-only slice of live child pointers (length = `num_keys + 1`).
    #[inline]
    #[allow(dead_code)]
    pub fn children(&self) -> &[u64] {
        unsafe {
            slice::from_raw_parts(
                self.data.as_ptr().add(self.layout.internal_children_start) as *const u64,
                self.num_keys() + 1,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{NULL_PAGE, NodeLayout};

    fn leaf_layout() -> NodeLayout {
        // K = u32 (4, 4), V = u64 (8, 8)
        NodeLayout::new(4096, 4, 4, 8, 8)
    }

    fn internal_layout() -> NodeLayout {
        // K = u32 (4, 4)
        NodeLayout::new(4096, 4, 4, 8, 8)
    }

    #[test]
    fn leaf_init_and_accessors() {
        let layout = leaf_layout();
        let mut page = vec![0u8; 4096];

        let mut view: LeafViewMut<u32, u64> = LeafViewMut::new(&mut page, &layout);
        view.init();
        assert_eq!(view.num_keys(), 0);
        assert_eq!(view.header().node_kind, crate::storage::NODE_KIND_LEAF);

        // Write two entries.
        view.keys_mut()[0] = 10u32;
        view.keys_mut()[1] = 20u32;
        view.values_mut()[0] = 100u64;
        view.values_mut()[1] = 200u64;
        view.set_num_keys(2);
        view.set_next_leaf(NULL_PAGE);

        // Read back through the immutable view.
        let read: LeafView<u32, u64> = LeafView::new(&page, &layout);
        assert_eq!(read.num_keys(), 2);
        assert_eq!(read.keys(), &[10u32, 20u32]);
        assert_eq!(read.values(), &[100u64, 200u64]);
        assert_eq!(read.next_leaf(), NULL_PAGE);
    }

    #[test]
    fn internal_init_and_accessors() {
        let layout = internal_layout();
        let mut page = vec![0u8; 4096];

        let mut view: InternalViewMut<u32> = InternalViewMut::new(&mut page, &layout);
        view.init();
        assert_eq!(view.num_keys(), 0);
        assert_eq!(view.header().node_kind, crate::storage::NODE_KIND_INTERNAL);

        // One key, two children.
        view.keys_mut()[0] = 42u32;
        view.children_mut()[0] = 5;
        view.children_mut()[1] = 7;
        view.set_num_keys(1);

        let read: InternalView<u32> = InternalView::new(&page, &layout);
        assert_eq!(read.num_keys(), 1);
        assert_eq!(read.keys(), &[42u32]);
        assert_eq!(read.children(), &[5u64, 7u64]);
    }
}
