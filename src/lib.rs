//! # mappedbptree
//!
//! A Rust library providing a memory-mapped B+tree data structure with:
//! - File-backed persistence via memory mapping
//! - Thread-safe concurrent read access (RwLock-based)
//! - BTreeMap-like API
//! - Proper I/O error handling
//!
//! ## Quick Start
//!
//! Keys and values must implement [`bytemuck::Pod`] — they are stored
//! directly as bytes in the memory-mapped file.  This means only plain-data
//! types (integers, fixed-size arrays, `#[repr(C)]` structs without pointers)
//! are supported; heap-owning types like `String` or `Vec` are not.
//!
//! ```no_run
//! use mappedbptree::MmapBTreeBuilder;
//!
//! let tree = MmapBTreeBuilder::<i32, u64>::new()
//!     .path("my_tree.db")
//!     .build()?;
//!
//! tree.insert(1_i32, 42_u64)?;
//! assert_eq!(tree.get(&1_i32)?, Some(42_u64));
//! # Ok::<_, Box<dyn std::error::Error>>(())
//! ```

pub mod tree;
pub(crate) mod storage;
pub(crate) mod node;

pub use tree::{BTreeError, MmapBTree, MmapBTreeBuilder, Result};
