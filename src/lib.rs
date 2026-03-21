//! # mappedbptree
//!
//! A persistent, memory-mapped B+tree for Rust.
//!
//! ## Features
//!
//! - **File-backed persistence** via `mmap` — data survives process restarts.
//! - **Crash safety** — every `insert` and `remove` is protected by a
//!   write-ahead log (WAL).  If the process dies mid-write, the pending
//!   operation is automatically replayed the next time the tree is opened,
//!   leaving the data structure in a fully consistent state.
//! - **Corruption detection** — every node page carries a CRC32 checksum.
//!   A partial write (e.g. from a power loss) is detected immediately and
//!   reported as [`BTreeError::Corruption`] rather than silently returning
//!   wrong data.
//! - **Thread-safe** — multiple threads may read concurrently; writes are
//!   serialised via an internal `RwLock`.
//! - **Zero-copy reads** — [`MmapBTree::get`] returns a [`MmapBTreeValueRef`]
//!   that borrows directly from the mmap without copying.
//! - **`BTreeMap`-like API** — `insert`, `get`, `remove`, `iter`, `range`,
//!   `clear`, `len`, and `contains_key`.
//!
//! ## Type constraints
//!
//! Keys (`K: Ord + Pod`) and values (`V: Pod`) must implement
//! [`bytemuck::Pod`], which guarantees they are plain-data types safe to store
//! as raw bytes.  Integers, fixed-size arrays, and `#[repr(C)]` structs work;
//! heap-owning types like `String` or `Vec` do not.
//!
//! ## Quick Start
//!
//! ```no_run
//! use mappedbptree::MmapBTreeBuilder;
//!
//! let tree = MmapBTreeBuilder::<i32, u64>::new()
//!     .path("my_tree.db")
//!     .build()?;
//!
//! tree.insert(1_i32, 42_u64)?;
//! assert_eq!(tree.get_value(&1_i32)?, Some(42_u64));
//! tree.remove(&1_i32)?;
//! # Ok::<_, Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Crash safety in detail
//!
//! Before each `insert` or `remove`, a WAL file (`{db}.wal`) is written and
//! fsynced to disk.  The mmap mutations follow, then the mmap is flushed and
//! the WAL deleted.  On open, any leftover WAL is replayed — both operations
//! are idempotent, so replay is always safe.  If the WAL itself was
//! truncated (the fsync didn't finish), it is silently ignored and the tree
//! remains in its clean pre-crash state.

pub mod tree;
pub(crate) mod storage;
pub(crate) mod node;
pub(crate) mod wal;

pub use tree::{BTreeError, MmapBTree, MmapBTreeBuilder, MmapBTreeValueRef, Result};
