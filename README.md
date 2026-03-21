# mappedbptree

A persistent, memory-mapped B+tree for Rust with a `BTreeMap`-like API.

Built by Claude.

## Features

- File-backed storage via `mmap` — data survives process restarts
- Crash-safe writes with a write-ahead log (WAL) is fsynced before every
  `insert` or `remove`; an interrupted write is automatically replayed on
  next open, leaving the tree consistent (since 0.2)
- Corruption detection — every node page carries a CRC32 checksum;
  a partial write is detected immediately and reported as an error (since 0.2)
- Thread-safe: multiple concurrent readers, exclusive writers (`RwLock`)
- Zero-copy reads via `get` — borrows directly from the mmap without copying
- Full B+tree operations: `insert`, `get`, `get_value`, `remove`, `iter`, `range`, `clear`
- Auto-tuned node capacity based on system page size (4096 bytes)

## Constraints

Keys (`K: Ord + Pod`) and values (`V: Pod`) must be plain-data types — fixed-size,
no heap allocations. Integers, arrays, and `#[repr(C)]` structs work; `String` and
`Vec` do not.

## Quick start

```toml
[dependencies]
mappedbptree = "0.2"
```

```rust
use mappedbptree::MmapBTreeBuilder;

let tree = MmapBTreeBuilder::<i32, u64>::new()
    .path("data.db")
    .build()?;

tree.insert(1, 100)?;
assert_eq!(tree.get_value(&1)?, Some(100));

tree.remove(&1)?;

for (k, v) in tree.range(0..50)? {
    println!("{k}: {v}");
}
```

## License

MIT
