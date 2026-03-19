# mappedbptree

A persistent, memory-mapped B+tree for Rust with a `BTreeMap`-like API.

Built by Claude.

## Features

- File-backed storage via `mmap` — data survives process restarts
- Thread-safe: multiple concurrent readers, exclusive writers (`RwLock`)
- Zero-copy access to keys and values via `bytemuck::Pod`
- Full B+tree operations: `insert`, `get`, `remove`, `iter`, `range`, `clear`
- Auto-tuned node capacity based on system page size (4096 bytes)

## Constraints

Keys (`K: Ord + Pod`) and values (`V: Pod`) must be plain-data types — fixed-size,
no heap allocations. Integers, arrays, and `#[repr(C)]` structs work; `String` and
`Vec` do not.

## Quick start

```toml
[dependencies]
mappedbptree = "0.1"
```

```rust
use mappedbptree::MmapBTreeBuilder;

let tree = MmapBTreeBuilder::<i32, u64>::new()
    .path("data.db")
    .build()?;

tree.insert(1, 100)?;
assert_eq!(tree.get(&1)?, Some(100));

tree.remove(&1)?;

for (k, v) in tree.range(0..50)? {
    println!("{k}: {v}");
}
```

## License

MIT
