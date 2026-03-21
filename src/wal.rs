//! Write-Ahead Log (WAL) for crash-safe insert and remove operations.
//!
//! # Protocol
//!
//! Before every public `insert` or `remove`:
//! 1. Write a WAL record to `{db_path}.wal` and call `fsync`.
//! 2. Perform the mmap mutations.
//! 3. `msync` the mmap.
//! 4. Delete the WAL file.
//!
//! On `open`, if a WAL file is present the pending operation is replayed
//! (both `insert` and `remove` are idempotent, so replay is always safe).
//!
//! # File format
//!
//! ```text
//! offset  0: [4]  magic b"BWAL"
//! offset  4: [2]  key_size   (u16 little-endian) — validated on read
//! offset  6: [2]  value_size (u16 little-endian) — validated on read
//! offset  8: [1]  op: INSERT = 0x01, REMOVE = 0x02
//! offset  9: [key_size]   key bytes
//! offset 9+K: [value_size] value bytes (all-zero for REMOVE)
//! ```
//!
//! Total: `9 + key_size + value_size` bytes. Fixed-size, no framing needed.
//! If the file is shorter than the expected length, it was truncated by an
//! interrupted write; `read_existing` treats it as absent (returns `Ok(None)`).

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::tree::{BTreeError, Result};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Magic bytes at the start of every WAL file.
pub const WAL_MAGIC: [u8; 4] = *b"BWAL";

/// Op tag: insert a key-value pair.
pub const WAL_OP_INSERT: u8 = 0x01;

/// Op tag: remove a key.
pub const WAL_OP_REMOVE: u8 = 0x02;

/// Fixed-size WAL header: magic (4) + key_size (2) + value_size (2) + op (1).
const WAL_HEADER_LEN: usize = 9;

// ---------------------------------------------------------------------------
// WAL record
// ---------------------------------------------------------------------------

/// A pending operation decoded from the WAL file.
pub struct WalRecord {
    /// `WAL_OP_INSERT` or `WAL_OP_REMOVE`.
    pub op: u8,
    /// Raw key bytes (`size_of::<K>()` bytes, native endian).
    pub key: Vec<u8>,
    /// Raw value bytes (`size_of::<V>()` bytes, native endian).
    /// All-zero when `op == WAL_OP_REMOVE`.
    pub value: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Path helper
// ---------------------------------------------------------------------------

/// Returns the WAL file path for a given database path: `{db_path}.wal`.
pub fn wal_path(db_path: &Path) -> PathBuf {
    let mut p = db_path.as_os_str().to_os_string();
    p.push(".wal");
    PathBuf::from(p)
}

// ---------------------------------------------------------------------------
// Core WAL functions
// ---------------------------------------------------------------------------

/// Reads and validates an existing WAL file.
///
/// Returns `Ok(None)` if:
/// - The file does not exist.
/// - The file is shorter than the expected length (interrupted write — safe to ignore).
/// - The magic bytes are wrong.
/// - The key or value sizes in the header don't match the current tree's sizes.
///
/// Returns `Ok(Some(record))` if a complete, valid WAL record is found.
pub fn read_existing(
    db_path: &Path,
    key_size: usize,
    value_size: usize,
) -> Result<Option<WalRecord>> {
    let path = wal_path(db_path);

    let mut file = match File::open(&path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(BTreeError::from(e)),
    };

    let expected_len = WAL_HEADER_LEN + key_size + value_size;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).map_err(BTreeError::from)?;

    // Truncated write — the fsync didn't complete; treat as absent.
    if buf.len() < expected_len {
        return Ok(None);
    }

    // Validate magic.
    if buf[..4] != WAL_MAGIC {
        return Ok(None);
    }

    // Validate stored key_size and value_size.
    let stored_ks = u16::from_le_bytes([buf[4], buf[5]]) as usize;
    let stored_vs = u16::from_le_bytes([buf[6], buf[7]]) as usize;
    if stored_ks != key_size || stored_vs != value_size {
        return Ok(None);
    }

    let op = buf[8];
    let key = buf[WAL_HEADER_LEN..WAL_HEADER_LEN + key_size].to_vec();
    let value = buf[WAL_HEADER_LEN + key_size..WAL_HEADER_LEN + key_size + value_size].to_vec();

    Ok(Some(WalRecord { op, key, value }))
}

/// Writes the WAL file for an `insert` or `remove` and calls `sync_all` (fsync).
///
/// Truncates any existing WAL before writing so that a previous incomplete
/// WAL cannot be mistaken for the new one.
///
/// The `value_bytes` slice must have exactly `value_size` bytes; pass
/// all-zeros for a `REMOVE` record.
pub fn write_and_sync(
    db_path: &Path,
    op: u8,
    key_bytes: &[u8],
    value_bytes: &[u8],
) -> Result<()> {
    let path = wal_path(db_path);

    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)
        .map_err(BTreeError::from)?;

    let key_size = key_bytes.len();
    let value_size = value_bytes.len();

    // Build the header: magic + key_size (u16 le) + value_size (u16 le) + op.
    let mut header = [0u8; WAL_HEADER_LEN];
    header[..4].copy_from_slice(&WAL_MAGIC);
    header[4..6].copy_from_slice(&(key_size as u16).to_le_bytes());
    header[6..8].copy_from_slice(&(value_size as u16).to_le_bytes());
    header[8] = op;

    file.write_all(&header).map_err(BTreeError::from)?;
    file.write_all(key_bytes).map_err(BTreeError::from)?;
    file.write_all(value_bytes).map_err(BTreeError::from)?;

    // fsync — must reach storage before any mmap mutation.
    file.sync_all().map_err(BTreeError::from)?;

    Ok(())
}

/// Deletes the WAL file. Returns `Ok(())` if the file is already absent.
pub fn delete(db_path: &Path) -> Result<()> {
    let path = wal_path(db_path);
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(BTreeError::from(e)),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn db_path(dir: &tempfile::TempDir) -> PathBuf {
        dir.path().join("tree.db")
    }

    #[test]
    fn wal_round_trip_insert() {
        let dir = tempfile::tempdir().unwrap();
        let db = db_path(&dir);
        let key: u32 = 42;
        let val: u64 = 999;
        write_and_sync(&db, WAL_OP_INSERT,
            bytemuck::bytes_of(&key), bytemuck::bytes_of(&val)).unwrap();

        let rec = read_existing(&db, 4, 8).unwrap().unwrap();
        assert_eq!(rec.op, WAL_OP_INSERT);
        assert_eq!(rec.key, bytemuck::bytes_of(&key));
        assert_eq!(rec.value, bytemuck::bytes_of(&val));
    }

    #[test]
    fn wal_round_trip_remove() {
        let dir = tempfile::tempdir().unwrap();
        let db = db_path(&dir);
        let key: i32 = -5;
        let zeros = vec![0u8; 8];
        write_and_sync(&db, WAL_OP_REMOVE,
            bytemuck::bytes_of(&key), &zeros).unwrap();

        let rec = read_existing(&db, 4, 8).unwrap().unwrap();
        assert_eq!(rec.op, WAL_OP_REMOVE);
        assert_eq!(rec.key, bytemuck::bytes_of(&key));
    }

    #[test]
    fn wal_missing_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let db = db_path(&dir);
        assert!(read_existing(&db, 4, 8).unwrap().is_none());
    }

    #[test]
    fn wal_truncated_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let db = db_path(&dir);
        // Write 3 bytes — shorter than WAL_HEADER_LEN + any K/V.
        std::fs::write(wal_path(&db), b"BWA").unwrap();
        assert!(read_existing(&db, 4, 8).unwrap().is_none());
    }

    #[test]
    fn wal_bad_magic_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let db = db_path(&dir);
        // Correct length but wrong magic.
        let buf = vec![0xFFu8; 9 + 4 + 8];
        std::fs::write(wal_path(&db), &buf).unwrap();
        assert!(read_existing(&db, 4, 8).unwrap().is_none());
    }

    #[test]
    fn wal_size_mismatch_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let db = db_path(&dir);
        // Write a valid WAL for key_size=4, val_size=8.
        let key: u32 = 1;
        let val: u64 = 2;
        write_and_sync(&db, WAL_OP_INSERT,
            bytemuck::bytes_of(&key), bytemuck::bytes_of(&val)).unwrap();
        // Try to read it back claiming key_size=8 (mismatch).
        assert!(read_existing(&db, 8, 8).unwrap().is_none());
    }

    #[test]
    fn wal_delete_removes_file() {
        let dir = tempfile::tempdir().unwrap();
        let db = db_path(&dir);
        let key: u32 = 1;
        let val: u32 = 2;
        write_and_sync(&db, WAL_OP_INSERT,
            bytemuck::bytes_of(&key), bytemuck::bytes_of(&val)).unwrap();
        assert!(wal_path(&db).exists());
        delete(&db).unwrap();
        assert!(!wal_path(&db).exists());
    }

    #[test]
    fn wal_delete_absent_is_ok() {
        let dir = tempfile::tempdir().unwrap();
        let db = db_path(&dir);
        // Should not error even when the file doesn't exist.
        delete(&db).unwrap();
    }
}
