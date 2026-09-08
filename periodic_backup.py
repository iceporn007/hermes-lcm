"""Verified, self-contained periodic backups for one LCM SQLite store.

Automatic restore is intentionally absent.  A successful generation contains a
transactionally consistent SQLite image plus every externalized payload that the
staged image can actually load.  Publication, pointer update, and retention are
serialized by a nonblocking POSIX advisory lock scoped to the canonical source
DB namespace.  Other platforms fail closed; the existing manual backup and
rotate operations remain portable and unchanged.

The guarantee is for local filesystems.  ``flock`` and directory ``fsync`` do
not establish a distributed lock or durability contract on network filesystems.
Canonical source and destination paths resolve symlink aliases before identity
is computed.  Symlinks, reparse points, and multiply-linked payload files are
rejected; hardlink aliases of the source database are not claimed to share an
identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import quote
import uuid

from .externalize import (
    get_large_output_storage_dir,
    is_externalized_placeholder,
    load_externalized_payload,
)
from .ingest_protection import (
    extract_all_externalized_payload_refs,
    is_externalized_ingest_placeholder,
)


logger = logging.getLogger(__name__)

BUNDLE_SCHEMA = "lcm-periodic-backup/v1"
POINTER_SCHEMA = "lcm-periodic-backup-pointer/v1"
SOURCE_IDENTITY_VERSION = 1
_GENERATION_PREFIX = "lcm-periodic-"
_LOCK_NAME = ".periodic-backup.lock"
_POINTER_NAME = "latest-good.json"
_BACKUP_BUSY_TIMEOUT_SECONDS = 5.0
_BACKUP_MAX_SECONDS = 300.0
_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
_MAX_METADATA_BYTES = 4 * 1024 * 1024
_SCHEDULER_SHUTDOWN_TIMEOUT_SECONDS = 10.0
_MAX_FAILURE_BACKOFF_SECONDS = 300.0

try:  # POSIX only by contract; Windows automatic backups fail closed.
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised through capability tests
    _fcntl = None

FaultHook = Callable[[str], None]


class PeriodicBackupError(RuntimeError):
    """A generation cannot be safely produced or verified."""


class PeriodicBackupUnsupported(PeriodicBackupError):
    """Required local locking or durability primitives are unavailable."""


class PeriodicBackupCancelled(PeriodicBackupError):
    """Scheduler shutdown cancelled an in-progress snapshot."""


class PointerPublicationError(PeriodicBackupError):
    def __init__(self, message: str, *, renamed: bool):
        super().__init__(message)
        self.renamed = renamed


@dataclass(frozen=True)
class PeriodicBackupSpec:
    source_db: Path
    source_identity: str
    payload_root: Path
    destination_root: Path
    namespace: Path
    interval_seconds: float
    keep_last: int


@dataclass(frozen=True)
class PeriodicBackupRegistration:
    source_key: str
    owner: object
    active: bool
    error: str = ""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PeriodicBackupError("backup timestamp is not canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise PeriodicBackupError("backup timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise PeriodicBackupError("backup timestamp has no timezone")
    return parsed.astimezone(timezone.utc)


def _source_identity(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()


def build_periodic_backup_spec(engine) -> PeriodicBackupSpec:
    """Resolve immutable scheduler settings without creating backup state."""
    raw_db = str(engine._store.db_path)
    if raw_db == ":memory:":
        raise PeriodicBackupUnsupported("periodic backup does not support in-memory databases")
    source_db = Path(raw_db).expanduser().resolve(strict=True)
    source_stat = os.lstat(source_db)
    if stat.S_ISLNK(source_stat.st_mode) or not stat.S_ISREG(source_stat.st_mode):
        raise PeriodicBackupError("source database is not a regular file")

    configured_root = str(getattr(engine._config, "periodic_backup_path", "") or "")
    raw_root = Path(configured_root).expanduser() if configured_root else engine.backup_dir() / "periodic"
    destination_root = raw_root.resolve(strict=False)
    identity = _source_identity(source_db)
    payload_root = get_large_output_storage_dir(
        engine._config,
        hermes_home=str(getattr(engine, "_hermes_home", "") or ""),
        create=False,
    ).resolve(strict=False)
    return PeriodicBackupSpec(
        source_db=source_db,
        source_identity=identity,
        payload_root=payload_root,
        destination_root=destination_root,
        namespace=destination_root / identity,
        interval_seconds=float(engine._config.periodic_backup_interval_hours) * 3600.0,
        keep_last=int(engine._config.periodic_backup_keep_last),
    )


def _identity_payload(spec: PeriodicBackupSpec) -> dict[str, Any]:
    return {
        "version": SOURCE_IDENTITY_VERSION,
        "canonical_db_path": str(spec.source_db),
        "sha256": spec.source_identity,
    }


def _identity_matches(value: Any, spec: PeriodicBackupSpec) -> bool:
    return isinstance(value, dict) and value == _identity_payload(spec)


def _private_directory(path: Path, *, parents: bool = False) -> None:
    if parents:
        path.mkdir(parents=True, mode=0o700, exist_ok=True)
    else:
        path.mkdir(mode=0o700)
    observed = os.lstat(path)
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise PeriodicBackupError(f"backup path is not a plain directory: {path}")
    path.chmod(0o700)


def _prepare_private_directory_tree(path: Path) -> None:
    """Create each missing directory and durably publish its parent entry."""
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    for component in reversed(missing):
        _private_directory(component)
        _fsync_directory(component.parent)
    _private_directory(path, parents=True)


def _prepare_namespace(spec: PeriodicBackupSpec) -> None:
    _prepare_private_directory_tree(spec.destination_root)
    expected_root = spec.destination_root.resolve(strict=True)
    if spec.namespace.parent.resolve(strict=True) != expected_root:
        raise PeriodicBackupError("backup namespace escaped its configured root")
    namespace_existed = spec.namespace.exists()
    _private_directory(spec.namespace, parents=True)
    if not namespace_existed:
        _fsync_directory(spec.destination_root)
    if spec.namespace.resolve(strict=True).parent != expected_root:
        raise PeriodicBackupError("backup namespace identity changed during creation")


def _fsync_directory(path: Path) -> None:
    if os.name != "posix" or not hasattr(os, "O_DIRECTORY"):
        raise PeriodicBackupUnsupported("directory fsync is unsupported on this platform")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except (OSError, TypeError, NotImplementedError) as exc:
        raise PeriodicBackupUnsupported(f"cannot open directory for fsync: {path}: {exc}") from exc
    try:
        observed = os.fstat(fd)
        if not stat.S_ISDIR(observed.st_mode):
            raise PeriodicBackupUnsupported(f"fsync target is not a directory: {path}")
        try:
            os.fsync(fd)
        except (OSError, TypeError, NotImplementedError) as exc:
            raise PeriodicBackupUnsupported(f"directory fsync failed for {path}: {exc}") from exc
    finally:
        os.close(fd)


def _fsync_file(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        observed = os.fstat(fd)
        if not stat.S_ISREG(observed.st_mode):
            raise PeriodicBackupError(f"fsync target is not a regular file: {path}")
        os.fsync(fd)
    finally:
        os.close(fd)


def _create_private_file(path: Path) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
    finally:
        os.close(fd)


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    encoded = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write while publishing backup metadata")
            view = view[written:]
        os.fsync(fd)
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
    finally:
        os.close(fd)


def _read_json_regular(path: Path) -> dict[str, Any]:
    expected = os.lstat(path)
    if stat.S_ISLNK(expected.st_mode) or not stat.S_ISREG(expected.st_mode):
        raise PeriodicBackupError(f"metadata is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino)
            or opened.st_size > _MAX_METADATA_BYTES
        ):
            raise PeriodicBackupError(f"metadata identity or size is unsafe: {path}")
        raw = bytearray()
        while len(raw) <= _MAX_METADATA_BYTES:
            chunk = os.read(fd, min(1024 * 1024, _MAX_METADATA_BYTES + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        current = os.lstat(path)
        if (
            (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
            or current.st_size != opened.st_size
            or current.st_mtime_ns != opened.st_mtime_ns
            or len(raw) > _MAX_METADATA_BYTES
        ):
            raise PeriodicBackupError(f"metadata changed during read: {path}")
        value = json.loads(bytes(raw).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PeriodicBackupError(f"invalid metadata file: {path}") from exc
    finally:
        os.close(fd)
    if not isinstance(value, dict):
        raise PeriodicBackupError(f"metadata must be a JSON object: {path}")
    return value


def _sha256_file(
    path: Path,
    *,
    cancel: threading.Event | None = None,
) -> tuple[int, str]:
    expected = os.lstat(path)
    if stat.S_ISLNK(expected.st_mode) or not stat.S_ISREG(expected.st_mode):
        raise PeriodicBackupError(f"hashed backup entry is not a regular file: {path}")
    digest = hashlib.sha256()
    size = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
            raise PeriodicBackupError(f"hashed backup entry changed during open: {path}")
        while True:
            if cancel is not None and cancel.is_set():
                raise PeriodicBackupCancelled("periodic backup cancelled during verification")
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
        current = os.lstat(path)
        if (
            (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
            or current.st_size != opened.st_size
            or current.st_mtime_ns != opened.st_mtime_ns
            or size != opened.st_size
        ):
            raise PeriodicBackupError(f"hashed backup entry changed during read: {path}")
    finally:
        os.close(fd)
    return size, digest.hexdigest()


def _integrity_check(
    path: Path,
    *,
    cancel: threading.Event | None = None,
) -> None:
    uri = f"file:{quote(str(path), safe='/')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=_BACKUP_BUSY_TIMEOUT_SECONDS)
    try:
        if cancel is not None:
            conn.set_progress_handler(lambda: 1 if cancel.is_set() else 0, 1000)
        rows = conn.execute("PRAGMA integrity_check").fetchall()
    except sqlite3.OperationalError as exc:
        if cancel is not None and cancel.is_set():
            raise PeriodicBackupCancelled(
                "periodic backup cancelled during SQLite verification"
            ) from exc
        raise
    finally:
        conn.close()
    if rows != [("ok",)]:
        raise PeriodicBackupError(f"SQLite integrity_check rejected staged backup: {rows!r}")


def _snapshot_database(
    spec: PeriodicBackupSpec,
    destination: Path,
    *,
    cancel: threading.Event | None,
) -> None:
    source_before = os.lstat(spec.source_db)
    if stat.S_ISLNK(source_before.st_mode) or not stat.S_ISREG(source_before.st_mode):
        raise PeriodicBackupError("source database is not a stable regular file")
    _create_private_file(destination)
    source_uri = f"file:{quote(str(spec.source_db), safe='/')}?mode=ro"
    source = sqlite3.connect(source_uri, uri=True, timeout=_BACKUP_BUSY_TIMEOUT_SECONDS)
    target = sqlite3.connect(str(destination), timeout=_BACKUP_BUSY_TIMEOUT_SECONDS)
    deadline = time.monotonic() + _BACKUP_MAX_SECONDS

    def progress(_status: int, _remaining: int, _total: int) -> None:
        if cancel is not None and cancel.is_set():
            raise PeriodicBackupCancelled("periodic backup cancelled during SQLite snapshot")
        if time.monotonic() >= deadline:
            raise PeriodicBackupError("periodic SQLite snapshot exceeded its bounded deadline")

    try:
        source.execute(f"PRAGMA busy_timeout={int(_BACKUP_BUSY_TIMEOUT_SECONDS * 1000)}")
        target.execute(f"PRAGMA busy_timeout={int(_BACKUP_BUSY_TIMEOUT_SECONDS * 1000)}")
        source.backup(target, pages=128, progress=progress, sleep=0.05)
    finally:
        target.close()
        source.close()
    source_after = os.lstat(spec.source_db)
    if (source_after.st_dev, source_after.st_ino) != (
        source_before.st_dev,
        source_before.st_ino,
    ):
        raise PeriodicBackupError("source database path changed during snapshot")
    destination.chmod(0o600)
    _fsync_file(destination)


def _walk_exact_placeholders(value: Any) -> list[str]:
    refs: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            stripped = item.strip()
            if is_externalized_placeholder(stripped) or is_externalized_ingest_placeholder(stripped):
                for ref in extract_all_externalized_payload_refs(stripped):
                    if ref not in refs:
                        refs.append(ref)
            elif stripped.startswith(("{", "[")):
                try:
                    nested = json.loads(stripped)
                except json.JSONDecodeError:
                    return
                if not isinstance(nested, str):
                    visit(nested)
            return
        if isinstance(item, list):
            for nested in item:
                visit(nested)
            return
        if isinstance(item, dict):
            for nested in item.values():
                visit(nested)

    visit(value)
    return refs


def _enumerate_recovery_refs(
    staged_db: Path,
    *,
    cancel: threading.Event | None,
) -> list[str]:
    """Find loader-reachable refs, not broad regex lookalikes.

    Storage-produced content placeholders occupy the whole content value.
    Tool-call payloads may nest those exact strings in JSON content/arguments.
    Quoted examples, templates, log fragments, and arbitrary prose containing a
    marker are not loader-reachable placeholders and are therefore ignored.
    """
    uri = f"file:{quote(str(staged_db), safe='/')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=_BACKUP_BUSY_TIMEOUT_SECONDS)
    refs: list[str] = []
    try:
        if cancel is not None:
            conn.set_progress_handler(lambda: 1 if cancel.is_set() else 0, 1000)
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages'"
        ).fetchone()
        if table is None:
            raise PeriodicBackupError("staged database has no messages table")
        for content, tool_calls in conn.execute(
            "SELECT content, tool_calls FROM messages ORDER BY store_id ASC"
        ):
            if cancel is not None and cancel.is_set():
                raise PeriodicBackupCancelled(
                    "periodic backup cancelled during reference enumeration"
                )
            for ref in _walk_exact_placeholders(content):
                if ref not in refs:
                    refs.append(ref)
            if not isinstance(tool_calls, str) or not tool_calls:
                continue
            try:
                parsed = json.loads(tool_calls)
            except json.JSONDecodeError as exc:
                if "ref=" in tool_calls:
                    raise PeriodicBackupError(
                        "tool_calls contains an unresolved externalized reference"
                    ) from exc
                continue
            for ref in _walk_exact_placeholders(parsed):
                if ref not in refs:
                    refs.append(ref)
    finally:
        conn.close()
    return sorted(refs)


def _owned_regular_file(file_stat: os.stat_result) -> bool:
    geteuid = getattr(os, "geteuid", None)
    return geteuid is None or getattr(file_stat, "st_uid", None) in (None, geteuid())


def _copy_payload(
    spec: PeriodicBackupSpec,
    ref: str,
    destination: Path,
    *,
    cancel: threading.Event | None,
) -> dict[str, Any]:
    if not ref.endswith(".json") or Path(ref).name != ref or "/" in ref or "\\" in ref:
        raise PeriodicBackupError(f"invalid externalized payload reference: {ref!r}")
    if not spec.payload_root.exists():
        raise PeriodicBackupError(f"referenced payload store does not exist: {spec.payload_root}")
    expected_dir = os.lstat(spec.payload_root)
    if stat.S_ISLNK(expected_dir.st_mode) or not stat.S_ISDIR(expected_dir.st_mode):
        raise PeriodicBackupError("externalized payload store is not a plain directory")
    if spec.payload_root.resolve(strict=True) != spec.payload_root:
        raise PeriodicBackupError("externalized payload store changed canonical identity")

    dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    dir_fd = os.open(spec.payload_root, dir_flags)
    source_fd = -1
    destination_fd = -1
    try:
        opened_dir = os.fstat(dir_fd)
        if (opened_dir.st_dev, opened_dir.st_ino) != (expected_dir.st_dev, expected_dir.st_ino):
            raise PeriodicBackupError("externalized payload store changed during backup")
        before = os.stat(ref, dir_fd=dir_fd, follow_symlinks=False)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or getattr(before, "st_nlink", 1) != 1
            or not _owned_regular_file(before)
        ):
            raise PeriodicBackupError(f"referenced payload is not a safe regular file: {ref}")
        source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        source_fd = os.open(ref, source_flags, dir_fd=dir_fd)
        opened = os.fstat(source_fd)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise PeriodicBackupError(f"referenced payload changed during open: {ref}")

        destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        destination_fd = os.open(destination, destination_flags, 0o600)
        digest = hashlib.sha256()
        size = 0
        raw = bytearray()
        while True:
            if cancel is not None and cancel.is_set():
                raise PeriodicBackupCancelled("periodic backup cancelled during payload copy")
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > _MAX_PAYLOAD_BYTES:
                raise PeriodicBackupError(f"referenced payload exceeds loader limit: {ref}")
            digest.update(chunk)
            raw.extend(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise OSError("short write while copying externalized payload")
                view = view[written:]
        os.fsync(destination_fd)
        if hasattr(os, "fchmod"):
            os.fchmod(destination_fd, 0o600)
        after = os.stat(ref, dir_fd=dir_fd, follow_symlinks=False)
        if (
            (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or getattr(after, "st_ctime_ns", None) != getattr(opened, "st_ctime_ns", None)
            or size != opened.st_size
        ):
            raise PeriodicBackupError(f"referenced payload changed during copy: {ref}")
        try:
            decoded = json.loads(bytes(raw).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PeriodicBackupError(f"referenced payload is not valid JSON: {ref}") from exc
        if not isinstance(decoded, dict) or "content" not in decoded:
            raise PeriodicBackupError(f"referenced payload is not loader-compatible: {ref}")
        return {"basename": ref, "size": size, "sha256": digest.hexdigest()}
    except FileNotFoundError as exc:
        raise PeriodicBackupError(f"referenced payload is missing: {ref}") from exc
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)
        if source_fd >= 0:
            os.close(source_fd)
        os.close(dir_fd)


def _verify_payload_loader(
    payload_dir: Path,
    entries: list[dict[str, Any]],
    *,
    cancel: threading.Event | None = None,
) -> None:
    config = SimpleNamespace(large_output_externalization_path=str(payload_dir))
    for entry in entries:
        if cancel is not None and cancel.is_set():
            raise PeriodicBackupCancelled("periodic backup cancelled during loader verification")
        loaded = load_externalized_payload(str(entry["basename"]), config=config)
        if loaded is None or "content" not in loaded:
            raise PeriodicBackupError(
                f"existing payload loader rejected copied payload: {entry['basename']}"
            )


def _generation_manifest(
    spec: PeriodicBackupSpec,
    generation: Path,
    *,
    expected_generation_id: str | None = None,
) -> dict[str, Any]:
    manifest = _read_json_regular(generation / "manifest.json")
    expected_id = expected_generation_id or generation.name
    if manifest.get("schema") != BUNDLE_SCHEMA or manifest.get("generation_id") != expected_id:
        raise PeriodicBackupError("generation manifest schema or id mismatch")
    if not _identity_matches(manifest.get("source_identity"), spec):
        raise PeriodicBackupError("generation source identity mismatch")
    return manifest


def _verify_generation(
    spec: PeriodicBackupSpec,
    generation: Path,
    *,
    expected_generation_id: str | None = None,
    cancel: threading.Event | None = None,
) -> dict[str, Any]:
    namespace = spec.namespace.resolve(strict=True)
    observed = os.lstat(generation)
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise PeriodicBackupError("generation is not a plain directory")
    if generation.resolve(strict=True).parent != namespace:
        raise PeriodicBackupError("generation escaped its source namespace")
    manifest = _generation_manifest(
        spec,
        generation,
        expected_generation_id=expected_generation_id,
    )
    database = manifest.get("database")
    payloads = manifest.get("payloads")
    if not isinstance(database, dict) or not isinstance(payloads, list):
        raise PeriodicBackupError("generation manifest content is invalid")
    db_path = generation / "lcm.sqlite3"
    size, digest = _sha256_file(db_path, cancel=cancel)
    if database != {"basename": "lcm.sqlite3", "size": size, "sha256": digest, "integrity_check": "ok"}:
        raise PeriodicBackupError("generation SQLite metadata mismatch")
    _integrity_check(db_path, cancel=cancel)
    payload_dir = generation / "payloads"
    observed_payload_dir = os.lstat(payload_dir)
    if stat.S_ISLNK(observed_payload_dir.st_mode) or not stat.S_ISDIR(observed_payload_dir.st_mode):
        raise PeriodicBackupError("generation payload directory is unsafe")
    seen: set[str] = set()
    for entry in payloads:
        if not isinstance(entry, dict):
            raise PeriodicBackupError("generation payload manifest entry is invalid")
        name = entry.get("basename")
        if not isinstance(name, str) or Path(name).name != name or name in seen:
            raise PeriodicBackupError("generation payload basename is invalid or duplicated")
        seen.add(name)
        payload_path = payload_dir / name
        payload_stat = os.lstat(payload_path)
        if stat.S_ISLNK(payload_stat.st_mode) or not stat.S_ISREG(payload_stat.st_mode):
            raise PeriodicBackupError(f"generation payload is unsafe: {name}")
        item_size, item_digest = _sha256_file(payload_path, cancel=cancel)
        if entry != {"basename": name, "size": item_size, "sha256": item_digest}:
            raise PeriodicBackupError(f"generation payload metadata mismatch: {name}")
    actual = {
        child.name
        for child in payload_dir.iterdir()
        if child.is_file() and not child.is_symlink()
    }
    if actual != seen:
        raise PeriodicBackupError("generation payload set does not match manifest")
    _verify_payload_loader(payload_dir, payloads, cancel=cancel)
    _parse_utc(manifest.get("completed_at"))
    return manifest


def _read_verified_pointer(
    spec: PeriodicBackupSpec,
    *,
    cancel: threading.Event | None = None,
) -> tuple[Path, dict[str, Any]] | None:
    pointer_path = spec.namespace / _POINTER_NAME
    if not pointer_path.exists():
        return None
    pointer = _read_json_regular(pointer_path)
    if pointer.get("schema") != POINTER_SCHEMA or not _identity_matches(pointer.get("source_identity"), spec):
        raise PeriodicBackupError("latest-good pointer identity mismatch")
    generation_id = pointer.get("generation_id")
    if (
        not isinstance(generation_id, str)
        or not generation_id.startswith(_GENERATION_PREFIX)
        or Path(generation_id).name != generation_id
    ):
        raise PeriodicBackupError("latest-good pointer target is invalid")
    generation = spec.namespace / generation_id
    manifest = _verify_generation(spec, generation, cancel=cancel)
    if pointer.get("completed_at") != manifest.get("completed_at"):
        raise PeriodicBackupError("latest-good pointer timestamp mismatch")
    return generation, manifest


def _seconds_until_due(
    spec: PeriodicBackupSpec,
    *,
    now: datetime | None = None,
    cancel: threading.Event | None = None,
) -> float:
    if not spec.namespace.exists():
        return 0.0
    try:
        verified = _read_verified_pointer(spec, cancel=cancel)
    except (OSError, sqlite3.Error, PeriodicBackupError):
        return 0.0
    if verified is None:
        return 0.0
    completed = _parse_utc(verified[1]["completed_at"])
    wall_now = (now or _utc_now()).astimezone(timezone.utc)
    elapsed = max(0.0, (wall_now - completed).total_seconds())
    return max(0.0, spec.interval_seconds - elapsed)


class _NamespaceLock:
    def __init__(self, namespace: Path):
        self.namespace = namespace
        self.fd = -1

    def acquire(self) -> bool:
        if os.name != "posix" or _fcntl is None:
            raise PeriodicBackupUnsupported("POSIX flock is required for automatic periodic backup")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        self.fd = os.open(self.namespace / _LOCK_NAME, flags, 0o600)
        observed = os.fstat(self.fd)
        if not stat.S_ISREG(observed.st_mode) or getattr(observed, "st_nlink", 1) != 1:
            self.close()
            raise PeriodicBackupUnsupported("periodic backup lock inode is unsafe")
        try:
            _fcntl.flock(self.fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except BlockingIOError:
            self.close()
            return False
        except (OSError, TypeError, AttributeError) as exc:
            self.close()
            raise PeriodicBackupUnsupported(f"periodic backup flock is unavailable: {exc}") from exc
        return True

    def close(self) -> None:
        if self.fd >= 0:
            try:
                if _fcntl is not None:
                    _fcntl.flock(self.fd, _fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = -1

    def __enter__(self) -> "_NamespaceLock":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


def _cleanup_partial(path: Path) -> None:
    try:
        if path.exists() and not path.is_symlink():
            shutil.rmtree(path)
    except OSError:
        logger.warning("LCM periodic backup could not clean partial generation %s", path, exc_info=True)


def _call_fault(fault: FaultHook | None, stage: str) -> None:
    if fault is not None:
        fault(stage)


def _publish_pointer(
    spec: PeriodicBackupSpec,
    generation_id: str,
    completed_at: str,
    *,
    fault: FaultHook | None,
) -> None:
    pointer = {
        "schema": POINTER_SCHEMA,
        "source_identity": _identity_payload(spec),
        "generation_id": generation_id,
        "completed_at": completed_at,
    }
    tmp = spec.namespace / f".{_POINTER_NAME}.{uuid.uuid4().hex}.tmp"
    renamed = False
    try:
        _call_fault(fault, "pointer_write")
        _write_json_exclusive(tmp, pointer)
        _call_fault(fault, "pointer_rename")
        os.replace(tmp, spec.namespace / _POINTER_NAME)
        renamed = True
        _call_fault(fault, "pointer_fsync")
        _fsync_directory(spec.namespace)
    except Exception as exc:
        raise PointerPublicationError(str(exc), renamed=renamed) from exc
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _owned_generations(
    spec: PeriodicBackupSpec,
    *,
    cancel: threading.Event | None,
) -> list[tuple[datetime, Path]]:
    owned: list[tuple[datetime, Path]] = []
    for candidate in spec.namespace.iterdir():
        if not candidate.name.startswith(_GENERATION_PREFIX) or candidate.name.endswith(".partial"):
            continue
        try:
            observed = os.lstat(candidate)
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
                continue
            manifest = _verify_generation(spec, candidate, cancel=cancel)
            completed = _parse_utc(manifest.get("completed_at"))
        except PeriodicBackupCancelled:
            raise
        except (OSError, PeriodicBackupError):
            continue
        owned.append((completed, candidate))
    return sorted(owned, key=lambda item: (item[0], item[1].name), reverse=True)


def _apply_retention(
    spec: PeriodicBackupSpec,
    *,
    fault: FaultHook | None,
    cancel: threading.Event | None,
) -> tuple[list[str], str | None]:
    verified = _read_verified_pointer(spec, cancel=cancel)
    if verified is None:
        raise PeriodicBackupError("retention requires a verified latest-good target")
    protected = verified[0].resolve(strict=True)
    generations = _owned_generations(spec, cancel=cancel)
    keep: set[Path] = {path.resolve(strict=True) for _, path in generations[: spec.keep_last]}
    keep.add(protected)
    deleted: list[str] = []
    for _completed, candidate in reversed(generations):
        resolved = candidate.resolve(strict=True)
        if resolved in keep:
            continue
        try:
            expected = os.lstat(candidate)
            _verify_generation(spec, candidate, cancel=cancel)
            current = os.lstat(candidate)
            if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
                raise PeriodicBackupError("retention candidate changed before deletion")
            _call_fault(fault, f"retention_delete:{candidate.name}")
            shutil.rmtree(candidate)
            _fsync_directory(spec.namespace)
            deleted.append(candidate.name)
        except PeriodicBackupCancelled:
            raise
        except (OSError, PeriodicBackupError, PeriodicBackupUnsupported) as exc:
            return deleted, str(exc)
    return deleted, None


def run_periodic_backup(
    spec: PeriodicBackupSpec,
    *,
    due_only: bool = True,
    now: datetime | None = None,
    cancel: threading.Event | None = None,
    _fault: FaultHook | None = None,
) -> dict[str, Any]:
    """Attempt one locked backup transaction without touching the live DB.

    The only writes are under ``spec.namespace``.  No code path restores,
    renames, replaces, truncates, or deletes ``spec.source_db`` or the source
    payload directory.
    """
    started = now or _utc_now()
    partial: Path | None = None
    final: Path | None = None
    pointer_renamed = False
    try:
        if cancel is not None and cancel.is_set():
            return {"ok": False, "status": "cancelled"}
        _prepare_namespace(spec)
        lock = _NamespaceLock(spec.namespace)
        if not lock.acquire():
            return {"ok": False, "status": "deferred_busy", "namespace": spec.namespace}
        with lock:
            _call_fault(_fault, "locked")
            # Existing metadata is an ownership boundary.  Never overwrite a
            # corrupt or foreign pointer and then treat that as recovery.
            if (spec.namespace / _POINTER_NAME).exists():
                _read_verified_pointer(spec, cancel=cancel)
            if due_only and _seconds_until_due(spec, now=started, cancel=cancel) > 0:
                return {"ok": True, "status": "noop_not_due", "namespace": spec.namespace}
            if cancel is not None and cancel.is_set():
                return {"ok": False, "status": "cancelled", "namespace": spec.namespace}

            generation_id = f"{_GENERATION_PREFIX}{started.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:12]}"
            partial = spec.namespace / f"{generation_id}.partial"
            final = spec.namespace / generation_id
            _private_directory(partial)
            payload_dir = partial / "payloads"
            _private_directory(payload_dir)

            _call_fault(_fault, "snapshot")
            staged_db = partial / "lcm.sqlite3"
            _snapshot_database(spec, staged_db, cancel=cancel)
            _call_fault(_fault, "integrity")
            _integrity_check(staged_db, cancel=cancel)

            refs = _enumerate_recovery_refs(staged_db, cancel=cancel)
            payload_entries: list[dict[str, Any]] = []
            for ref in refs:
                _call_fault(_fault, f"payload:{ref}")
                payload_entries.append(
                    _copy_payload(
                        spec,
                        ref,
                        payload_dir / ref,
                        cancel=cancel,
                    )
                )
            _fsync_directory(payload_dir)
            _verify_payload_loader(payload_dir, payload_entries, cancel=cancel)

            database_size, database_hash = _sha256_file(staged_db, cancel=cancel)
            completed_at = _utc_text(_utc_now())
            manifest = {
                "schema": BUNDLE_SCHEMA,
                "generation_id": generation_id,
                "source_identity": _identity_payload(spec),
                "source_version": {
                    "format": SOURCE_IDENTITY_VERSION,
                    "implementation": "hermes-lcm-periodic-backup",
                },
                "started_at": _utc_text(started),
                "completed_at": completed_at,
                "database": {
                    "basename": "lcm.sqlite3",
                    "size": database_size,
                    "sha256": database_hash,
                    "integrity_check": "ok",
                },
                "payloads": payload_entries,
            }
            _call_fault(_fault, "manifest")
            _write_json_exclusive(partial / "manifest.json", manifest)
            _call_fault(_fault, "stage_fsync")
            _fsync_file(staged_db)
            for entry in payload_entries:
                _fsync_file(payload_dir / str(entry["basename"]))
            _fsync_directory(payload_dir)
            _fsync_directory(partial)
            _verify_generation(
                spec,
                partial,
                expected_generation_id=generation_id,
                cancel=cancel,
            )

            _call_fault(_fault, "final_rename")
            os.rename(partial, final)
            partial = None
            _call_fault(_fault, "final_fsync")
            _fsync_directory(spec.namespace)

            try:
                _publish_pointer(
                    spec,
                    generation_id,
                    completed_at,
                    fault=_fault,
                )
                pointer_renamed = True
            except PointerPublicationError as exc:
                # If os.replace succeeded but namespace fsync failed, pointer bytes
                # may already name the candidate.  Preserve old and new bundles,
                # skip retention, and let restart reverify rather than pretending
                # rollback can restore pointer durability.
                return {
                    "ok": False,
                    "status": "published_pointer_failed",
                    "generation": final,
                    "pointer_durability": "uncertain" if exc.renamed else "unchanged",
                    "pointer_renamed": exc.renamed,
                    "error": str(exc),
                }

            _call_fault(_fault, "retention")
            deleted, retention_error = _apply_retention(
                spec,
                fault=_fault,
                cancel=cancel,
            )
            if retention_error is not None:
                logger.warning(
                    "LCM periodic backup retention stopped after deletion failure "
                    "source=%s error=%s",
                    spec.source_db,
                    retention_error,
                )
                return {
                    "ok": True,
                    "status": "ok_retention_failed",
                    "generation": final,
                    "deleted": deleted,
                    "retention_error": retention_error,
                }
            return {
                "ok": True,
                "status": "ok",
                "generation": final,
                "deleted": deleted,
                "payload_count": len(payload_entries),
            }
    except PeriodicBackupUnsupported as exc:
        return {"ok": False, "status": "unsupported", "error": str(exc)}
    except PeriodicBackupCancelled as exc:
        return {"ok": False, "status": "cancelled", "error": str(exc)}
    except (OSError, sqlite3.Error, PeriodicBackupError, ValueError, TypeError) as exc:
        return {
            "ok": False,
            "status": "failed",
            "error": str(exc),
            "published": final is not None and final.exists(),
            "pointer_renamed": pointer_renamed,
        }
    finally:
        if partial is not None:
            _cleanup_partial(partial)


class _Scheduler:
    def __init__(self, spec: PeriodicBackupSpec):
        self.spec = spec
        self.owners: set[object] = set()
        self.cancel = threading.Event()
        self.condition = threading.Condition()
        self.thread = threading.Thread(
            target=self._run,
            name=f"lcm-periodic-backup-{spec.source_identity[:12]}",
            daemon=True,
        )
        self.thread.start()

    def _run(self) -> None:
        retry_delay = 0.0
        while not self.cancel.is_set():
            try:
                due_in = _seconds_until_due(self.spec, cancel=self.cancel)
            except Exception:
                due_in = 0.0
            retrying_failed_transaction = retry_delay > 0
            wait_for = retry_delay if retrying_failed_transaction else due_in
            if wait_for > 0:
                with self.condition:
                    self.condition.wait_for(self.cancel.is_set, timeout=wait_for)
                if self.cancel.is_set():
                    return
            # A failed transaction never advances this process's last-success
            # schedule.  Retry it even if an uncertain pointer rename is visible
            # in the same process; only a fresh scheduler/restart may recover
            # that pointer by full verification.
            result = run_periodic_backup(
                self.spec,
                due_only=not retrying_failed_transaction,
                cancel=self.cancel,
            )
            status = str(result.get("status") or "failed")
            if status in {"ok", "ok_retention_failed", "noop_not_due"}:
                retry_delay = 0.0
            elif status == "cancelled":
                return
            else:
                retry_delay = min(
                    _MAX_FAILURE_BACKOFF_SECONDS,
                    max(1.0, min(60.0, self.spec.interval_seconds / 4.0)),
                )
                logger.warning(
                    "LCM periodic backup deferred or failed status=%s source=%s error=%s",
                    status,
                    self.spec.source_db,
                    result.get("error", ""),
                )

    def stop(self) -> bool:
        self.cancel.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=_SCHEDULER_SHUTDOWN_TIMEOUT_SECONDS)
        return not self.thread.is_alive()


_REGISTRY_LOCK = threading.RLock()
_SCHEDULERS: dict[str, _Scheduler] = {}


def register_periodic_backup(engine) -> PeriodicBackupRegistration:
    """Reference-count one process-local scheduler for an enabled engine."""
    owner = object()
    if not bool(getattr(engine._config, "periodic_backup_enabled", False)):
        return PeriodicBackupRegistration("", owner, False)
    try:
        spec = build_periodic_backup_spec(engine)
    except Exception as exc:
        logger.warning("LCM periodic backup registration failed closed: %s", exc)
        return PeriodicBackupRegistration("", owner, False, str(exc))
    key = str(spec.source_db)
    with _REGISTRY_LOCK:
        scheduler = _SCHEDULERS.get(key)
        if scheduler is not None and scheduler.cancel.is_set():
            error = "previous periodic backup worker is still shutting down"
            logger.warning("LCM %s for %s", error, spec.source_db)
            return PeriodicBackupRegistration(key, owner, False, error)
        if scheduler is not None and scheduler.spec != spec:
            error = "conflicting periodic backup registration for canonical database"
            logger.warning("LCM %s %s", error, spec.source_db)
            return PeriodicBackupRegistration(key, owner, False, error)
        if scheduler is None:
            scheduler = _Scheduler(spec)
            _SCHEDULERS[key] = scheduler
        scheduler.owners.add(owner)
    return PeriodicBackupRegistration(key, owner, True)


def unregister_periodic_backup(registration: PeriodicBackupRegistration | None) -> bool:
    """Release an engine owner and stop the worker after the final owner exits."""
    if registration is None or not registration.active:
        return True
    with _REGISTRY_LOCK:
        current = _SCHEDULERS.get(registration.source_key)
        if current is None or registration.owner not in current.owners:
            return True
        current.owners.remove(registration.owner)
        if current.owners:
            return True
        # Keep the registry lock and entry until the old worker is confirmed
        # stopped.  Otherwise a concurrent engine registration can create a
        # second worker for the same canonical DB during this join window.
        stopped = current.stop()
        if stopped:
            _SCHEDULERS.pop(registration.source_key, None)
            return True
        logger.warning(
            "LCM periodic backup worker did not stop within %.1fs for %s; "
            "registration remains fail-closed",
            _SCHEDULER_SHUTDOWN_TIMEOUT_SECONDS,
            current.spec.source_db,
        )
        return False


def active_periodic_backup_scheduler_count() -> int:
    """Test/diagnostic visibility without exposing mutable registry state."""
    with _REGISTRY_LOCK:
        return len(_SCHEDULERS)
