from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

import hermes_lcm.periodic_backup as periodic
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import load_externalized_payload


UTC = timezone.utc


def _engine(
    tmp_path: Path,
    *,
    name: str = "lcm.db",
    enabled: bool = False,
    destination: Path | None = None,
    payload_root: Path | None = None,
    interval_hours: float = 6.0,
    keep_last: int = 10,
) -> LCMEngine:
    return LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "database" / name),
            large_output_externalization_path=str(payload_root or tmp_path / "payloads"),
            periodic_backup_enabled=enabled,
            periodic_backup_interval_hours=interval_hours,
            periodic_backup_keep_last=keep_last,
            periodic_backup_path=str(destination or tmp_path / "periodic"),
        ),
        hermes_home=str(tmp_path / "home"),
    )


def _payload(path: Path, content: str, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "kind": "tool_result",
        "tool_call_id": "call-1",
        "session_id": "session",
        "content": content,
        "content_chars": len(content),
        "content_bytes": len(content.encode("utf-8")),
        **extra,
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)


def _placeholder(ref: str) -> str:
    return f"[Externalized tool output: tool_call_id=call-1; chars=7; bytes=7; ref={ref}]"


def _append(engine: LCMEngine, *, content: str, tool_calls=None, role: str = "tool") -> None:
    engine._store.append(
        "session",
        {
            "role": role,
            "content": content,
            "tool_calls": tool_calls,
            "timestamp": time.time(),
        },
    )
    engine._store.commit()


def _wait_for(predicate, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition did not become true before timeout")


def _generation_dirs(spec: periodic.PeriodicBackupSpec) -> list[Path]:
    if not spec.namespace.exists():
        return []
    return sorted(
        path
        for path in spec.namespace.iterdir()
        if path.is_dir() and path.name.startswith("lcm-periodic-") and not path.name.endswith(".partial")
    )


def _run_holding_backup(spec, acquired, release, result_queue) -> None:
    def fault(stage: str) -> None:
        if stage == "locked":
            acquired.set()
            if not release.wait(10):
                raise RuntimeError("test lock release timed out")

    result_queue.put(periodic.run_periodic_backup(spec, due_only=False, _fault=fault))


def test_periodic_config_defaults_env_and_strict_validation(monkeypatch):
    config = LCMConfig()
    assert config.periodic_backup_enabled is False
    assert config.periodic_backup_interval_hours == 6.0
    assert config.periodic_backup_keep_last == 10
    assert config.periodic_backup_path == ""

    monkeypatch.setenv("LCM_PERIODIC_BACKUP_ENABLED", "true")
    monkeypatch.setenv("LCM_PERIODIC_BACKUP_INTERVAL_HOURS", "1.5")
    monkeypatch.setenv("LCM_PERIODIC_BACKUP_KEEP_LAST", "3")
    monkeypatch.setenv("LCM_PERIODIC_BACKUP_PATH", "/tmp/synthetic-periodic")
    configured = LCMConfig.from_env()
    assert configured.periodic_backup_enabled is True
    assert configured.periodic_backup_interval_hours == 1.5
    assert configured.periodic_backup_keep_last == 3
    assert configured.periodic_backup_path == "/tmp/synthetic-periodic"

    invalid = [
        ("LCM_PERIODIC_BACKUP_ENABLED", "maybe"),
        ("LCM_PERIODIC_BACKUP_INTERVAL_HOURS", "nan"),
        ("LCM_PERIODIC_BACKUP_INTERVAL_HOURS", "0"),
        ("LCM_PERIODIC_BACKUP_INTERVAL_HOURS", "inf"),
        ("LCM_PERIODIC_BACKUP_KEEP_LAST", "0"),
        ("LCM_PERIODIC_BACKUP_KEEP_LAST", "1.5"),
    ]
    for key, value in invalid:
        monkeypatch.setenv("LCM_PERIODIC_BACKUP_ENABLED", "false")
        monkeypatch.setenv("LCM_PERIODIC_BACKUP_INTERVAL_HOURS", "6")
        monkeypatch.setenv("LCM_PERIODIC_BACKUP_KEEP_LAST", "10")
        monkeypatch.setenv(key, value)
        with pytest.raises(ValueError):
            LCMConfig.from_env()

    with pytest.raises(ValueError):
        LCMConfig(periodic_backup_enabled=1)
    with pytest.raises(ValueError):
        LCMConfig(periodic_backup_interval_hours=float("nan"))
    with pytest.raises(ValueError):
        LCMConfig(periodic_backup_keep_last=True)


def test_disabled_is_filesystem_noop_and_idle_scheduler_owns_clones(tmp_path):
    disabled = _engine(tmp_path / "disabled")
    disabled_spec = periodic.build_periodic_backup_spec(disabled)
    try:
        assert disabled._periodic_backup_registration.active is False
        assert not disabled_spec.destination_root.exists()
    finally:
        disabled.shutdown()
    assert periodic.active_periodic_backup_scheduler_count() == 0

    enabled = _engine(tmp_path / "enabled", enabled=True, interval_hours=0.0001)
    clone = enabled.clone_for_agent()
    try:
        assert enabled._periodic_backup_registration.active is True
        assert clone._periodic_backup_registration.active is True
        assert periodic.active_periodic_backup_scheduler_count() == 1
        spec = periodic.build_periodic_backup_spec(enabled)
        _wait_for(lambda: (spec.namespace / "latest-good.json").exists())
        first_count = len(_generation_dirs(spec))
        _wait_for(lambda: len(_generation_dirs(spec)) > first_count)
        clone.shutdown()
        assert periodic.active_periodic_backup_scheduler_count() == 1
    finally:
        enabled.shutdown()
    assert periodic.active_periodic_backup_scheduler_count() == 0
    assert not any(thread.name.startswith("lcm-periodic-backup-") for thread in threading.enumerate())


def test_conflicting_same_db_registration_fails_closed(tmp_path):
    shared_db_root = tmp_path / "shared"
    first = _engine(
        shared_db_root,
        enabled=True,
        destination=tmp_path / "destination",
        payload_root=tmp_path / "payload-a",
    )
    second = _engine(
        shared_db_root,
        enabled=True,
        destination=tmp_path / "destination",
        payload_root=tmp_path / "payload-b",
    )
    try:
        assert first._periodic_backup_registration.active is True
        assert second._periodic_backup_registration.active is False
        assert "conflicting" in second._periodic_backup_registration.error
        assert periodic.active_periodic_backup_scheduler_count() == 1
    finally:
        second.shutdown()
        first.shutdown()


def test_bundle_roundtrip_uses_exact_refs_and_existing_loader(tmp_path):
    payload_root = tmp_path / "payloads"
    _payload(payload_root / "content.json", "content payload")
    _payload(payload_root / "ingest.json", "ingest payload", kind="raw_content")
    _payload(payload_root / "tool-call.json", "nested tool-call payload")
    engine = _engine(tmp_path, payload_root=payload_root)
    try:
        _append(engine, content=_placeholder("content.json"))
        _append(
            engine,
            role="assistant",
            content='quoted example: "' + _placeholder("fake-quoted.json") + '"',
            tool_calls=[
                {
                    "id": "call-2",
                    "function": {
                        "arguments": json.dumps({"recovery": _placeholder("tool-call.json")})
                    },
                }
            ],
        )
        _append(
            engine,
            role="user",
            content=(
                "[Externalized LCM ingest payload: kind=raw_content; field=content; "
                "chars=14; bytes=14; ref=ingest.json]"
            ),
        )
        _append(engine, role="user", content="template: {{ " + _placeholder("fake-template.json") + " }}")
        spec = periodic.build_periodic_backup_spec(engine)
        source_before = hashlib.sha256(spec.source_db.read_bytes()).hexdigest()
        result = periodic.run_periodic_backup(spec, due_only=False)
        assert result["status"] == "ok"
        assert hashlib.sha256(spec.source_db.read_bytes()).hexdigest() == source_before

        generation = Path(result["generation"])
        manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["schema"] == periodic.BUNDLE_SCHEMA
        assert manifest["source_identity"]["sha256"] == spec.source_identity
        assert [item["basename"] for item in manifest["payloads"]] == [
            "content.json",
            "ingest.json",
            "tool-call.json",
        ]

        restore = tmp_path / "disposable-restore"
        shutil.copytree(generation, restore)
        with sqlite3.connect(restore / "lcm.sqlite3") as conn:
            assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
            stored = conn.execute("SELECT content FROM messages ORDER BY store_id LIMIT 1").fetchone()[0]
        ref = periodic.extract_all_externalized_payload_refs(stored)[0]
        loaded = load_externalized_payload(
            ref,
            config=SimpleNamespace(
                large_output_externalization_path=str(restore / "payloads")
            ),
        )
        assert loaded is not None
        assert loaded["content"] == "content payload"
    finally:
        engine.shutdown()


@pytest.mark.parametrize("mode", ["missing", "corrupt", "symlink", "hardlink"])
def test_missing_corrupt_and_symlink_payloads_reject_generation(tmp_path, mode):
    payload_root = tmp_path / "payloads"
    payload_root.mkdir()
    ref = "bad.json"
    if mode == "corrupt":
        (payload_root / ref).write_text("not-json", encoding="utf-8")
    elif mode == "symlink":
        target = tmp_path / "outside.json"
        _payload(target, "outside")
        (payload_root / ref).symlink_to(target)
    elif mode == "hardlink":
        target = tmp_path / "outside.json"
        _payload(target, "outside")
        os.link(target, payload_root / ref)
    engine = _engine(tmp_path, payload_root=payload_root)
    try:
        _append(engine, content=_placeholder(ref))
        spec = periodic.build_periodic_backup_spec(engine)
        result = periodic.run_periodic_backup(spec, due_only=False)
        assert result["status"] == "failed"
        assert not (spec.namespace / "latest-good.json").exists()
        assert _generation_dirs(spec) == []
    finally:
        engine.shutdown()


def test_payload_replaced_during_copy_is_rejected(tmp_path, monkeypatch):
    payload_root = tmp_path / "payloads"
    ref = "changing.json"
    _payload(payload_root / ref, "x" * (2 * 1024 * 1024))
    engine = _engine(tmp_path, payload_root=payload_root)
    try:
        _append(engine, content=_placeholder(ref))
        spec = periodic.build_periodic_backup_spec(engine)
        real_read = periodic.os.read
        replaced = False

        def replacing_read(fd, count):
            nonlocal replaced
            value = real_read(fd, count)
            if value and not replaced:
                replaced = True
                replacement = payload_root / "replacement.json"
                _payload(replacement, "replacement")
                os.replace(replacement, payload_root / ref)
            return value

        monkeypatch.setattr(periodic.os, "read", replacing_read)
        result = periodic.run_periodic_backup(spec, due_only=False)
        assert result["status"] == "failed"
        assert "changed during copy" in result["error"]
        assert not (spec.namespace / "latest-good.json").exists()
    finally:
        engine.shutdown()


def test_new_destination_and_namespace_entries_are_fsynced(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    try:
        spec = periodic.build_periodic_backup_spec(engine)
        fsynced: list[Path] = []
        real_fsync_directory = periodic._fsync_directory

        def recording_fsync(path):
            fsynced.append(Path(path))
            real_fsync_directory(path)

        monkeypatch.setattr(periodic, "_fsync_directory", recording_fsync)
        assert periodic.run_periodic_backup(spec, due_only=False)["status"] == "ok"
        assert spec.destination_root.parent in fsynced
        assert spec.destination_root in fsynced
        assert spec.namespace in fsynced
    finally:
        engine.shutdown()


def test_payload_copy_cancels_before_publication(monkeypatch, tmp_path):
    payload_root = tmp_path / "payloads"
    _payload(payload_root / "large.json", "x" * (2 * 1024 * 1024))
    engine = _engine(tmp_path, payload_root=payload_root)
    try:
        _append(engine, content=_placeholder("large.json"))
        spec = periodic.build_periodic_backup_spec(engine)
        cancel = threading.Event()
        entered = threading.Event()
        release = threading.Event()
        real_read = periodic.os.read
        blocked = False

        def blocking_read(fd, count):
            nonlocal blocked
            if count == 1024 * 1024 and not blocked:
                blocked = True
                entered.set()
                assert release.wait(5)
            return real_read(fd, count)

        monkeypatch.setattr(periodic.os, "read", blocking_read)
        results: list[dict] = []
        worker = threading.Thread(
            target=lambda: results.append(
                periodic.run_periodic_backup(spec, due_only=False, cancel=cancel)
            )
        )
        worker.start()
        assert entered.wait(5)
        cancel.set()
        release.set()
        worker.join(5)
        assert not worker.is_alive()
        assert results[0]["status"] == "cancelled"
        assert _generation_dirs(spec) == []
        assert not (spec.namespace / "latest-good.json").exists()
    finally:
        engine.shutdown()


def test_scheduler_registry_does_not_overlap_during_last_owner_shutdown(
    monkeypatch,
    tmp_path,
):
    first = _engine(tmp_path, enabled=True)
    second = _engine(tmp_path, enabled=False)
    entered = threading.Event()
    release = threading.Event()
    real_stop = periodic._Scheduler.stop

    def delayed_stop(scheduler):
        entered.set()
        assert release.wait(5)
        return real_stop(scheduler)

    monkeypatch.setattr(periodic._Scheduler, "stop", delayed_stop)
    unregister_results: list[bool] = []
    register_results = []
    unregister_thread = threading.Thread(
        target=lambda: unregister_results.append(
            periodic.unregister_periodic_backup(first._periodic_backup_registration)
        )
    )
    try:
        unregister_thread.start()
        assert entered.wait(5)
        second._config.periodic_backup_enabled = True
        register_thread = threading.Thread(
            target=lambda: register_results.append(
                periodic.register_periodic_backup(second)
            )
        )
        register_thread.start()
        time.sleep(0.05)
        assert register_results == []
        release.set()
        unregister_thread.join(5)
        register_thread.join(5)
        assert unregister_results == [True]
        assert register_results[0].active is True
        second._periodic_backup_registration = register_results[0]
        assert periodic.active_periodic_backup_scheduler_count() == 1
    finally:
        release.set()
        unregister_thread.join(5)
        second.shutdown()
        first.shutdown()


def test_two_process_lock_collision_and_release_on_exit(tmp_path):
    engine = _engine(tmp_path)
    try:
        spec = periodic.build_periodic_backup_spec(engine)
        context = multiprocessing.get_context("fork")
        acquired = context.Event()
        release = context.Event()
        queue = context.Queue()
        process = context.Process(
            target=_run_holding_backup,
            args=(spec, acquired, release, queue),
        )
        process.start()
        assert acquired.wait(10)
        contender = periodic.run_periodic_backup(spec, due_only=False)
        assert contender["status"] == "deferred_busy"
        release.set()
        process.join(15)
        assert process.exitcode == 0
        assert queue.get(timeout=2)["status"] == "ok"
        after_exit = periodic.run_periodic_backup(spec, due_only=False)
        assert after_exit["status"] == "ok"
        lock_path = spec.namespace / ".periodic-backup.lock"
        assert lock_path.exists()
    finally:
        engine.shutdown()


def test_same_basename_sources_share_root_without_state_collision(tmp_path):
    destination = tmp_path / "shared-destination"
    first = _engine(tmp_path / "one", destination=destination)
    second = _engine(tmp_path / "two", destination=destination)
    try:
        _append(first, content="first", role="user")
        _append(second, content="second", role="user")
        first_spec = periodic.build_periodic_backup_spec(first)
        second_spec = periodic.build_periodic_backup_spec(second)
        assert first_spec.source_db.name == second_spec.source_db.name == "lcm.db"
        assert first_spec.namespace != second_spec.namespace
        assert periodic.run_periodic_backup(first_spec, due_only=False)["status"] == "ok"
        assert periodic.run_periodic_backup(second_spec, due_only=False)["status"] == "ok"
        assert (first_spec.namespace / "latest-good.json").exists()
        assert (second_spec.namespace / "latest-good.json").exists()
        assert len(_generation_dirs(first_spec)) == 1
        assert len(_generation_dirs(second_spec)) == 1
    finally:
        second.shutdown()
        first.shutdown()


def test_due_restart_and_wall_clock_jumps(monkeypatch, tmp_path):
    engine = _engine(tmp_path, interval_hours=6.0)
    try:
        spec = periodic.build_periodic_backup_spec(engine)
        start = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
        monkeypatch.setattr(periodic, "_utc_now", lambda: start)
        assert periodic.run_periodic_backup(spec, due_only=True, now=start)["status"] == "ok"
        assert periodic._seconds_until_due(spec, now=start + timedelta(hours=3)) == pytest.approx(3 * 3600)
        assert periodic._seconds_until_due(spec, now=start - timedelta(days=2)) == pytest.approx(6 * 3600)
        assert periodic.run_periodic_backup(
            spec,
            due_only=True,
            now=start + timedelta(hours=3),
        )["status"] == "noop_not_due"
        monkeypatch.setattr(periodic, "_utc_now", lambda: start + timedelta(hours=7))
        assert periodic.run_periodic_backup(
            spec,
            due_only=True,
            now=start + timedelta(hours=7),
        )["status"] == "ok"
        assert len(_generation_dirs(spec)) == 2
    finally:
        engine.shutdown()


def test_scheduler_retries_failed_transaction_without_accepting_visible_pointer(
    monkeypatch,
    tmp_path,
):
    engine = _engine(tmp_path)
    try:
        spec = periodic.build_periodic_backup_spec(engine)
        calls: list[bool] = []
        completed = threading.Event()

        def fake_due(_spec, **_kwargs):
            return 0.0 if not calls else 999.0

        def fake_run(_spec, *, due_only=True, **_kwargs):
            calls.append(due_only)
            if len(calls) == 1:
                return {"ok": False, "status": "published_pointer_failed"}
            completed.set()
            return {"ok": True, "status": "ok"}

        monkeypatch.setattr(periodic, "_seconds_until_due", fake_due)
        monkeypatch.setattr(periodic, "run_periodic_backup", fake_run)
        monkeypatch.setattr(periodic, "_MAX_FAILURE_BACKOFF_SECONDS", 0.01)
        scheduler = periodic._Scheduler(spec)
        try:
            assert completed.wait(2)
        finally:
            assert scheduler.stop() is True
        assert calls[:2] == [True, False]
    finally:
        engine.shutdown()


def test_concurrent_sqlite_writer_yields_consistent_committed_snapshot(tmp_path):
    engine = _engine(tmp_path)
    try:
        source = Path(engine._store.db_path)
        engine._store.connection.execute(
            "CREATE TABLE writer_probe (sequence INTEGER PRIMARY KEY, batch INTEGER NOT NULL)"
        )
        engine._store.connection.commit()
        stop = threading.Event()
        commits: list[int] = []

        def writer() -> None:
            conn = sqlite3.connect(source, timeout=5)
            try:
                for batch in range(50):
                    if stop.is_set():
                        return
                    conn.execute("BEGIN")
                    base = batch * 10
                    conn.executemany(
                        "INSERT INTO writer_probe(sequence, batch) VALUES (?, ?)",
                        [(base + offset, batch) for offset in range(10)],
                    )
                    conn.commit()
                    commits.append(batch)
                    time.sleep(0.002)
            finally:
                conn.close()

        thread = threading.Thread(target=writer)
        thread.start()
        _wait_for(lambda: bool(commits))
        spec = periodic.build_periodic_backup_spec(engine)
        result = periodic.run_periodic_backup(spec, due_only=False)
        stop.set()
        thread.join(5)
        assert result["status"] == "ok"
        with sqlite3.connect(Path(result["generation"]) / "lcm.sqlite3") as restored:
            assert restored.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
            rows = restored.execute(
                "SELECT sequence, batch FROM writer_probe ORDER BY sequence"
            ).fetchall()
        assert rows
        assert len(rows) % 10 == 0
        assert [sequence for sequence, _batch in rows] == list(range(len(rows)))
        assert all(batch == sequence // 10 for sequence, batch in rows)
    finally:
        engine.shutdown()


def test_corrupted_staged_snapshot_is_rejected_before_publication(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    try:
        spec = periodic.build_periodic_backup_spec(engine)
        good = periodic.run_periodic_backup(spec, due_only=False)
        assert good["status"] == "ok"
        good_generation = Path(good["generation"])
        old_pointer = (spec.namespace / "latest-good.json").read_bytes()
        real_snapshot = periodic._snapshot_database

        def corrupt_snapshot(snapshot_spec, destination, *, cancel):
            real_snapshot(snapshot_spec, destination, cancel=cancel)
            with destination.open("r+b") as handle:
                handle.seek(0)
                handle.write(b"not sqlite")
                handle.flush()
                os.fsync(handle.fileno())

        monkeypatch.setattr(periodic, "_snapshot_database", corrupt_snapshot)
        rejected = periodic.run_periodic_backup(spec, due_only=False)
        assert rejected["status"] == "failed"
        assert good_generation.exists()
        assert (spec.namespace / "latest-good.json").read_bytes() == old_pointer
        assert len(_generation_dirs(spec)) == 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "stage",
    [
        "snapshot",
        "integrity",
        "manifest",
        "stage_fsync",
        "final_rename",
        "final_fsync",
        "pointer_write",
        "pointer_rename",
        "pointer_fsync",
    ],
)
def test_faults_never_prune_previous_good(stage, tmp_path):
    engine = _engine(tmp_path, keep_last=1)
    try:
        spec = periodic.build_periodic_backup_spec(engine)
        first = periodic.run_periodic_backup(spec, due_only=False)
        assert first["status"] == "ok"
        first_generation = Path(first["generation"])
        old_pointer = (spec.namespace / "latest-good.json").read_bytes()
        retention_called = False

        def fault(current: str) -> None:
            nonlocal retention_called
            if current == "retention":
                retention_called = True
            if current == stage:
                raise OSError(f"synthetic {stage} failure")

        failed = periodic.run_periodic_backup(spec, due_only=False, _fault=fault)
        assert failed["status"] in {"failed", "published_pointer_failed"}
        assert retention_called is False
        assert first_generation.exists()
        if stage != "pointer_fsync":
            assert (spec.namespace / "latest-good.json").read_bytes() == old_pointer
        else:
            assert failed["pointer_renamed"] is True
            assert failed["pointer_durability"] == "uncertain"
    finally:
        engine.shutdown()


@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_payload_fault_and_retention_delete_failure_stop_without_losing_latest(tmp_path, fail_at):
    payload_root = tmp_path / "payloads"
    _payload(payload_root / "payload.json", "payload")
    engine = _engine(tmp_path, payload_root=payload_root, keep_last=4)
    try:
        _append(engine, content=_placeholder("payload.json"))
        wide = periodic.build_periodic_backup_spec(engine)
        for _ in range(3):
            assert periodic.run_periodic_backup(wide, due_only=False)["status"] == "ok"
        prior = _generation_dirs(wide)

        def payload_fault(stage: str) -> None:
            if stage == "payload:payload.json":
                raise OSError("synthetic payload failure")

        failed_payload = periodic.run_periodic_backup(wide, due_only=False, _fault=payload_fault)
        assert failed_payload["status"] == "failed"
        assert _generation_dirs(wide) == prior

        narrow = replace(wide, keep_last=1)
        deletion_attempts = 0

        def retention_fault(stage: str) -> None:
            nonlocal deletion_attempts
            if stage.startswith("retention_delete:"):
                deletion_attempts += 1
                if deletion_attempts == fail_at:
                    raise OSError("synthetic retention deletion failure")

        result = periodic.run_periodic_backup(narrow, due_only=False, _fault=retention_fault)
        assert result["status"] == "ok_retention_failed"
        assert deletion_attempts == fail_at
        pointer = periodic._read_verified_pointer(narrow)
        assert pointer is not None
        assert pointer[0] == Path(result["generation"])
        assert len(_generation_dirs(narrow)) == 5 - fail_at
    finally:
        engine.shutdown()


def test_canonical_aliases_share_identity_and_symlink_namespace_is_rejected(tmp_path):
    engine = _engine(tmp_path / "source")
    try:
        spec = periodic.build_periodic_backup_spec(engine)
        db_alias = tmp_path / "db-alias"
        db_alias.symlink_to(spec.source_db)
        destination = tmp_path / "destination"
        destination.mkdir()
        destination_alias = tmp_path / "destination-alias"
        destination_alias.symlink_to(destination, target_is_directory=True)
        alias_engine = SimpleNamespace(
            _store=SimpleNamespace(db_path=db_alias),
            _config=LCMConfig(
                periodic_backup_path=str(destination_alias),
                large_output_externalization_path=str(tmp_path / "payloads"),
            ),
            _hermes_home="",
            backup_dir=lambda: tmp_path / "unused",
        )
        alias_spec = periodic.build_periodic_backup_spec(alias_engine)
        assert alias_spec.source_db == spec.source_db
        assert alias_spec.source_identity == spec.source_identity
        assert alias_spec.destination_root == destination.resolve()

        outside = tmp_path / "outside"
        outside.mkdir()
        alias_spec.namespace.symlink_to(outside, target_is_directory=True)
        result = periodic.run_periodic_backup(alias_spec, due_only=False)
        assert result["status"] == "failed"
        assert list(outside.iterdir()) == []
    finally:
        engine.shutdown()


def test_same_second_ids_identity_mismatch_and_foreign_entries(tmp_path):
    engine = _engine(tmp_path)
    try:
        spec = periodic.build_periodic_backup_spec(engine)
        fixed = datetime(2035, 6, 7, 8, 9, 10, tzinfo=UTC)
        first = periodic.run_periodic_backup(spec, due_only=False, now=fixed)
        second = periodic.run_periodic_backup(spec, due_only=False, now=fixed)
        assert first["status"] == second["status"] == "ok"
        assert Path(first["generation"]).name != Path(second["generation"]).name

        foreign = spec.namespace / "manual-do-not-touch"
        foreign.mkdir()
        (foreign / "sentinel").write_text("keep", encoding="utf-8")
        partial = spec.namespace / "lcm-periodic-hostile.partial"
        partial.mkdir()
        assert periodic.run_periodic_backup(
            replace(spec, keep_last=1), due_only=False
        )["status"] == "ok"
        assert (foreign / "sentinel").read_text(encoding="utf-8") == "keep"
        assert partial.exists()

        pointer_path = spec.namespace / "latest-good.json"
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        pointer["source_identity"]["sha256"] = "0" * 64
        pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
        before = set(spec.namespace.iterdir())
        assert periodic._seconds_until_due(spec) == 0.0
        failed = periodic.run_periodic_backup(spec, due_only=False)
        assert failed["status"] == "failed"
        assert "identity mismatch" in failed["error"]
        assert set(spec.namespace.iterdir()) == before
    finally:
        engine.shutdown()


def test_corrupt_latest_good_blocks_future_publication_and_pruning(tmp_path):
    engine = _engine(tmp_path, keep_last=2)
    try:
        spec = periodic.build_periodic_backup_spec(engine)
        first = periodic.run_periodic_backup(spec, due_only=False)
        second = periodic.run_periodic_backup(spec, due_only=False)
        assert first["status"] == second["status"] == "ok"
        latest = Path(second["generation"])
        with (latest / "lcm.sqlite3").open("r+b") as handle:
            handle.seek(0)
            handle.write(b"corrupt")
        before = set(spec.namespace.iterdir())
        result = periodic.run_periodic_backup(replace(spec, keep_last=1), due_only=False)
        assert result["status"] == "failed"
        assert set(spec.namespace.iterdir()) == before
        assert latest.exists()
        assert Path(first["generation"]).exists()
    finally:
        engine.shutdown()


def test_unsupported_locking_fails_closed_without_publication(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    try:
        spec = periodic.build_periodic_backup_spec(engine)
        monkeypatch.setattr(periodic, "_fcntl", None)
        result = periodic.run_periodic_backup(spec, due_only=False)
        assert result["status"] == "unsupported"
        assert _generation_dirs(spec) == []
        assert not (spec.namespace / "latest-good.json").exists()
    finally:
        engine.shutdown()
