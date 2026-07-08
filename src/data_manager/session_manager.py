#!/usr/bin/env python3
import argparse
import datetime as _dt
import hashlib
import json
import os
import tempfile
from pathlib import Path


def now_iso():
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, type(default)) else default
    except Exception:
        return default


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def data_paths(data_root):
    root = Path(data_root)
    return {
        "root": root,
        "sessions": root / "sessions",
        "index": root / "index",
        "sessions_index": root / "index" / "sessions_index.json",
        "transfer_index": root / "index" / "transfer_index.json",
    }


def ensure_data_root(data_root):
    root = Path(data_root)
    for rel in [
        "sessions",
        "incoming",
        "exports",
        "cache",
        "temp",
        "diagnostics",
        "diagnostics/runtime",
        "diagnostics/legacy_unclassified",
        "transferred",
        "index",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)
    paths = data_paths(data_root)
    if not paths["sessions_index"].exists():
        write_json(paths["sessions_index"], {"created_by": "Codex", "sessions": []})
    if not paths["transfer_index"].exists():
        write_json(paths["transfer_index"], {"created_by": "Codex", "transfers": []})


def parse_env_file(path):
    result = {}
    p = Path(path)
    if not p.exists():
        return result
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            result[key] = value
    return result


def collect_stats(session_dir):
    session_dir = Path(session_dir)
    stats = {
        "file_count": 0,
        "total_bytes": 0,
        "video_count": 0,
        "audio_count": 0,
        "screenshot_count": 0,
        "log_count": 0,
        "metadata_count": 0,
    }
    if not session_dir.exists():
        return stats
    for path in session_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(session_dir).as_posix()
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        stats["file_count"] += 1
        stats["total_bytes"] += size
        if rel.startswith("video/"):
            stats["video_count"] += 1
        elif rel.startswith("audio/"):
            stats["audio_count"] += 1
        elif rel.startswith("screenshots/"):
            stats["screenshot_count"] += 1
        elif rel.startswith("logs/"):
            stats["log_count"] += 1
        elif rel.startswith("metadata/"):
            stats["metadata_count"] += 1
    return stats


def write_sha256s(session_dir):
    session_dir = Path(session_dir)
    output = session_dir / "SHA256SUMS.txt"
    rows = []
    if session_dir.exists():
        for path in sorted(session_dir.rglob("*")):
            if not path.is_file() or path == output:
                continue
            try:
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                rows.append(f"{h.hexdigest()}  {path.relative_to(session_dir).as_posix()}")
            except OSError:
                continue
    output.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def upsert_session_index(data_root, entry):
    paths = data_paths(data_root)
    index = load_json(paths["sessions_index"], {"created_by": "Codex", "sessions": []})
    sessions = index.get("sessions")
    if not isinstance(sessions, list):
        sessions = []
    replaced = False
    for i, item in enumerate(sessions):
        if isinstance(item, dict) and item.get("session_id") == entry["session_id"]:
            merged = dict(item)
            merged.update(entry)
            sessions[i] = merged
            replaced = True
            break
    if not replaced:
        sessions.append(entry)
    sessions.sort(key=lambda x: str(x.get("session_id", "")), reverse=True)
    index["created_by"] = index.get("created_by", "Codex")
    index["updated_at"] = now_iso()
    index["sessions"] = sessions
    write_json(paths["sessions_index"], index)


def update_session(root, data_root, session_id, session_type, state, write_hash=False):
    ensure_data_root(data_root)
    root = Path(root)
    session_dir = Path(data_root) / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    for rel in [
        "video/raw",
        "video/annotated",
        "video/thumbnails",
        "audio/raw_6ch",
        "audio/selected_channel",
        "audio/features",
        "screenshots",
        "logs/system",
        "logs/yolo",
        "logs/audio",
        "logs/qt",
        "logs/ptz",
        "logs/rid900",
        "logs/transfer",
        "metadata",
        "config",
    ]:
        (session_dir / rel).mkdir(parents=True, exist_ok=True)

    runtime_env = root / "configs" / "runtime.env"
    runtime_snapshot = session_dir / "config" / "runtime_config_snapshot.json"
    write_json(runtime_snapshot, {
        "created_by": "Codex",
        "updated_at": now_iso(),
        "source": str(runtime_env),
        "values": parse_env_file(runtime_env),
    })
    env_copy = session_dir / "config" / "runtime_config_snapshot.env"
    if runtime_env.exists():
        env_copy.write_text(runtime_env.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")

    if write_hash:
        write_sha256s(session_dir)
    elif not (session_dir / "SHA256SUMS.txt").exists():
        (session_dir / "SHA256SUMS.txt").write_text("", encoding="utf-8")

    stats = collect_stats(session_dir)
    manifest_path = session_dir / "manifest.json"
    manifest = load_json(manifest_path, {})
    if not manifest:
        manifest = {
            "created_by": "Codex",
            "session_id": session_id,
            "session_type": session_type,
            "created_at": now_iso(),
            "transfer_state": "LOCAL_ONLY",
        }
    manifest.update({
        "updated_at": now_iso(),
        "session_id": session_id,
        "session_type": session_type,
        "state": state,
        "data_root": str(data_root),
        "project_root": str(root),
        "session_dir": str(session_dir),
        "runtime_config_snapshot": "config/runtime_config_snapshot.json",
        "sha256s": "SHA256SUMS.txt",
        "stats": stats,
    })
    if state in {"stopped", "failed", "abnormal"}:
        manifest["ended_at"] = now_iso()
    write_json(manifest_path, manifest)

    summary_path = session_dir / "session_summary.json"
    summary = load_json(summary_path, {})
    if not summary:
        summary = {
            "created_by": "Codex",
            "session_id": session_id,
            "session_type": session_type,
            "started_at": manifest.get("created_at", now_iso()),
        }
    summary.update({
        "updated_at": now_iso(),
        "status": state,
        "session_dir": str(session_dir),
        **stats,
    })
    if state in {"stopped", "failed", "abnormal"}:
        summary["ended_at"] = now_iso()
    write_json(summary_path, summary)

    entry = {
        "session_id": session_id,
        "session_type": session_type,
        "state": state,
        "path": str(session_dir),
        "manifest": str(manifest_path),
        "summary": str(summary_path),
        "transfer_state": manifest.get("transfer_state", "LOCAL_ONLY"),
        "updated_at": now_iso(),
        **stats,
    }
    upsert_session_index(data_root, entry)
    return session_dir


def reindex(root, data_root):
    ensure_data_root(data_root)
    sessions_root = Path(data_root) / "sessions"
    count = 0
    for session_dir in sorted(sessions_root.glob("*")):
        if not session_dir.is_dir():
            continue
        session_id = session_dir.name
        parts = session_id.split("_")
        session_type = "_".join(parts[2:-1]) if len(parts) >= 4 else "unknown"
        manifest = load_json(session_dir / "manifest.json", {})
        state = str(manifest.get("state") or "indexed")
        update_session(root, data_root, session_id, session_type, state, write_hash=False)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["ensure", "register", "finish", "reindex"])
    parser.add_argument("--root", default=".")
    parser.add_argument("--data-root", default=os.environ.get("ANTI_UAV_DATA_ROOT") or os.environ.get("ANTIUAV_DATA_ROOT") or "/home/elf/AntiUAV_Data")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--session-type", default="realtime_camera")
    parser.add_argument("--state", default="created")
    parser.add_argument("--hash", action="store_true")
    args = parser.parse_args()

    if args.command == "ensure":
        ensure_data_root(args.data_root)
    elif args.command == "register":
        if not args.session_id:
            raise SystemExit("--session-id is required")
        update_session(args.root, args.data_root, args.session_id, args.session_type, args.state, args.hash)
    elif args.command == "finish":
        if not args.session_id:
            raise SystemExit("--session-id is required")
        update_session(args.root, args.data_root, args.session_id, args.session_type, args.state, True)
    elif args.command == "reindex":
        count = reindex(args.root, args.data_root)
        print(count)


if __name__ == "__main__":
    main()
