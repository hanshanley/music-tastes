"""Refresh orchestration: private persistent data, staged reports, explicit status."""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .cache import fingerprint
from .paths import DATA, DERIVED, INTERIM, RAW, REPORTS, ROOT, atomic_json


def inventory() -> dict:
    artifacts = {}
    for folder in (RAW, INTERIM, DERIVED):
        for path in sorted(folder.glob("*")):
            if path.is_file():
                artifacts[str(path.relative_to(DATA))] = {"bytes": path.stat().st_size}
    complete = DATA / "last_complete.json"
    previous = json.loads(complete.read_text()) if complete.exists() else None
    if previous:
        previous = {key: previous.get(key) for key in (
            "run_id", "status", "chart_through", "completed_at", "output_signature"
        )}
    return {
        "python": sys.version.split()[0], "supported_python": sys.version_info >= (3, 11),
        "artifacts": artifacts,
        "last_complete": previous,
    }


def status() -> dict:
    result = inventory()
    print(json.dumps(result, indent=2))
    return result


def preflight() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("Use Python >=3.11 (recommended: uv run --python 3.12 --extra all music-tastes update).")
    missing = [
        name for name in ("sentence_transformers", "transformers", "torch", "langdetect", "wordfreq")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise SystemExit(f"Missing pipeline dependencies: {', '.join(missing)}. Install .[all,test].")


def validate_feature_completion() -> dict:
    import pandas as pd

    idx = pd.read_parquet(DERIVED / "lyrics_index.parquet")
    songs = pd.read_parquet(DERIVED / "songs_weighted.parquet")
    if not set(songs["song_id"]).issubset(set(idx["song_id"])):
        raise RuntimeError("Lyrics fetch has not attempted every chart song")
    eligible = idx[
        idx["has_lyrics"].fillna(False)
        & idx["is_english"].fillna(False)
        & ~idx["is_instrumental"].fillna(False)
    ]["song_id"]
    if len(eligible) < 50:
        raise RuntimeError("Insufficient usable lyrics; refusing to publish a complete refresh")
    for name, required in [
        ("lyric_features_method_a.parquet", [
            "n_tokens", "emb_relationship_margin", "emb_independence_axis", "emb_stance",
        ]),
        ("lyric_features_method_b.parquet", ["p_relationship_doc"]),
    ]:
        features = pd.read_parquet(DERIVED / name)
        if not set(required).issubset(features.columns):
            raise RuntimeError(f"Incomplete successful features: {name}")
        present = features.dropna(subset=required)["song_id"]
        if not set(eligible).issubset(set(present)):
            raise RuntimeError(f"Incomplete successful features: {name}")
    return {"songs": len(songs), "eligible": len(eligible)}


def publish(staged: Path, destination: Path, after_publish=None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidate = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    backup = candidate / "previous"
    incoming = candidate / "incoming"
    moved = False
    installed = False
    try:
        shutil.copytree(staged, incoming)
        if destination.exists():
            destination.replace(backup)
            moved = True
        incoming.replace(destination)
        installed = True
        if after_publish is not None:
            after_publish()
    except BaseException:
        if installed:
            destination.replace(incoming)
        if moved:
            backup.replace(destination)
        raise
    finally:
        shutil.rmtree(candidate)


def run(
    limit: int | None = None, workers: int = 4, offline: bool = False,
    use_api: bool = True, as_of: str | None = None, retry_after_days: float = 7,
) -> dict:
    preflight()
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "refresh.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another update is using this data directory.")
        return _run_locked(limit, workers, offline, use_api, as_of, retry_after_days)


def _run_locked(limit, workers, offline, use_api, as_of, retry_after_days=7) -> dict:
    from .lyrics_features import EMBED_MODEL, EMBED_REVISION
    from .stance_nli import MODEL, MODEL_REVISION, HYPOTHESES, DOC_WORDS, CHUNK_WORDS, MAX_CHUNKS

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = DATA / "runs" / run_id
    staged = run_dir / "reports"
    staged.mkdir(parents=True)
    manifest = {
        "run_id": run_id, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "limited": limit is not None, "offline": offline, "stages": {},
        "python": sys.version.split()[0],
        "models": {
            "nli": {"model": MODEL, "revision": MODEL_REVISION},
            "embedding": {"model": EMBED_MODEL, "revision": EMBED_REVISION},
        },
        "configuration": {
            "as_of": as_of, "genius_api_requested": use_api, "workers": workers,
            "retry_after_days": retry_after_days, "lrclib_fallback": True,
            "hypotheses": HYPOTHESES, "doc_words": DOC_WORDS,
            "chunk_words": CHUNK_WORDS, "max_chunks": MAX_CHUNKS,
        },
        "baseline": inventory(),
    }
    manifest_path = run_dir / "manifest.json"
    environment = {
        **os.environ, "MUSIC_TASTES_DATA_DIR": str(DATA),
        "MUSIC_TASTES_REPORTS_DIR": str(staged),
    }
    if offline:
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})

    def stage(name: str, *options: str) -> None:
        manifest["stages"][name] = "running"
        atomic_json(manifest_path, manifest)
        subprocess.run(
            [sys.executable, "-m", "music_tastes.cli", name, *options],
            cwd=ROOT, env=environment, check=True,
        )
        manifest["stages"][name] = "complete"
        atomic_json(manifest_path, manifest)

    try:
        stage("ingest-charts", "--offline" if offline else "--refresh",
              *(["--as-of", as_of] if as_of else []))
        stage("resolve-songs")
        stage("exposure")
        if offline:
            for name in ("nrc_vad", "nrc_emolex", "vader") + (
                ("nrc_vad_v21",) if limit is None else ()
            ):
                if not (RAW / "lexicons" / f"{name}.parquet").exists():
                    raise RuntimeError(f"Missing offline lexicon: {name}")
        else:
            stage("lexicons")
            options = [
                "--workers", str(workers), "--retry-failed", "--lrclib",
                "--retry-after-days", str(retry_after_days),
            ]
            if not use_api:
                options.append("--no-api")
            if limit:
                options += ["--limit", str(limit)]
            stage("fetch-lyrics", *options)
        limited = ["--limit", str(limit)] if limit else []
        stage("features-a", *limited)
        stage("stance-b", *limited)
        if limit:
            manifest["status"] = "partial"
            manifest["note"] = "Limited smoke run; reports and last-complete marker unchanged."
            return manifest
        manifest["population"] = validate_feature_completion()
        manifest["audio_status"] = (
            "historical_cached_only" if (DERIVED / "acoustic_features.parquet").exists()
            else "unavailable"
        )
        for name in (
            "language", "coverage", "trends", "vad-robustness", "validity", "prompt-robustness",
            "stance-composition", "confounds", "gold-set", "exhibit", "report",
        ):
            stage(name, *(["--offline"] if offline and name == "vad-robustness" else []))
        integrity = json.loads((INTERIM / "chart_integrity.json").read_text())
        manifest["chart_through"] = integrity["chart_through"]
        manifest["source_signature"] = fingerprint({}, [
            RAW / "billboard_hot100_all.json", RAW / "billboard_hot100_secondary.parquet",
        ])
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["output_signature"] = fingerprint(
            {}, sorted(path for path in staged.rglob("*") if path.is_file())
        )
        atomic_json(staged / "run_manifest.json", manifest)
        publish(staged, REPORTS, lambda: atomic_json(DATA / "last_complete.json", manifest))
        return manifest
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)[:500]
        raise
    finally:
        atomic_json(manifest_path, manifest)
        print(f"Update {manifest['status']}: {manifest_path}")
