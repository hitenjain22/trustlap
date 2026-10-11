"""Export checks: the committed artifacts and parity fixtures match the current code."""

import json
from pathlib import Path

import pytest

from trustlap.config import physics_hash, to_dict
from trustlap.export import ARTIFACTS, GOLDEN

GOLDEN_FILES = sorted(GOLDEN.glob("*.json"))


@pytest.mark.parametrize("path", GOLDEN_FILES, ids=[p.name for p in GOLDEN_FILES])
def test_golden_fixtures_match_current_physics(path: Path) -> None:
    meta = json.loads(path.read_text())
    assert meta["physics_hash"] == physics_hash(), (
        f"{path.name} is stale: physics config changed. Run `python -m trustlap.export --golden`."
    )


def test_exported_config_matches_code() -> None:
    exported = json.loads((ARTIFACTS / "config.json").read_text())
    assert exported == json.loads(json.dumps(to_dict())), "Run `python -m trustlap.export`."


def test_manifest_hashes_match_files() -> None:
    import hashlib

    manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
    for entry in manifest["files"]:
        data = (ARTIFACTS / entry["path"]).read_bytes()
        assert len(data) == entry["bytes"], entry["path"]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["path"]
