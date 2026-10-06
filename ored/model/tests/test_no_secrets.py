from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCANNER = ROOT / ".github" / "scripts" / "check_secrets.py"


def _scanner():
    spec = importlib.util.spec_from_file_location("check_secrets", SCANNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_holds_no_secrets():
    scanner = _scanner()
    assert scanner.scan(scanner.tracked_files()) == []


def test_scanner_catches_real_shaped_keys(tmp_path, monkeypatch):
    scanner = _scanner()
    monkeypatch.setattr(scanner, "ROOT", tmp_path)
    (tmp_path / "leak.txt").write_text(
        "ORED_SB_SERVICE_KEY=sb_secret_" + "a1B2" * 8 + "\n"
        "ORED_R2_SECRET_ACCESS_KEY=" + "0f" * 32 + "\n"
        "set ORED_R2_SECRET_ACCESS_KEY=YOUR_SECRET_ACCESS_KEY\n"
        "ORED_AUTH_KEY: sb_publishable_" + "x" * 30 + "\n")
    (tmp_path / ".env").write_text("A=1\n")
    problems = scanner.scan(["leak.txt", ".env"])
    assert any("leak.txt:1" in p for p in problems) and any("leak.txt:2" in p for p in problems)
    assert not any("leak.txt:3" in p or "leak.txt:4" in p for p in problems)
    assert any(p.startswith(".env:") for p in problems)
