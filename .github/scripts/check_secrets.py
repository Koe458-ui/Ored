from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PATTERNS = {
    "Supabase secret key": re.compile(r"\bsb_secret_[A-Za-z0-9_-]{10,}"),
    "JWT (Supabase service_role / anon legacy key)": re.compile(
        r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    "AWS access key id": re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}"),
    "Cloudflare API token": re.compile(r"\bcf(ut|at|k)_[A-Za-z0-9_-]{30,}"),
    "Stripe / Razorpay live key": re.compile(r"\b(sk_live_|rk_live_|rzp_live_)[A-Za-z0-9]{10,}"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "OpenAI-style key": re.compile(r"\bsk-(proj-)?[A-Za-z0-9_-]{32,}"),
    "Slack token": re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}"),
    "credential assigned in clear": re.compile(
        r"(?i)\b[A-Z0-9_]*(SECRET|SERVICE_KEY|ACCESS_KEY|API_KEY|API_TOKEN|PASSWORD|PRIVATE_KEY)[A-Z0-9_]*"
        r"""\s*[:=]\s*["']?(?!YOUR_|PASTE|<|\$|%|\{|process\.env|os\.environ|env\.|self\.|cfg\.)"""
        r"""([A-Za-z0-9/+_.=-]{24,})(?![A-Za-z0-9/+_.=-]*\()"""),
}

FORBIDDEN_FILES = re.compile(
    r"(^|/)(\.env(\.[^/]*)?|\.dev\.vars|credentials\.json|secrets\.ya?ml|id_rsa|id_ed25519)$"
    r"|\.(pem|key|p12|pfx|jks)$")
ALLOWED_FILES = re.compile(r"(^|/)\.env\.example$")
SKIP = re.compile(r"(^|/)(public/js/vendor/|\.github/scripts/check_secrets\.py$)")
MAX_BYTES = 5 * 1024 * 1024


def tracked_files() -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
        return [f for f in out.decode("utf-8", "replace").split("\0") if f]
    except (OSError, subprocess.CalledProcessError):
        return [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]


def scan(files: list[str]) -> list[str]:
    problems = []
    for name in files:
        path = ROOT / name
        if FORBIDDEN_FILES.search(name) and not ALLOWED_FILES.search(name):
            problems.append(f"{name}: secret-type file must not be committed")
            continue
        if SKIP.search(name) or not path.is_file() or path.stat().st_size > MAX_BYTES:
            continue
        data = path.read_bytes()
        if b"\0" in data[:4096]:
            continue
        text = data.decode("utf-8", "replace")
        for number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    problems.append(f"{name}:{number}: looks like a {label}")
    return problems


def main(argv: list[str]) -> int:
    files = argv or tracked_files()
    problems = scan(files)
    for problem in problems:
        print(problem)
    if problems:
        print(f"\n{len(problems)} possible secret(s). Remove them, rotate the key, and keep secrets in "
              f"environment variables or Cloudflare / GitHub secrets.")
        return 1
    print(f"no secrets found in {len(files)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
