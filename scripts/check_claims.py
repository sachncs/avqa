"""Reject unsupported performance and integration claims in public copy."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_COPY = [ROOT / "README.md", ROOT / "site" / "src", ROOT / "docs"]
FORBIDDEN = (
    re.compile(r"FlashAttention-2", re.IGNORECASE),
    re.compile(r"sub[- ]quadratic", re.IGNORECASE),
    re.compile(r"O\(N\s*[·.]\s*log\s*N\)", re.IGNORECASE),
    re.compile(r"full coverage", re.IGNORECASE),
    re.compile(r"production[- ]grade", re.IGNORECASE),
)


def files_under(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(candidate for candidate in path.rglob("*") if candidate.is_file())


def main() -> int:
    violations: list[str] = []
    for root in PUBLIC_COPY:
        for path in files_under(root):
            if path.suffix not in {".md", ".tsx", ".ts", ".css"}:
                continue
            text = path.read_text(encoding="utf-8")
            for pattern in FORBIDDEN:
                if pattern.search(text):
                    violations.append(f"{path.relative_to(ROOT)}: {pattern.pattern}")
    if violations:
        print("Unsupported public claims found:")
        print("\n".join(f"- {item}" for item in violations))
        return 1
    print("Public claim checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
