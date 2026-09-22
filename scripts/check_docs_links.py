"""Check relative Markdown links against the repository tree."""

from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"!?(?:\[[^\]]*\])\(([^)\s]+)(?:\s+[^)]*)?\)")


def markdown_files() -> list[Path]:
    """Return public repository Markdown files covered by the link gate."""
    files = set(ROOT.glob("*.md"))
    files.update((ROOT / "docs").rglob("*.md"))
    files.update((ROOT / ".github").rglob("*.md"))
    files.add(ROOT / "site" / "README.md")
    return sorted(path for path in files if path.is_file())


def relative_target(source: Path, target: str) -> Path | None:
    """Resolve a local Markdown target, ignoring URLs and anchors."""
    parsed = urlsplit(unquote(target))
    if parsed.scheme or parsed.netloc or target.startswith(("//", "mailto:")):
        return None
    if not parsed.path:
        return None
    return (source.parent / parsed.path).resolve()


def main() -> int:
    """Return non-zero when a local Markdown link does not resolve."""
    failures: list[str] = []
    for source in markdown_files():
        text = source.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(text):
            target = relative_target(source, match.group(1))
            if target is not None and not target.exists():
                failures.append(
                    f"{source.relative_to(ROOT)}: {match.group(1)} "
                    f"(resolved to {target.relative_to(ROOT) if target.is_relative_to(ROOT) else target})"
                )
    if failures:
        print("Broken local Markdown links:")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    print(f"Markdown link check passed for {len(markdown_files())} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
