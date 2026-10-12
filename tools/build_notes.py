"""Convert lecture notes (Markdown) to PDF: pandoc → HTML (MathML) → Chromium.

Provenance: new content (course tooling).

Chromium renders MathML and right-to-left Persian text correctly, which is why
this route is used instead of LaTeX or LibreOffice. Fonts (Vazirmatn for
Persian, Noto Sans for English) are loaded from Google Fonts by default.

Usage:
    python tools/build_notes.py notes/en/week01_notes_EN.md
    python tools/build_notes.py notes/fa/week01_notes_FA.md --chrome /path/to/chrome

Requirements: pandoc (>= 2.11) and Chrome/Chromium on the PATH (or --chrome).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Final

STYLE_FILE: Final[Path] = Path(__file__).with_name("notes_style.css")
GOOGLE_FONTS_CSS: Final[str] = (
    "https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700"
    "&family=Noto+Sans:ital,wght@0,400;0,700;1,400&family=Noto+Sans+Mono&display=swap"
)
CHROME_CANDIDATES: Final[tuple[str, ...]] = (
    "chromium",
    "chromium-browser",
    "google-chrome",
    "google-chrome-stable",
    "chrome",
)


def find_chrome(explicit: str | None) -> str:
    """Return the Chrome/Chromium executable to use.

    Args:
        explicit: Path given on the command line, or None to search the PATH.

    Raises:
        FileNotFoundError: If no browser is found.
    """
    if explicit:
        return explicit
    for name in CHROME_CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    raise FileNotFoundError("Chrome/Chromium not found; pass --chrome /path/to/chrome")


def markdown_to_html(source: Path, html: Path, font_css: str) -> None:
    """Render ``source`` to a standalone HTML file with MathML equations."""
    subprocess.run(
        [
            "pandoc",
            str(source),
            "--standalone",
            "--mathml",
            "--embed-resources",
            "--css",
            font_css,
            "--css",
            str(STYLE_FILE),
            "-o",
            str(html),
        ],
        check=True,
        cwd=source.parent,
    )


def html_to_pdf(chrome: str, html: Path, pdf: Path) -> None:
    """Print ``html`` to ``pdf`` with headless Chromium (A4, no header/footer)."""
    subprocess.run(
        [
            chrome,
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf}",
            html.as_uri(),
        ],
        check=True,
        capture_output=True,
    )


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--chrome", default=None, help="Chrome/Chromium executable")
    parser.add_argument(
        "--font-css",
        default=GOOGLE_FONTS_CSS,
        help="font stylesheet (URL or local file); default: Google Fonts",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output = (args.output or source.with_suffix(".pdf")).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        html = Path(tmp) / (source.stem + ".html")
        markdown_to_html(source, html, args.font_css)
        html_to_pdf(find_chrome(args.chrome), html, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
