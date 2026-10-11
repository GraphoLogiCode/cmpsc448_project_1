"""Turn a Markdown report into a PDF with the same name.

    python report/build_pdf.py                       # report/report_draft.md -> report/report_draft.pdf
    python report/build_pdf.py report/report.md      # any other file

The Markdown is converted to HTML with mistune and printed to PDF by a headless Chrome or Edge,
so the only requirements are the mistune package and one of the two browsers.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import mistune

BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]

CSS = """
@page { size: Letter; margin: 0.9in 0.95in 0.95in; @bottom-center { content: counter(page); font: 9pt "Segoe UI", sans-serif; color: #666; } }
body { font: 10.5pt/1.45 "Segoe UI", Arial, sans-serif; color: #111; }
h1 { font-size: 19pt; line-height: 1.2; margin: 0 0 4pt; }
h2 { font-size: 13.5pt; margin: 18pt 0 5pt; border-bottom: 0.6pt solid #bbb; padding-bottom: 2pt; break-after: avoid; }
h3 { font-size: 11pt; margin: 12pt 0 3pt; break-after: avoid; }
p { margin: 0 0 7pt; }
ul, ol { margin: 0 0 7pt; padding-left: 18pt; }
li { margin-bottom: 2.5pt; }
table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 9.2pt; line-height: 1.3; break-inside: avoid; }
th, td { border-bottom: 0.5pt solid #ccc; padding: 3pt 7pt; text-align: left; vertical-align: top; }
th { border-bottom: 0.9pt solid #555; font-weight: 600; }
img { display: block; margin: 8pt auto 3pt; max-width: 100%; break-inside: avoid; }
.caption { text-align: center; font-size: 9pt; color: #444; margin-bottom: 11pt; break-before: avoid; }
.byline { color: #333; margin-bottom: 2pt; }
code { font: 9.2pt Consolas, monospace; background: #f3f3f1; padding: 0 2pt; }
pre { background: #f3f3f1; padding: 6pt 8pt; font: 8.8pt/1.35 Consolas, monospace; break-inside: avoid; }
pre code { background: none; padding: 0; }
"""


def main():
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("report_draft.md")
    source = source.resolve()
    # escape=False lets the few HTML lines in the report (the byline and the figure captions) through.
    to_html = mistune.create_markdown(escape=False, plugins=["table", "strikethrough"])
    body = to_html(source.read_text(encoding="utf-8"))
    html = source.with_suffix(".html")          # next to the Markdown, so the relative image paths still work
    html.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>{source.stem}</title>"
                    f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if browser is None:
        raise SystemExit(f"no Chrome or Edge found; open {html} in a browser and print it to PDF")
    pdf = source.with_suffix(".pdf")
    with tempfile.TemporaryDirectory() as profile:      # a throwaway profile, so an open browser is left alone
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={profile}", f"--print-to-pdf={pdf}", html.as_uri()],
                       check=True, capture_output=True, timeout=120)
    html.unlink()
    print(f"wrote {pdf} ({pdf.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
