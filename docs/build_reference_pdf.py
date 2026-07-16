#!/usr/bin/env python3
"""Render VRP_SYSTEM_REFERENCE.md (mermaid diagrams included) to a self-contained
HTML and a print-ready PDF, with a consistent colour scheme and clickable
§-cross-references for easy back-and-forth navigation.

Most markdown->PDF converters don't run mermaid's JavaScript, so the diagrams come
out as raw code or clipped. This script pre-renders them locally: it embeds the
mermaid library and the diagrams into one HTML file and prints it with headless
Chrome, which runs the JS and rasterises the SVGs. Your diagram text never leaves
the machine (only the mermaid library is fetched, and only if not already cached).

Colour scheme (also shown as a legend at the top of the output):
  - navy section headings with an accent bar   -> structure, easy to scan
  - magenta inline `code`                       -> identifiers (params/functions/symbols)
  - blue §N.M                                    -> clickable cross-reference to that section
  - amber boxes                                  -> callouts / guarantees / notes

Requirements:
    pip install markdown pymdown-extensions
    google-chrome / chromium on PATH (for the PDF step; the HTML alone is printable)

Usage:
    python docs/build_reference_pdf.py
"""
import re, html, shutil, subprocess, sys, urllib.request, pathlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
MD = REPO / "VRP_SYSTEM_REFERENCE.md"
OUT_HTML = REPO / "VRP_SYSTEM_REFERENCE.html"
OUT_PDF = REPO / "VRP_SYSTEM_REFERENCE.pdf"
MERMAID_JS = HERE / "mermaid.min.js"
MERMAID_URL = "https://cdn.jsdelivr.net/npm/mermaid@9.4.3/dist/mermaid.min.js"

CSS = """
:root {
  --ink: #1a1a1a; --navy: #1f3a5f; --navy2: #2c5282;
  --accent: #2b6cb0; --xref: #1a56b8; --xref-bg: #e8f0fe;
  --codefg: #a01a63; --codebg: #fbeef5;
  --callout-bd: #c8901a; --callout-bg: #fdf5e2;
  --rule: #e2e2e2; --tblhdr: #eaf1f8; --tblzebra: #f7f9fc;
}
@page { size: A4; margin: 14mm 14mm; }
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       font-size: 10.5pt; line-height: 1.45; color: var(--ink); max-width: 100%; }
h1 { font-size: 20pt; color: var(--navy); border-bottom: 3px solid var(--accent);
     padding-bottom: 4px; }
h2 { font-size: 15pt; color: var(--navy); margin-top: 1.5em; padding: 2px 0 3px 10px;
     border-left: 5px solid var(--accent); border-bottom: 1px solid var(--rule);
     page-break-after: avoid; }
h3 { font-size: 12.5pt; color: var(--navy2); page-break-after: avoid; }
h4 { font-size: 11pt; color: var(--navy2); page-break-after: avoid; }
code { font-family: SFMono-Regular, Consolas, Menlo, monospace; font-size: 9pt;
       color: var(--codefg); background: var(--codebg); padding: 1px 4px;
       border-radius: 3px; }
pre { background: #f6f8fa; border: 1px solid var(--rule); border-left: 4px solid var(--accent);
      border-radius: 5px; padding: 8px 10px; font-size: 8.6pt; line-height: 1.35;
      white-space: pre-wrap; word-wrap: break-word; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: inherit; color: var(--ink); }
table { border-collapse: collapse; width: 100%; font-size: 8.8pt; margin: 0.6em 0;
        page-break-inside: avoid; }
th, td { border: 1px solid #d0d0d0; padding: 3px 6px; text-align: left; vertical-align: top; }
th { background: var(--tblhdr); color: var(--navy); }
tbody tr:nth-child(even) { background: var(--tblzebra); }
blockquote { border-left: 4px solid var(--callout-bd); background: var(--callout-bg);
             margin: 0.7em 0; padding: 4px 12px; color: #4a3a12; page-break-inside: avoid; }
a { color: var(--accent); text-decoration: none; }
hr { border: none; border-top: 1px solid var(--rule); margin: 1.2em 0; }
/* clickable cross-reference badge */
a.xref, span.xref { color: var(--xref); background: var(--xref-bg); border-radius: 3px;
                    padding: 0 3px; font-weight: 600; white-space: nowrap; }
/* colour legend banner (injected) */
.legend { border: 1px solid var(--rule); border-radius: 6px; background: #fafbfc;
          padding: 8px 12px; margin: 0 0 1.4em; font-size: 9pt; }
.legend b { color: var(--navy); }
.legend .sw { display: inline-block; padding: 0 5px; border-radius: 3px; font-weight: 600; }
/* mermaid: hide raw source until rendered, then scale to fit one page */
pre.mermaid { background: none; border: none; padding: 0; text-align: center;
              page-break-inside: avoid; white-space: normal; margin: 1em 0; }
pre.mermaid svg { max-width: 100% !important; max-height: 235mm !important;
                  height: auto !important; }
"""

LEGEND = (
    '<div class="legend"><b>Colour key</b> &nbsp; '
    '<span class="sw" style="color:#1f3a5f">■ section heading</span> &nbsp; '
    '<code>code identifier</code> (params / functions / symbols) &nbsp; '
    '<span class="xref">§N.M</span> clickable cross-reference &nbsp; '
    '<span class="sw" style="background:#fdf5e2;border-left:4px solid #c8901a;">callout / '
    'guarantee / note</span></div>'
)


def _heading_map(body):
    """number (e.g. '10.1', '0', 'A') -> heading anchor id, from generated <hN id=...>."""
    m = {}
    for h in re.finditer(r'<h[1-6][^>]*\sid="([^"]+)"[^>]*>(.*?)</h[1-6]>', body, re.S):
        anchor = h.group(1)
        text = html.unescape(re.sub(r'<[^>]+>', '', h.group(2))).strip()
        num = re.match(r'§?\s*(\d+(?:\.\d+)?)\b', text)
        if num:
            m.setdefault(num.group(1), anchor)
        elif text.startswith('Appendix A'):
            m.setdefault('A', anchor)
    return m


def main():
    try:
        import markdown
    except ImportError:
        sys.exit("Install deps first:  pip install markdown pymdown-extensions")

    if not MERMAID_JS.exists():
        print("fetching mermaid library (one-time)...")
        MERMAID_JS.write_bytes(urllib.request.urlopen(MERMAID_URL, timeout=60).read())

    src = MD.read_text()

    # Pull out ```mermaid fenced blocks so the markdown converter leaves them alone.
    blocks = []
    src = re.sub(
        r"```mermaid\n(.*?)```",
        lambda m: blocks.append(m.group(1)) or f"\n\nMERMAIDBLOCK{len(blocks)-1}ENDBLOCK\n\n",
        src, flags=re.S)

    body = markdown.markdown(
        src, extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list", "md_in_html"])

    hmap = _heading_map(body)

    # (1) Repoint any anchor whose visible text starts with a section number to the
    #     real generated id (keeps the TOC clickable regardless of slug differences).
    def _fix_anchor(m):
        href, text = m.group(1), m.group(2)
        plain = html.unescape(re.sub(r'<[^>]+>', '', text)).strip()
        num = re.match(r'§?\s*(\d+(?:\.\d+)?)\b', plain)
        key = num.group(1) if num else ('A' if plain.startswith('Appendix A') else None)
        if key in hmap:
            return f'<a href="#{hmap[key]}">{text}</a>'
        return m.group(0)
    body = re.sub(r'<a href="#([^"]*)">(.*?)</a>', _fix_anchor, body, flags=re.S)

    # (2) Turn inline "§N" / "§N.M" references in prose into clickable badges — but
    #     protect existing links and headings first so we never nest anchors.
    stash = []
    body = re.sub(r'(<a\b.*?</a>|<h[1-6][^>]*>.*?</h[1-6]>)',
                  lambda m: stash.append(m.group(0)) or f"@@P{len(stash)-1}@@", body, flags=re.S)

    def _xref(m):
        num = m.group(1)
        if num in hmap:
            return f'<a class="xref" href="#{hmap[num]}">§{num}</a>'
        return f'<span class="xref">§{num}</span>'
    body = re.sub(r'§(\d+(?:\.\d+)?)', _xref, body)
    body = re.sub(r'@@P(\d+)@@', lambda m: stash[int(m.group(1))], body)

    # Restore mermaid diagrams (HTML-escaped so textContent keeps <br/> line breaks).
    body = re.sub(r"<p>MERMAIDBLOCK(\d+)ENDBLOCK</p>",
                  lambda m: f'<pre class="mermaid">{html.escape(blocks[int(m.group(1))])}</pre>',
                  body)

    doc = (f'<!doctype html>\n<html><head><meta charset="utf-8"><style>{CSS}</style></head>\n'
           f'<body>\n{LEGEND}\n{body}\n<script>{MERMAID_JS.read_text()}</script>\n'
           "<script>mermaid.initialize({ startOnLoad: true, securityLevel: 'loose', "
           "theme: 'neutral', flowchart: { useMaxWidth: true, htmlLabels: true } });</script>\n"
           "</body></html>")
    OUT_HTML.write_text(doc)

    # accuracy check: every internal href must resolve to a real id
    ids = set(re.findall(r'\sid="([^"]+)"', doc))
    broken = sorted({h for h in re.findall(r'href="#([^"]+)"', doc) if h and h not in ids})
    print(f"wrote {OUT_HTML}  ({len(blocks)} diagrams, {len(hmap)} sections, "
          f"{len(broken)} broken links)")
    if broken:
        print("  BROKEN:", broken[:10])

    chrome = next((c for c in ("google-chrome", "chromium", "chromium-browser",
                               "google-chrome-stable") if shutil.which(c)), None)
    if not chrome:
        print("No Chrome/Chromium found — open the HTML and Ctrl+P -> Save as PDF.")
        return
    subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
                    "--virtual-time-budget=25000", "--run-all-compositor-stages-before-draw",
                    "--no-pdf-header-footer", f"--print-to-pdf={OUT_PDF}",
                    OUT_HTML.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
