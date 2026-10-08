"""Markdown-документ репозитория → PDF (A4): pandoc → HTML со стилями печати → Chrome без интерфейса.

Запуск из корня: .venv/bin/python tools/md_pdf.py METHODOLOGY.md [-o METHODOLOGY.pdf]
Нужны pandoc и Google Chrome. Номера страниц ставятся, если доступны pypdf и reportlab (иначе PDF без них).
Раздел «1. …» начинается с новой страницы: первая страница — заголовок, сводка и оглавление.
"""
from __future__ import annotations

import argparse
import io
import re
import subprocess
import tempfile
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """
@page { size: A4; margin: 17mm 16mm 19mm 16mm; }
html { font-size: 9.5pt; }
body { font-family: "PT Sans", "Helvetica Neue", Arial, sans-serif; color: #1d1d1f; line-height: 1.45;
       margin: 0; padding: 0 !important; max-width: none !important; }
header#title-block-header { display: none; }
h1 { font-size: 20pt; line-height: 1.2; margin: 0 0 6pt; }
h2 { font-size: 14pt; margin: 20pt 0 6pt; padding-bottom: 3pt; border-bottom: 1px solid #d0d4da; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
p { margin: 0 0 6pt; }
p:has(+ table), p:has(+ ul), p:has(+ ol), p:has(+ pre) { break-after: avoid; }
ul, ol { margin: 0 0 6pt; padding-left: 18pt; }
li { margin-bottom: 2pt; }
a { color: #1f5fbf; text-decoration: none; }
code { font-family: "PT Mono", Menlo, monospace; font-size: 8.6pt; background: #f2f4f7; padding: 0 2pt; border-radius: 2pt; }
pre { background: #f2f4f7; padding: 7pt 9pt; border-radius: 3pt; font-size: 8.3pt; line-height: 1.35;
      white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 10pt; font-size: 8.4pt; break-inside: avoid; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th, td { border: 1px solid #d0d4da; padding: 3pt 5pt; vertical-align: top; text-align: left; }
th { background: #eef1f5; font-weight: 600; }
img { max-width: 92%; max-height: 92mm; display: block; margin: 6pt auto 2pt; }
p:has(> img) { break-inside: avoid; break-after: avoid; }
em { color: #555; }
h2#содержание + ol { columns: 2; column-gap: 24pt; font-size: 9pt; }
h2#содержание + ol li { margin-bottom: 0; }
h2#коротко + ul { line-height: 1.4; }
h2#коротко + ul li { margin-bottom: 1pt; }
hr { display: none; }
blockquote { margin: 6pt 0 10pt; padding: 5pt 9pt; border-left: 3px solid #2c6fc9; background: #f4f7fb;
             break-inside: avoid; }
blockquote p { margin: 0; }
"""


def to_html(md: Path, root: Path) -> str:
    html = subprocess.run(["pandoc", str(md), "-f", "gfm", "-t", "html5", "-s",
                           "--metadata", f"pagetitle={md.stem}"], capture_output=True, text=True, check=True).stdout
    html = html.replace("</head>", f"<style>{CSS}</style></head>", 1)
    # картинки — абсолютные пути: HTML печатается из временной папки
    html = re.sub(r'src="(?!https?:|file:|/)([^"]+)"', lambda m: f'src="{(root / m.group(1)).resolve().as_uri()}"', html)
    # индексы в формулах вида y_t, y_{t−1}
    html = re.sub(r"\b([a-zA-Z])_\{([^}<]+)\}", r"\1<sub>\2</sub>", html)
    html = re.sub(r"\b([a-zA-Z])_([a-z0-9])\b", r"\1<sub>\2</sub>", html)
    # короткие таблицы не разрываются между страницами, длинные (больше 8 строк) — можно
    html = re.sub(r"<table>(.*?)</table>", lambda m: ('<table style="break-inside: auto">' if m.group(1).count("<tr") > 9
                                                      else "<table>") + m.group(1) + "</table>", html, flags=re.S)
    # первый нумерованный раздел — с новой страницы
    return re.sub(r'<h2 id="1-', '<h2 style="break-before: page" id="1-', html, count=1)


def number_pages(pdf: Path) -> bool:
    try:
        from pypdf import PdfReader, PdfWriter
        from reportlab.pdfgen import canvas
    except ImportError:
        return False
    reader = PdfReader(pdf)
    writer = PdfWriter()
    n = len(reader.pages)
    for i, page in enumerate(reader.pages, 1):
        w, h = float(page.mediabox.width), float(page.mediabox.height)
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=(w, h))
        c.setFont("Helvetica", 8)
        c.setFillGray(0.45)
        c.drawCentredString(w / 2, 28, f"{i} / {n}")
        c.save()
        buf.seek(0)
        page.merge_page(PdfReader(buf).pages[0])
        writer.add_page(page)
    with open(pdf, "wb") as f:
        writer.write(f)
    return True


def main(md: Path, out: Path) -> None:
    root = md.resolve().parent
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "doc.html"
        page.write_text(to_html(md, root))
        subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                        f"--print-to-pdf={out.resolve()}", page.as_uri()], capture_output=True, check=True)
    print(f"{out}: номера страниц {'есть' if number_pages(out) else 'нет (нужны pypdf и reportlab)'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("md", type=Path)
    p.add_argument("-o", "--out", type=Path)
    a = p.parse_args()
    main(a.md, a.out or a.md.with_suffix(".pdf"))
