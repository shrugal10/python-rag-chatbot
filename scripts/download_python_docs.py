from pathlib import Path
import urllib.request, zipfile, shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "documents"
ARCHIVE = ROOT / "data" / "python-docs-html.zip"
URL = "https://docs.python.org/3/archives/python-3.14-docs-html.zip"

OUT.mkdir(parents=True, exist_ok=True)
print("Downloading official Python 3.14 HTML documentation...")
urllib.request.urlretrieve(URL, ARCHIVE)

extract = ROOT / "data" / "_python_docs"
if extract.exists():
    shutil.rmtree(extract)
extract.mkdir(parents=True)

with zipfile.ZipFile(ARCHIVE) as z:
    z.extractall(extract)

print("Downloaded and extracted. For the PDF-based pipeline, install wkhtmltopdf/weasyprint")
print("or use the included plain-text conversion path below.")
print("The official archive itself is the corpus source; the RAG code can be adapted to ingest HTML directly.")
