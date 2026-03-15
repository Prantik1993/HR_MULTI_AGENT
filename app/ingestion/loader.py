from __future__ import annotations
from pathlib import Path
from docx import Document
import pymupdf4llm

SEP = chr(10)

def load_file(path: Path) -> str:
    match path.suffix.lower():
        case ".pdf":
            return pymupdf4llm.to_markdown(str(path))
        case ".docx":
            return SEP.join(p.text for p in Document(path).paragraphs)
        case _:
            return path.read_text(encoding="utf-8")

def load_directory(dir_path: Path) -> list[tuple[str, Path]]:
    return [
        (load_file(f), f)
        for f in dir_path.rglob("*")
        if f.suffix.lower() in {".pdf", ".docx", ".txt"} and f.stat().st_size > 0
    ]
