from __future__ import annotations

from typing import Dict, List

import fitz  # PyMuPDF


def extract_pdf_content(file_bytes: bytes) -> List[Dict[str, object]]:
    """Extract textual and image metadata content from a PDF byte-stream."""
    document = fitz.open(stream=file_bytes, filetype="pdf")

    pages: List[Dict[str, object]] = []
    try:
        for page_index in range(len(document)):
            page = document[page_index]
            text = page.get_text("text")
            images_metadata = [
                {
                    "xref": image[0],
                    "width": image[2],
                    "height": image[3],
                    "colorspace": image[4],
                }
                for image in page.get_images(full=True)
            ]
            pages.append(
                {
                    "page_number": page_index + 1,
                    "text": text or "",
                    "images": images_metadata,
                }
            )
    finally:
        document.close()

    return pages


def normalize_text(text: str) -> str:
    """Basic whitespace normalization."""
    return " ".join(text.split())

