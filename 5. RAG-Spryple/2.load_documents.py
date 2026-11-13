# Pypdf loader

from dotenv import load_dotenv
load_dotenv()

from pypdf import PdfReader
from langchain_core.documents import Document

def pdf_loader(pdf_path: str):
    reader = PdfReader(pdf_path)
    documents = [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": pdf_path, "page": i},
        )
        for i, page in enumerate(reader.pages)
    ]
    print(f"Loaded {len(documents)} pages from {pdf_path}")


if __name__ == "__main__":
    pdf_path = "./docs/DataScienceSyllabusbyRamakrushna.pdf"
    pdf_loader(pdf_path)
