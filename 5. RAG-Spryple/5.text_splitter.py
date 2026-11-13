"""
Text Splitters and Chunking Strategies
Optimizing document chunks for RAG
"""

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
)
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Sample documents for testing
SAMPLE_TEXT = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train models. The algorithm learns to map inputs to outputs based on example input-output pairs.

Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. The algorithm discovers structure without predefined labels.

Common algorithms include:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Applications

Machine learning is used in many fields:
1. Image recognition
2. Natural language processing
3. Recommendation systems
4. Fraud detection
5. Autonomous vehicles
""".strip()

SAMPLE_CODE = '''
def quicksort(arr):
    """
    Quicksort implementation in Python.
    Time complexity: O(n log n) average, O(n²) worst case.
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def binary_search(arr, target):
    """
    Binary search implementation.
    Requires sorted array.
    Time complexity: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
'''


def recursive_splitter():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(SAMPLE_TEXT)

    print(f"Original length: {len(SAMPLE_TEXT)} chars")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Chunk sizes: {[len(c) for c in chunks]}")
    print(f"\nFirst chunk preview:\n{chunks[0][:200]}...")


def chunk_size_comparison():
    sizes = [200, 500, 1000]

    print("=== Chunk Size Comparison ===")
    for size in sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=size // 5
        )  # 20% overlap
        chunks = splitter.split_text(SAMPLE_TEXT)
        print(f" Size {size}: {len(chunks)} chunks")


def overlap_importance():
    text = "The quick brown fox jumps over the lazy dog. " * 10  # Repeated text

    # without overlap
    no_overlap = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)

    # with overlap
    with_overlap = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)

    chunks_no_overlap = no_overlap.split_text(text)
    chunks_with_overlap = with_overlap.split_text(text)

    print("Without overlap:")
    print(f"  Chunk 1 end: ...{chunks_no_overlap[0][-20:]}")
    print(f"  Chunk 2 start: {chunks_no_overlap[1][:20]}...")

    print("\nWith overlap:")
    print(f"  Chunk 1 end: ...{chunks_with_overlap[0][-20:]}")
    print(f"  Chunk 2 start: {chunks_with_overlap[1][:20]}...")


def markdown_splitter():
    headers_to_consider = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_consider)
    chunks = splitter.split_text(SAMPLE_TEXT)

    print(f"Markdown Splitter produced {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i} ---")
        print(f" Metadata: {chunk.metadata}\n")
        print(f" Content: {chunk.page_content[:200]}...\n")


def code_splitter():
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=500, chunk_overlap=50
    )
    chunks = python_splitter.split_text(SAMPLE_CODE)
    print(f"Code Splitter produced {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(chunk[:150] + "..." if len(chunk) > 150 else chunk)


def document_splitter():
    from pypdf import PdfReader
    from langchain_core.documents import Document

    pdf_path = "./docs/DataScienceSyllabusbyRamakrushna.pdf"
    reader = PdfReader(pdf_path)
    docs = [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": pdf_path, "page": i},
        )
        for i, page in enumerate(reader.pages)
    ]

    print(f"Loaded {len(docs)} documents from PDF.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # split the docs
    split_docs = splitter.split_documents(docs)

    print(f"Split into {len(split_docs)} chunks")
    print(f"\nFirst chunk metadata: {split_docs[0].metadata}")
    print(f"First chunk content: {split_docs[0].page_content[:200]}...")
    print(f"\nLast chunk metadata: {split_docs[-1].metadata}")


if __name__ == "__main__":
    print("=== Recursive Character Text Splitter ===")
    recursive_splitter()
    chunk_size_comparison()
    overlap_importance()
    print("=== Markdown Header Text Splitter ===")
    markdown_splitter()
    print("=== Code Splitter ===")
    code_splitter()
    print("=== Document Splitter from PDF ===")
    document_splitter()