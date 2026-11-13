# pip install rank_bm25

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from typing import List

from pydantic import ConfigDict
from rank_bm25 import BM25Okapi

from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()



class BM25Retriever(BaseRetriever):
    '''Minimal BM25 retriever (replaces langchain_community's BM25Retriever)'''
    model_config = ConfigDict(arbitrary_types_allowed=True)

    docs: List[Document]
    bm25: BM25Okapi
    k: int = 4

    @classmethod
    def from_documents(cls, documents, k: int = 4):
        tokenized = [d.page_content.lower().split() for d in documents]
        return cls(docs=documents, bm25=BM25Okapi(tokenized), k=k)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        scores = self.bm25.get_scores(query.lower().split())
        top = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[: self.k]
        return [self.docs[i] for i in top]


embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Documents with both semantic content AND specific identifiers
documents = [
    Document(
        page_content='Product SKU-7742X is our flagship router. It supports '
                     'gigabit speeds and advanced QoS features.',
        metadata={'type': 'product'}
    ),
    Document(
        page_content='For network connectivity issues, first check the '
                     'ethernet cable and router status lights.',
        metadata={'type': 'troubleshooting'}
    ),
    Document(
        page_content='Error code E_CONN_REFUSED indicates the server '
                     'rejected the connection. Check firewall settings.',
        metadata={'type': 'error'}
    ),
    Document(
        page_content='Product SKU-9981B is our budget mesh Wi-Fi extender. '
                     'It covers up to 2,500 sq ft with dual-band support.',
        metadata={'type': 'product'}
    ),
    Document(
        page_content='If your Wi-Fi signal is weak, move the router to a '
                     'central location and away from thick walls.',
        metadata={'type': 'troubleshooting'}
    ),
    Document(
        page_content='Error code E_TIMEOUT_504 means the gateway did not '
                     'receive a response in time. Check your DNS settings.',
        metadata={'type': 'error'}
    ),
    Document(
        page_content='Firmware version FW-3.2.1 fixes a critical security '
                     'vulnerability and improves VPN throughput.',
        metadata={'type': 'firmware'}
    ),
]

print(f"Loaded {len(documents)} documents for retrieval.")

vectorstore = Chroma.from_documents(
    documents, 
    embeddings,
    collection_name="hybrid_test")


# create vector retriever 

vector_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
    ) #returning top 3 results

print("Vector retriever ready.")

# BM25 works on raw text
bm25_retriever = BM25Retriever.from_documents(
    documents, 
    k=3)

# Combine EnsembleRetriever to use both vector and BM25 retrievers

enumerated_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]  # Adjust weights as needed
)

print("Hybrid Retriever ready.")

def test_query(query, name, retriever):
    '''Test a query and show results'''
    results = retriever.invoke(query)
    print(f'\n{name} - Query: "{query}"')
    for i, doc in enumerate(results[:3]):
        preview = doc.page_content[:80] + '...'
        print(f'  {i+1}. {preview}')
    return results


# Test queries designed to challenge vector search
test_queries = [
    'SKU-7742X specifications',      # Exact product code
    'E_CONN_REFUSED error',          # Error code
    'How do I authenticate?',        # Semantic question
    'WCAG compliance',               # Acronym
    'router configuration',          # General semantic
]

for query in test_queries:
    test_query(query, 'Vector', vector_retriever)
    test_query(query, 'BM25', bm25_retriever)
    test_query(query, 'Hybrid', enumerated_retriever)

