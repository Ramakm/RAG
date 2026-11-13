from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import shutil
from dotenv import load_dotenv

load_dotenv()

# Groq has no embedding models, so use local Ollama (run `ollama serve` and `ollama pull nomic-embed-text`)
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

# Sample documents
SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
]


def chroma_basics():
    with tempfile.TemporaryDirectory() as tmpdir:
        # create vector store from documents
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding=embeddings_model, persist_directory=tmpdir
        )
        print(
            f"Vector store created {vectorstore._collection.count()} documents and persisted."
        )

        # perform similarity search
        query = "What is LangChain?" # If langchain is a frame work, is it an opensource?
        results = vectorstore.similarity_search(query, k=3)

        print(f"Top 3 results for query '{query}':")
        for i, doc in enumerate(results):
            print(
                f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})"
            )


def similarity_search_with_scores():
    with tempfile.TemporaryDirectory() as tmpdir:
        # create vector store from documents
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding=embeddings_model, persist_directory=tmpdir
        )

        # perform similarity search with scores
        query = "Explain vector stores."
        results_with_scores = vectorstore.similarity_search_with_score(query, k=3)

        print(f"Top 3 results with scores for query '{query}':")
        for i, (doc, score) in enumerate(results_with_scores):
            final_score = 1 / (1 + score)  # Convert distance to similarity
            print(
                f"Result {i+1}: {doc.page_content} (Score: {final_score:.4f}, Source: {doc.metadata['source']})"
            )


def metadata_filtering():
    with tempfile.TemporaryDirectory() as tmpdir:
        # create vector store from documents
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding=embeddings_model, persist_directory=tmpdir
        )

        query = "What databases are available?"

        # without metadata filtering
        results = vectorstore.similarity_search(query, k=5)
        print(f"Results without metadata filtering for query '{query}':")
        for i, doc in enumerate(results):
            print(
                f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})"
            )

        # with metadata filtering
        filter_criteria = {"topic": "database"}
        filtered_results = vectorstore.similarity_search(
            query, k=5, filter=filter_criteria
        )
        print(f"\nResults with metadata filtering for query '{query}':")
        for i, doc in enumerate(filtered_results):
            print(
                f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})"
            )


def as_retriever():

    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=embeddings_model,
            persist_directory=tmpdir,
        )

        # basic retriever usage
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )
        # use retriever to get relevant documents
        docs = retriever.invoke("How do I build AI applications?")

        print("Retriever results:")
        for i, doc in enumerate(docs):
            print(
                f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})"
            )

        mmr_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 5},  # fetch 5 docs and return 3 diverse
        )
        mmr_docs = mmr_retriever.invoke("vector databases and embeddings")
        print("\nMMR Retriever results:")
        for i, doc in enumerate(mmr_docs):
            print(
                f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})"
            )


def persist_chroma():
    persist_dir = "./chroma_db/"

    vectorstore = Chroma.from_documents(
        documents=SAMPLE_DOCS,
        embedding=embeddings_model,
        persist_directory=persist_dir,
    )

    original_count = vectorstore._collection.count()
    print(f"Persisted vector store with {original_count} documents.")
    print(f"Vector store persisted at: {persist_dir}")

    # simulate restart - load from disk
    del vectorstore

    reloaded = Chroma(
        embedding_function=embeddings_model,
        persist_directory=persist_dir,
    )

    reloaded_count = reloaded._collection.count()
    print(f"Reloaded vector store with {reloaded_count} documents.")

    # verify search still works
    results = reloaded.similarity_search("LangChain", k=2)
    print(f"Search result: {results[0].page_content[:50]}...")


# Exercise: Set up Chroma + retriever
def exercise_vector_store_setup():
    """
    EXERCISE: Create a complete vector store setup that:
    1. Takes a list of text strings
    2. Splits them into chunks
    3. Stores in Chroma
    4. Returns a configured retriever

    Test with sample documents.
    """

    def create_retriever(
        texts: list[str], chunk_size: int = 500, chunk_overlap: int = 50, k: int = 3
    ):

        # Create documents
        docs = [Document(page_content=t) for t in texts]

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        split_docs = splitter.split_documents(docs)

        # Create vector store (in-memory for exercise)
        vectorstore = Chroma.from_documents(
            documents=split_docs, embedding=embeddings_model
        )

        # Return retriever
        return vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )

    # Test the function
    # Test
    sample_texts = [
        "Python is a versatile programming language used in web development, "
        "data science, machine learning, and automation. It has a simple syntax "
        "that makes it easy to learn and read.",
        "JavaScript is the language of the web. It runs in browsers and on "
        "servers with Node.js. Modern frameworks like React and Vue make "
        "building web applications efficient.",
        "Rust is a systems programming language focused on safety and "
        "performance. It prevents common bugs like null pointer dereferences "
        "and data races at compile time.",
    ]

    retriever = create_retriever(sample_texts, chunk_size=200, chunk_overlap=20, k=2)

    print("Testing retriever:\n")
    queries = [
        "What's good for web development?",
        "Which language is safest?",
    ]
    for query in queries:
        print(f"Query: {query}")
        results = retriever.invoke(query)
        for doc in results:
            print(f"  - {doc.page_content[:60]}...")
        print()


if __name__ == "__main__":
    chroma_basics()
    similarity_search_with_scores()
    # metadata_filtering()
    # as_retriever()
    # persist_chroma()
    # exercise_vector_store_setup()