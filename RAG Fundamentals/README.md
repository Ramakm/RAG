# RAG Fundamentals

A guided path through 39 RAG techniques, one notebook per folder, ordered from the simple baseline to advanced architectures. Everything runs locally: `sentence-transformers` for embeddings and [Ollama](https://ollama.com) for the LLM.

## Setup

```bash
pip install -r requirements.txt
ollama pull llama3.2        # generation + judging
ollama pull llava           # only for notebook 22 (image captioning)
ollama serve
jupyter lab
```

Open any notebook from inside its own folder. Shared code lives in [`common.py`](common.py) (chunking, embeddings, vector store, LLM helpers) and sample data in [`data/`](data/). Override models with the `RAG_LLM`, `RAG_EMBED` and `RAG_VISION` environment variables.

**Must-do core:** 1-7 and 17-18 (marked with a star below). After those, pick from the rest based on what you are building.


## Basics (start here)

| # | Notebook | What it covers |
|---|---|---|
| 1 ⭐ | [simple_rag.ipynb](01_simple_rag/simple_rag.ipynb) | The minimal RAG loop: chunk, embed, retrieve, generate. Start here. |
| 2 ⭐ | [simple_csv_rag.ipynb](02_simple_csv_rag/simple_csv_rag.ipynb) | RAG over tabular data: turn every row into a self-describing sentence and index it. |
| 3 ⭐ | [json_rag.ipynb](03_json_rag/json_rag.ipynb) | Flatten nested JSON into path-annotated facts so structure is not lost. |
| 4 ⭐ | [reliable_rag.ipynb](04_reliable_rag/reliable_rag.ipynb) | Add guard rails: grade retrieved chunks, generate, then verify the answer is grounded. |
| 5 ⭐ | [choose_chunk_size.ipynb](05_choose_chunk_size/choose_chunk_size.ipynb) | Measure retrieval hit-rate for several chunk sizes instead of guessing. |
| 6 ⭐ | [proposition_chunking.ipynb](06_proposition_chunking/proposition_chunking.ipynb) | Rewrite passages into atomic, self-contained facts (propositions) and index those. |

## Evaluation - learn it early

| # | Notebook | What it covers |
|---|---|---|
| 7 ⭐ | [evaluation_deep_eval.ipynb](07_evaluation_deep_eval/evaluation_deep_eval.ipynb) | Score a RAG pipeline with DeepEval's LLM-as-judge metrics, using a local Ollama model as the judge. |

## Query side

| # | Notebook | What it covers |
|---|---|---|
| 8 | [query_transformations.ipynb](08_query_transformations/query_transformations.ipynb) | Rewrite, step-back and decompose the question before retrieval. |
| 9 | [HyDe_Hypothetical_Document_Embedding.ipynb](09_HyDe_Hypothetical_Document_Embedding/HyDe_Hypothetical_Document_Embedding.ipynb) | Search with the embedding of a *hypothetical answer* instead of the question. |
| 10 | [HyPE_Hypothetical_Prompt_Embeddings.ipynb](10_HyPE_Hypothetical_Prompt_Embeddings/HyPE_Hypothetical_Prompt_Embeddings.ipynb) | Move the work to index time: embed questions each chunk could answer, not the chunk itself. |

## Chunk / context side

| # | Notebook | What it covers |
|---|---|---|
| 11 | [contextual_chunk_headers.ipynb](11_contextual_chunk_headers/contextual_chunk_headers.ipynb) | Prepend a descriptive header (document + section) to every chunk before embedding. |
| 12 | [context_enrichment_window_around_chunk.ipynb](12_context_enrichment_window_around_chunk/context_enrichment_window_around_chunk.ipynb) | Retrieve the best chunk, then add its neighbours so the LLM sees surrounding context. |
| 13 | [semantic_chunking.ipynb](13_semantic_chunking/semantic_chunking.ipynb) | Split where the *meaning* changes: a breakpoint appears when neighbouring sentences are dissimilar. |
| 14 | [relevant_segment_extraction.ipynb](14_relevant_segment_extraction/relevant_segment_extraction.ipynb) | Return the best *contiguous run* of chunks instead of isolated top-k chunks. |
| 15 | [contextual_compression.ipynb](15_contextual_compression/contextual_compression.ipynb) | Shrink each retrieved chunk to only the sentences that matter for the question. |
| 16 | [document_augmentation.ipynb](16_document_augmentation/document_augmentation.ipynb) | Enrich the index with generated questions that point back to the original chunk. |

## Retrieval (most useful in practice)

| # | Notebook | What it covers |
|---|---|---|
| 17 ⭐ | [fusion_retrieval.ipynb](17_fusion_retrieval/fusion_retrieval.ipynb) | Combine keyword and semantic search - one of the most useful upgrades in practice. |
| 18 ⭐ | [reranking.ipynb](18_reranking/reranking.ipynb) | Retrieve broadly, then let a cross-encoder pick the best few. |
| 19 | [multi_faceted_filtering.ipynb](19_multi_faceted_filtering/multi_faceted_filtering.ipynb) | Layer metadata filters, a score threshold and diversity (MMR) on top of vector search. |
| 20 | [hierarchical_indices.ipynb](20_hierarchical_indices/hierarchical_indices.ipynb) | Two-tier retrieval: find the right section via summaries, then search chunks only inside it. |
| 21 | [dartboard.ipynb](21_dartboard/dartboard.ipynb) | Pick results that jointly cover the query and each other - relevance *and* diversity in one score. |
| 22 | [multi_model_rag_with_captioning.ipynb](22_multi_model_rag_with_captioning/multi_model_rag_with_captioning.ipynb) | Caption images with a vision LLM, then index captions next to the text so one text index serves both. |
| 23 | [multi_model_rag_with_colpali.ipynb](23_multi_model_rag_with_colpali/multi_model_rag_with_colpali.ipynb) | Skip captioning: embed page images directly with a late-interaction vision retriever. |

## Iterative

| # | Notebook | What it covers |
|---|---|---|
| 24 | [retrieval_with_feedback_loop.ipynb](24_retrieval_with_feedback_loop/retrieval_with_feedback_loop.ipynb) | Learn from user ratings: boost chunks that helped on similar past questions, demote those that did not. |
| 25 | [adaptive_retrieval.ipynb](25_adaptive_retrieval/adaptive_retrieval.ipynb) | Classify the question first, then choose a retrieval strategy that fits it. |
| 26 | [explainable_retrieval.ipynb](26_explainable_retrieval/explainable_retrieval.ipynb) | Show *why* each passage was retrieved and cite sources in the answer. |

## Advanced architectures

| # | Notebook | What it covers |
|---|---|---|
| 27 | [crag.ipynb](27_crag/crag.ipynb) | Grade retrieval quality; refine good evidence, and fall back to web search when it is poor. |
| 28 | [self_rag.ipynb](28_self_rag/self_rag.ipynb) | Let the model reflect: is retrieval needed, is each passage relevant, is the answer supported, is it useful? |
| 29 | [graph_rag.ipynb](29_graph_rag/graph_rag.ipynb) | Extract a knowledge graph from the documents and answer questions by walking it. |
| 30 | [graphrag_with_milvus_vectordb.ipynb](30_graphrag_with_milvus_vectordb/graphrag_with_milvus_vectordb.ipynb) | Store relations as vectors in Milvus, retrieve the closest, then expand through the graph. |
| 31 | [Microsoft_GraphRag.ipynb](31_Microsoft_GraphRag/Microsoft_GraphRag.ipynb) | Detect communities in the entity graph, summarise each, and answer global questions with map-reduce. |
| 32 | [light_rag.ipynb](32_light_rag/light_rag.ipynb) | Dual-level retrieval: match specific entities (low level) and broad themes (high level), both through the graph. |
| 33 | [graph_rag_local_attribution.ipynb](33_graph_rag_local_attribution/graph_rag_local_attribution.ipynb) | Keep the source chunk on every edge so each fact in the answer can be traced to the original text. |
| 34 | [raptor.ipynb](34_raptor/raptor.ipynb) | Recursively cluster and summarise chunks into a tree; retrieve across all levels. |
| 35 | [memorag.ipynb](35_memorag/memorag.ipynb) | Memory-inspired RAG: compress the whole corpus into a global memory, then use it to draft retrieval clues. |
| 36 | [Agentic_RAG.ipynb](36_Agentic_RAG/Agentic_RAG.ipynb) | An agent loop that decides when to search, what to search for and when it has enough evidence. |

## Deeper evaluation

| # | Notebook | What it covers |
|---|---|---|
| 37 | [evaluation_grouse.ipynb](37_evaluation_grouse/evaluation_grouse.ipynb) | Grade grounded answers on completeness, usefulness and faithfulness, and check that the *judge* behaves. |
| 38 | [end-2-end_rag_evaluation.ipynb](38_end-2-end_rag_evaluation/end-2-end_rag_evaluation.ipynb) | Measure retrieval (hit rate, MRR) and generation (correctness, faithfulness) over a golden set. |
| 39 | [open-rag-eval-example.ipynb](39_open-rag-eval-example/open-rag-eval-example.ipynb) | Evaluate without golden answers: grade retrieved passages (UMBRELA-style) and check answer nuggets. |

> Notebook 36 (Agentic RAG) runs a local agent loop; swapping in Contextual AI's hosted agents needs a Contextual AI account.
