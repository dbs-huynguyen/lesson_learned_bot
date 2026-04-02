import textwrap
from typing import Any


# =============== SETUP ===============

# Giả định đã có:
# - vector_db: FAISS / Chroma / Pinecone / Weaviate / v.v.
# - bm25_index: keyword search (Elasticsearch / Whoosh / BM25 custom / v.v.)
# - embedding_model
# - llm (OpenAI / Local LLM / v.v.)
# - reranker_model (cross-encoder)


TOP_K_RETRIEVE = 20
TOP_K_FINAL = 5

# =============== 1. QUERY REWRITE ===============


def rewrite_query(user_query: str) -> str:
    prompt = textwrap.dedent(
        f"""
            Rewrite the query to be more clear and suitable for semantic search.
            Do NOT change the original meaning under any circumstances.
            
            Query:
            {user_query}
            
            Rewritten Query:
        """
    ).strip()
    rewritten = llm.invoke(prompt)
    return rewritten.strip()


# =============== 2. VECTOR SEARCH ===============


def vector_search(query: str, top_k=20) -> list[dict[str, Any]]:
    query_vec = embedding_model.embed_query(query)
    retrieved_docs = vector_db.search(vector=query_vec, top_k=top_k)
    # List of {id, text, score}
    return retrieved_docs


# =============== 3. KEYWORD SEARCH (BM25) ===============


def keyword_search(query: str, top_k=20) -> list[dict[str, Any]]:
    retrieved_docs = bm25_index.search(query, top_k=top_k)
    return retrieved_docs  # List of {id, text, score}


# =============== 4. HYBRID SEARCH ===============


def hybrid_search(query: str, top_k=20) -> list[list[dict[str, Any]]]:
    vec_results = vector_search(query, top_k)
    kw_results = keyword_search(query, top_k)

    vec_results.sort(key=lambda x: x["score"], reverse=True)
    kw_results.sort(key=lambda x: x["score"], reverse=True)

    fused_scores = {}

    def add_rrf(results: list[dict[str, Any]], weight=1.0) -> None:
        for rank, doc in enumerate(results):
            doc_id = doc["id"]

            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"doc": doc, "score": 0.0}

            # RRF formula
            fused_scores[doc_id]["score"] += weight * (1 / (60 + rank + 1))

    # add both sources
    add_rrf(vec_results, weight=1.0)
    add_rrf(kw_results, weight=1.0)

    # sort final
    merged = list(fused_scores.values())
    merged.sort(key=lambda x: x["score"], reverse=True)

    return [x["doc"] for x in merged[:top_k]]  # List of {id, text, score}


# =============== 5. RERANKING ===============


def rerank(query: str, docs: list[dict[str, Any]], top_k=5) -> list[dict[str, Any]]:
    pairs = [(query, doc["text"]) for doc in docs]
    scores = reranker_model.invoke(pairs)

    for i, d in enumerate(docs):
        d["rerank_score"] = scores[i]

    docs.sort(key=lambda x: x["rerank_score"], reverse=True)
    return docs[:top_k]


# =============== 6. GENERATE ANSWER ===============


def generate_answer(query: str, docs: list[dict[str, Any]]) -> str:
    context = "\n\n".join([d["text"] for d in docs])

    prompt = textwrap.dedent(
        f"""
            Answer the question based on the context below.
            
            Context:
            {context}

            Question:
            {query}
        """
    ).strip()
    answer = llm.invoke(prompt)
    return answer.strip()


# =============== MAIN PIPELINE ===============


def rag_pipeline(user_query: str) -> dict[str, Any]:
    # 1. rewrite
    rewritten_query = rewrite_query(user_query)

    # 2. vector search
    vec_results = vector_search(
        rewritten_query,
        top_k=TOP_K_RETRIEVE,
    )

    # 3. keyword search
    kw_results = keyword_search(
        rewritten_query,
        top_k=TOP_K_RETRIEVE,
    )

    # 4. rerank
    vec_top_docs = rerank(rewritten_query, vec_results, top_k=TOP_K_FINAL)
    kw_top_docs = rerank(rewritten_query, kw_results, top_k=TOP_K_FINAL)
    results_list = [vec_top_docs, kw_top_docs]

    # 5. RRF merge
    merged_docs = rrf_merge(results_list, k=TOP_K_FINAL)

    # 6. generate answer
    answer = generate_answer(rewritten_query, top_docs)
    return {
        "query": user_query,
        "rewritten_query": rewritten_query,
        "docs": top_docs,
        "answer": answer,
    }
