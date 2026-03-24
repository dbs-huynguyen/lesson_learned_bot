import argparse
import json
from pathlib import Path

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.embeddings import init_embeddings

from docx_extractor import DocxExtractor


extractor = DocxExtractor()
embeddings = init_embeddings(
    "huggingface:AITeamVN/Vietnamese_Embedding", show_progress=True
)

embedding_dim = len(embeddings.embed_query("hello world"))
vector_store = FAISS(
    embedding_function=embeddings,
    index=faiss.IndexFlatIP(embedding_dim),
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
    normalize_L2=True,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a RAG index from lesson learned documents."
    )
    parser.add_argument(
        "--folder_path",
        type=Path,
        required=True,
        help="Path to the folder containing lesson learned .docx files.",
    )
    args = parser.parse_args()

    file_paths: list[Path] = []
    if args.folder_path.is_dir():
        for file_path in sorted(args.folder_path.glob("*.docx")):
            file_paths.append(file_path)
    else:
        file_paths.append(args.folder_path)

    for file_path in file_paths:
        vector_store.add_documents(extractor.run(file_path))
        extractor.clear()

    vector_store.save_local("lesson_learned_index")
