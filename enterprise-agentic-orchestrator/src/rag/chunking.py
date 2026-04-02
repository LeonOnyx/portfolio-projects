"""Document chunking module for the RAG pipeline.

Uses llama-index SentenceSplitter to break long documents into
~512-token chunks with 50-token overlap, preserving all metadata
from the source document dictionary.
"""

from __future__ import annotations

import copy

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


def chunk_document(
    doc_dict: dict,
    text_key: str = "content",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[dict]:
    """Split a document dictionary into smaller chunks.

    Parameters
    ----------
    doc_dict:
        Source document as a flat dictionary.  Must contain the key
        specified by *text_key* with the text content to chunk.
    text_key:
        Dictionary key holding the text content.  Defaults to
        ``"content"``.
    chunk_size:
        Target chunk size in tokens.  Defaults to 512.
    chunk_overlap:
        Overlap between consecutive chunks in tokens.  Defaults to 50.

    Returns
    -------
    list[dict]
        One dictionary per chunk.  Each is a shallow copy of *doc_dict*
        with the text replaced by the chunk text, plus ``chunk_index``
        (0-based) and ``total_chunks`` fields added.
    """
    text = doc_dict.get(text_key, "")
    if not text:
        chunk = copy.copy(doc_dict)
        chunk["chunk_index"] = 0
        chunk["total_chunks"] = 1
        return [chunk]

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    document = Document(text=text)
    nodes = splitter.get_nodes_from_documents([document])

    total = len(nodes)
    chunks: list[dict] = []
    for idx, node in enumerate(nodes):
        chunk = copy.copy(doc_dict)
        chunk[text_key] = node.get_content()
        chunk["chunk_index"] = idx
        chunk["total_chunks"] = total
        chunks.append(chunk)

    return chunks
