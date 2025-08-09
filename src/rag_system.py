from typing import List, Optional
from .client import query_openai
from .client import get_openai_embedding

def generate_rag_response(user_question: str, knowledge_base: List[str], 
                         max_context_length: int = 2000, 
                         model: str = "gpt-3.5-turbo") -> Optional[str]:
    """
    Generate a RAG-based response using retrieved documents from a knowledge base.
    
    Args:
        user_question: The user's input question
        knowledge_base: List of document chunks/passages to use as context
        max_context_length: Maximum character length for context to avoid token limits
        model: OpenAI model to use for generation
    
    Returns:
        Generated response based on question and retrieved context
    """
    if not knowledge_base:
        print("No documents provided for RAG context.")
        return query_openai(user_question, model)
    
    # Build context while respecting length limits
    context_parts = []
    current_length = 0
    
    for i, doc in enumerate(knowledge_base):
        doc_text = f"[Source {i+1}]: {doc}"
        if current_length + len(doc_text) > max_context_length:
            break
        context_parts.append(doc_text)
        current_length += len(doc_text)
    
    context = "\n\n".join(context_parts)
    
    # Create enhanced RAG prompt with clear structure
    prompt = f"""You are an AI assistant. Answer the question based on the provided context sources.\ 
                If the information needed to answer the question is not available in the context, clearly state that you don't have enough information.

                Context Sources:
                {context}

                Question: {user_question}

                Please provide a comprehensive answer based on the context above:"""
                
    return query_openai(prompt, model)


def retrieve_documents(query: str, corpus: List[str], k: int = 5, method: str = "bm25") -> List[str]:
    """
    Retrieve top-K most relevant documents for a given query.
    
    Args:
        query: The search query
        corpus: List of all available documents
        k: Number of top documents to retrieve
        method: Retrieval method ("bm25" or "semantic")
    
    Returns:
        List of top-K retrieved documents
    """
    if method == "bm25":
        return _retrieve_with_bm25(query, corpus, k)
    elif method == "semantic":
        return _retrieve_with_semantic_similarity(query, corpus, k)
    else:
        raise ValueError(f"Unsupported retrieval method: {method}")


def _retrieve_with_bm25(query: str, corpus: List[str], k: int) -> List[str]:
    """Simple BM25-like retrieval based on term frequency."""
    query_terms = set(query.lower().split())
    
    def score_document(doc: str) -> float:
        doc_terms = doc.lower().split()
        # Simple scoring: count of query terms in document
        return sum(1 for term in doc_terms if term in query_terms)
    
    # Score and rank documents
    doc_scores = [(doc, score_document(doc)) for doc in corpus]
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, _ in doc_scores[:k]]


def _retrieve_with_semantic_similarity(query: str, corpus: List[str], k: int) -> List[str]:
    """Retrieve documents using OpenAI embeddings for semantic similarity."""
    try:
        # Get embeddings for query and all documents
        query_embedding = get_openai_embedding(query)
        doc_embeddings = [get_openai_embedding(doc) for doc in corpus]
        
        # Calculate cosine similarity
        similarities = []
        for i, doc_emb in enumerate(doc_embeddings):
            similarity = _cosine_similarity(query_embedding, doc_emb)
            similarities.append((corpus[i], similarity))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in similarities[:k]]
    
    except Exception as e:
        print(f"Error in semantic retrieval: {e}")
        # Fallback to BM25
        return _retrieve_with_bm25(query, corpus, k)


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def embed_documents(documents: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of documents using OpenAI.
    
    Args:
        documents: List of document texts to embed
    
    Returns:
        List of embedding vectors for comparison metrics
    """
    embeddings = []
    for doc in documents:
        try:
            embedding = get_openai_embedding(doc)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                embeddings.append([0.0] * 1536)  # Fallback empty embedding
        except Exception as e:
            print(f"Error embedding document: {e}")
            embeddings.append([0.0] * 1536)  # Fallback empty embedding
    
    return embeddings
