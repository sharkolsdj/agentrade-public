# rag/

This directory holds the retrieval knowledge base used by StrategyRAGAgent
(vector store collections and the underlying strategy documents).

The knowledge base is intentionally not included in this public repository.
It can be reconstructed with a Qdrant instance and a corpus of strategy
documentation. The agent interface that consumes it is published under
`agents/strategy_rag_agent.py` as a documented stub.
