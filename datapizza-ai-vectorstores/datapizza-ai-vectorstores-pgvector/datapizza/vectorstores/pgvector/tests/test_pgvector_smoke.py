"""
Fast Smoke Tests for PGVectorStore CI/CD.

Quick sanity checks that must pass in <10 seconds total.
Run before integration tests to catch obvious failures early.

Usage: pytest test_pgvector_smoke.py -v
"""
import uuid
import pytest
from datapizza.vectorstores.pgvector import PGVectorStore
from datapizza.core.vectorstore import VectorConfig  
from datapizza.type import Chunk, DenseEmbedding

CONN_STR = "postgresql://postgres:secret@localhost:5433/postgres"


def test_can_connect():
    """Verify PostgreSQL database is accessible."""
    try:
        store = PGVectorStore(connection_string=CONN_STR)
        # Trigger connection
        _ = store.get_client()
        store.close()
        assert True
    except Exception as e:
        pytest.fail(f"Cannot connect to PostgreSQL: {e}")


def test_basic_crud():
    """Fast end-to-end CRUD test (create, add, search, retrieve, delete)."""
    store = PGVectorStore(connection_string=CONN_STR)
    
    collection = "smoke_crud"
    
    # Create
    store.create_collection(
        collection,
        vector_config=[VectorConfig(dimensions=64, name="test")],
    )
    
    # Add
    chunk_id = str(uuid.uuid4())
    chunk = Chunk(
        id=chunk_id,
        text="Smoke test chunk",
        metadata={"test": True},
        embeddings=[DenseEmbedding(name="test", vector=[0.5] * 64)],
    )
    store.add([chunk], collection)
    
    # Search
    search_results = store.search(collection, query_vector=[0.5] * 64, k=1)
    assert len(search_results) == 1
    assert search_results[0].text == "Smoke test chunk"
    
    # Retrieve
    retrieve_results = store.retrieve(collection, ids=[chunk_id])
    assert len(retrieve_results) == 1
    assert retrieve_results[0].id == chunk_id
    
    # Delete collection
    store.delete_collection(collection)
    
    # Verify deleted
    collections = store.get_collections()
    assert collection not in collections
    
    store.close()


def test_new_methods_exist():
    """Verify new conformance methods are callable."""
    store = PGVectorStore(connection_string=CONN_STR)
    
    # Verify methods exist
    assert hasattr(store, "dump_collection")
    assert callable(store.dump_collection)
    
    assert hasattr(store, "get_collections")
    assert callable(store.get_collections)
    
    assert hasattr(store, "get_client")
    assert callable(store.get_client)
    
    assert hasattr(store, "_get_a_client")
    assert callable(store._get_a_client)
    
    # Verify get_collections works
    collections = store.get_collections()
    assert isinstance(collections, list)
    
    store.close()


def test_no_import_errors():
    """Verify module imports cleanly without errors."""
    try:
        from datapizza.vectorstores.pgvector import PGVectorStore
        from datapizza.core.vectorstore import VectorConfig
        from datapizza.type import Chunk, DenseEmbedding
        
        # Verify classes are importable
        assert PGVectorStore is not None
        assert VectorConfig is not None
        assert Chunk is not None
        assert DenseEmbedding is not None
        
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
