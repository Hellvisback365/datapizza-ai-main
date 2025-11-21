"""
Functional Tests for PGVectorStore.

Complete end-to-end tests for all PGVectorStore functionality.
Tests CRUD operations, filters, async methods, and edge cases.

Run with: pytest test_pgvector_functional.py -v
"""
import uuid
import pytest
import asyncio
import selectors
from datapizza.vectorstores.pgvector import PGVectorStore
from datapizza.core.vectorstore import VectorConfig
from datapizza.type import Chunk, DenseEmbedding

# PostgreSQL connection string - adjust if needed
CONN_STR = "postgresql://postgres:secret@localhost:5433/postgres"


@pytest.fixture
def vectorstore() -> PGVectorStore:
    """Create a PGVectorStore instance with test collection."""
    store = PGVectorStore(connection_string=CONN_STR)
    store.create_collection(
        collection_name="test",
        vector_config=[VectorConfig(dimensions=1536, name="dense_emb_name")],
    )
    yield store
    # Cleanup
    try:
        store.delete_collection("test")
    except:
        pass
    store.close()


def test_pgvector_init():
    """Test PGVectorStore initialization."""
    vectorstore = PGVectorStore(connection_string=CONN_STR)
    assert vectorstore is not None
    assert vectorstore.connection_string == CONN_STR
    assert vectorstore.schema == "public"
    vectorstore.close()


def test_lazy_init_sync_pool():
    """Test that sync pool is created lazily on first access."""
    vectorstore = PGVectorStore(connection_string=CONN_STR)
    # Pool should not exist yet
    assert not hasattr(vectorstore, "sync_pool") or vectorstore.sync_pool is None
    # Access should create it
    client = vectorstore.get_client()
    assert client is not None
    assert vectorstore.sync_pool is not None
    vectorstore.close()


def test_pgvector_add(vectorstore):
    """Test adding chunks and searching."""
    chunks = [
        Chunk(
            id=str(uuid.uuid4()),
            text="Hello world",
            embeddings=[DenseEmbedding(name="dense_emb_name", vector=[0.0] * 1536)],
        )
    ]
    vectorstore.add(chunks, collection_name="test")

    res = vectorstore.search(collection_name="test", query_vector=[0.0] * 1536)
    assert len(res) == 1
    assert res[0].text == "Hello world"


def test_pgvector_create_collection(vectorstore):
    """Test creating a new collection."""
    vectorstore.create_collection(
        collection_name="test2",
        vector_config=[VectorConfig(dimensions=1536, name="test2")],
    )

    colls = vectorstore.get_collections()
    assert "test2" in colls
    assert "test" in colls

    # Cleanup
    vectorstore.delete_collection("test2")


def test_delete_collection(vectorstore):
    """Test deleting a collection."""
    vectorstore.create_collection(
        collection_name="deleteme",
        vector_config=[VectorConfig(dimensions=1536, name="test2")],
    )

    colls = vectorstore.get_collections()
    assert "deleteme" in colls
    
    vectorstore.delete_collection(collection_name="deleteme")

    colls = vectorstore.get_collections()
    assert "deleteme" not in colls


def test_remove(vectorstore):
    """Test removing chunks by ID."""
    chunk_id = str(uuid.uuid4())
    chunks = [
        Chunk(
            id=chunk_id,
            text="To be removed",
            embeddings=[DenseEmbedding(name="dense_emb_name", vector=[0.5] * 1536)],
        ),
        Chunk(
            id=str(uuid.uuid4()),
            text="To be kept",
            embeddings=[DenseEmbedding(name="dense_emb_name", vector=[0.1] * 1536)],
        ),
    ]
    vectorstore.add(chunks, collection_name="test")

    # Remove one chunk
    vectorstore.remove(collection_name="test", ids=[chunk_id])

    # Search should only return one result
    res = vectorstore.search(collection_name="test", query_vector=[0.0] * 1536, k=10)
    assert len(res) == 1
    assert res[0].text == "To be kept"


def test_retrieve(vectorstore):
    """Test retrieving chunks by ID."""
    chunk_id1 = str(uuid.uuid4())
    chunk_id2 = str(uuid.uuid4())
    chunks = [
        Chunk(
            id=chunk_id1,
            text="First chunk",
            metadata={"category": "A"},
            embeddings=[DenseEmbedding(name="dense_emb_name", vector=[0.1] * 1536)],
        ),
        Chunk(
            id=chunk_id2,
            text="Second chunk",
            metadata={"category": "B"},
            embeddings=[DenseEmbedding(name="dense_emb_name", vector=[0.2] * 1536)],
        ),
    ]
    vectorstore.add(chunks, collection_name="test")

    # Retrieve specific chunks
    results = vectorstore.retrieve(collection_name="test", ids=[chunk_id1, chunk_id2])
    assert len(results) == 2
    
    # Verify content
    texts = {r.text for r in results}
    assert "First chunk" in texts
    assert "Second chunk" in texts


def test_update(vectorstore):
    """Test updating chunk metadata."""
    chunk_id = str(uuid.uuid4())
    chunks = [
        Chunk(
            id=chunk_id,
            text="Test chunk",
            metadata={"status": "draft"},
            embeddings=[DenseEmbedding(name="dense_emb_name", vector=[0.0] * 1536)],
        )
    ]
    vectorstore.add(chunks, collection_name="test")

    # Update metadata
    new_metadata = {"status": "published", "reviewed": True}
    vectorstore.update(
        collection_name="test",
        payload=new_metadata,
        points=[chunk_id],  # Note: using string ID
    )

    # Retrieve and verify
    results = vectorstore.retrieve(collection_name="test", ids=[chunk_id])
    assert len(results) == 1
    assert results[0].metadata["status"] == "published"
    assert results[0].metadata["reviewed"] is True


def test_dump_collection(vectorstore):
    """Test dumping all chunks from a collection."""
    # Add multiple chunks
    chunks = [
        Chunk(
            id=str(uuid.uuid4()),
            text=f"Chunk {i}",
            metadata={"index": i},
            embeddings=[DenseEmbedding(name="dense_emb_name", vector=[float(i)] * 1536)],
        )
        for i in range(5)
    ]
    vectorstore.add(chunks, collection_name="test")

    # Dump collection
    dumped = list(vectorstore.dump_collection(collection_name="test", page_size=2))
    
    assert len(dumped) == 5
    # Verify all chunks are present
    texts = {c.text for c in dumped}
    for i in range(5):
        assert f"Chunk {i}" in texts


def test_dump_collection_with_vectors(vectorstore):
    """Test dumping collection with vectors included."""
    chunk = Chunk(
        id=str(uuid.uuid4()),
        text="Test chunk",
        embeddings=[DenseEmbedding(name="dense_emb_name", vector=[0.5] * 1536)],
    )
    vectorstore.add([chunk], collection_name="test")

    # Dump with vectors
    dumped = list(vectorstore.dump_collection(
        collection_name="test", 
        with_vectors=True
    ))
    
    assert len(dumped) == 1
    assert len(dumped[0].embeddings) > 0
    assert dumped[0].embeddings[0].vector is not None


def test_get_collections(vectorstore):
    """Test getting list of collections."""
    # Create additional collection
    vectorstore.create_collection(
        "another_test",
        vector_config=[VectorConfig(dimensions=128, name="test")],
    )

    collections = vectorstore.get_collections()
    
    assert "test" in collections
    assert "another_test" in collections
    
    # Cleanup
    vectorstore.delete_collection("another_test")


def test_get_collections_empty():
    """Test get_collections with empty schema."""
    store = PGVectorStore(connection_string=CONN_STR)
    
    # Should return empty list or only system tables
    collections = store.get_collections()
    # At minimum, should not crash
    assert isinstance(collections, list)
    
    store.close()


def test_search_with_filters(vectorstore):
    """Test JSONB metadata filtering (PRO feature)."""
    chunks = [
        Chunk(
            id=str(uuid.uuid4()),
            text="Pizza napoletana",
            metadata={"tipo": "napoletana", "country": "Italy"},
            embeddings=[DenseEmbedding(name="dense_emb_name", vector=[0.9, 0.1] + [0.0] * 1534)],
        ),
        Chunk(
            id=str(uuid.uuid4()),
            text="Hawaiian pizza",
            metadata={"tipo": "hawaii", "country": "USA"},
            embeddings=[DenseEmbedding(name="dense_emb_name", vector=[0.1, 0.9] + [0.0] * 1534)],
        ),
    ]
    vectorstore.add(chunks, collection_name="test")

    # Search with filter
    results = vectorstore.search(
        collection_name="test",
        query_vector=[0.9, 0.1] + [0.0] * 1534,
        k=10,
        filters={"tipo": "napoletana"},
    )
    
    assert len(results) == 1
    assert results[0].text == "Pizza napoletana"


def test_sql_injection_collection_name(vectorstore):
    """Test that collection names are SQL-injection safe."""
    # Attempt SQL injection via collection name
    malicious_name = "test; DROP TABLE test; --"
    
    # This should either safely escape or fail gracefully
    try:
        vectorstore.create_collection(
            malicious_name,
            vector_config=[VectorConfig(dimensions=128, name="test")],
        )
        # If it succeeds, verify original table still exists
        colls = vectorstore.get_collections()
        assert "test" in colls
        # Cleanup malicious table if created
        try:
            vectorstore.delete_collection(malicious_name)
        except:
            pass
    except Exception:
        # Expected to fail safely
        pass
    
    # Original collection should still exist
    colls = vectorstore.get_collections()
    assert "test" in colls


def test_upsert_behavior(vectorstore):
    """Test that adding with same ID replaces previous version."""
    chunk_id = str(uuid.uuid4())
    
    # Add first version
    chunk_v1 = Chunk(
        id=chunk_id,
        text="Version 1",
        metadata={"version": 1},
        embeddings=[DenseEmbedding(name="dense_emb_name", vector=[0.1] * 1536)],
    )
    vectorstore.add([chunk_v1], collection_name="test")

    # Add second version with same ID
    chunk_v2 = Chunk(
        id=chunk_id,
        text="Version 2",
        metadata={"version": 2},
        embeddings=[DenseEmbedding(name="dense_emb_name", vector=[0.2] * 1536)],
    )
    vectorstore.add([chunk_v2], collection_name="test")

    # Retrieve - should only get latest version
    results = vectorstore.retrieve(collection_name="test", ids=[chunk_id])
    assert len(results) == 1
    assert results[0].text == "Version 2"
    assert results[0].metadata["version"] == 2


def test_async_add_and_search():
    """Test async functionality with Windows-compatible event loop."""
    async def async_test():
        store = PGVectorStore(connection_string=CONN_STR)
        
        # Create collection
        config = [VectorConfig(name="dense_emb_name", dimensions=128)]
        store.create_collection("async_test", config)

        # Add data asynchronously
        chunks = [
            Chunk(
                id=str(uuid.uuid4()),
                text="Async chunk",
                metadata={"async": True},
                embeddings=[DenseEmbedding(name="dense_emb_name", vector=[0.5] * 128)],
            )
        ]
        await store.a_add(chunks, "async_test")

        # Search asynchronously
        results = await store.a_search("async_test", query_vector=[0.5] * 128, k=1)
        assert len(results) == 1
        assert results[0].text == "Async chunk"

        # Cleanup
        store.delete_collection("async_test")
        await store.aclose()

    # Use Windows-compatible event loop
    loop = asyncio.SelectorEventLoop(selectors.SelectSelector())
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(async_test())
    finally:
        asyncio.set_event_loop(None)


def test_context_manager():
    """Test using PGVectorStore as context manager."""
    with PGVectorStore(connection_string=CONN_STR) as store:
        store.create_collection(
            "ctx_test",
            vector_config=[VectorConfig(dimensions=64, name="test")],
        )
        
        # Add and search
        chunk = Chunk(
            id=str(uuid.uuid4()),
            text="Context manager test",
            embeddings=[DenseEmbedding(name="test", vector=[0.0] * 64)],
        )
        store.add([chunk], "ctx_test")
        
        results = store.search("ctx_test", query_vector=[0.0] * 64, k=1)
        assert len(results) == 1
        
        store.delete_collection("ctx_test")
    
    # Connection should be closed automatically