import pytest
import asyncio
import selectors
from datapizza.vectorstores.pgvector_store import PGVectorStore
from datapizza.core.vectorstore import VectorConfig
from datapizza.type import Chunk, DenseEmbedding

CONN_STR = "postgresql://postgres:secret@localhost:5432/postgres"
COLLECTION_NAME = "test_collection"

@pytest.fixture
def pgvector_store():
    store = PGVectorStore(connection_string=CONN_STR)
    yield store
    # Cleanup
    try:
        store.delete_collection(COLLECTION_NAME)
        store.delete_collection(COLLECTION_NAME + "_async")
    except:
        pass
    store.close()

def test_sync_add_and_search_with_filters(pgvector_store):
    store = pgvector_store
    # Create collection
    config = [VectorConfig(name="default", dimensions=3)]
    store.create_collection(COLLECTION_NAME, config)

    # Add data
    chunks = [
        Chunk(
            id="1",
            text="La pizza napoletana è morbida.",
            metadata={"tipo": "napoletana"},
            embeddings=[DenseEmbedding(name="default", vector=[0.9, 0.1, 0.1])]
        ),
        Chunk(
            id="2",
            text="L'ananas sulla pizza è controverso.",
            metadata={"tipo": "hawaii"},
            embeddings=[DenseEmbedding(name="default", vector=[0.1, 0.9, 0.1])]
        ),
        Chunk(
            id="3",
            text="Il database Postgres è solido.",
            metadata={"tipo": "tech"},
            embeddings=[DenseEmbedding(name="default", vector=[0.1, 0.1, 0.9])]
        )
    ]
    store.add(chunks, COLLECTION_NAME)

    # Search
    results = store.search(COLLECTION_NAME, query_vector=[0.9, 0.15, 0.05], k=1)
    assert len(results) == 1
    assert results[0].id == "1"

@pytest.mark.skip(reason="Async tests have event loop issues on Windows")
async def test_async_add_and_search_old(pgvector_store):
    store = pgvector_store
    # Create collection (sync)
    config = [VectorConfig(name="default", dimensions=3)]
    store.create_collection(COLLECTION_NAME, config)

    # Add data (async)
    chunks = [
        Chunk(
            id="1",
            text="La pizza napoletana è morbida.",
            metadata={"tipo": "napoletana"},
            embeddings=[DenseEmbedding(name="default", vector=[0.9, 0.1, 0.1])]
        ),
        Chunk(
            id="2",
            text="L'ananas sulla pizza è controverso.",
            metadata={"tipo": "hawaii"},
            embeddings=[DenseEmbedding(name="default", vector=[0.1, 0.9, 0.1])]
        )
    ]
    await store.a_add(chunks, COLLECTION_NAME)

    # Search (async)
    results = await store.a_search(COLLECTION_NAME, query_vector=[0.9, 0.15, 0.05], k=1)
    assert len(results) == 1
    assert results[0].id == "1"

def test_async_add_and_search(pgvector_store):
    """Test async functionality using compatible event loop on Windows."""
    async def async_test():
        store = pgvector_store
        # Create collection (sync)
        config = [VectorConfig(name="default", dimensions=3)]
        store.create_collection(COLLECTION_NAME + "_async", config)

        # Add data (async)
        chunks = [
            Chunk(
                id="1",
                text="La pizza napoletana è morbida.",
                metadata={"tipo": "napoletana"},
                embeddings=[DenseEmbedding(name="default", vector=[0.9, 0.1, 0.1])]
            ),
            Chunk(
                id="2",
                text="L'ananas sulla pizza è controverso.",
                metadata={"tipo": "hawaii"},
                embeddings=[DenseEmbedding(name="default", vector=[0.1, 0.9, 0.1])]
            )
        ]
        await store.a_add(chunks, COLLECTION_NAME + "_async")

        # Search (async)
        results = await store.a_search(COLLECTION_NAME + "_async", query_vector=[0.9, 0.15, 0.05], k=1)
        assert len(results) == 1
        assert results[0].id == "1"

    # Usa un event loop compatibile con psycopg su Windows
    loop = asyncio.SelectorEventLoop(selectors.SelectSelector())
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(async_test())
    finally:
        asyncio.set_event_loop(None)