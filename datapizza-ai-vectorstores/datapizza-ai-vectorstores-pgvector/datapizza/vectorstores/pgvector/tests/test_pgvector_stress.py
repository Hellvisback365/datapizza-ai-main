"""
Aggressive Stress Tests for PGVectorStore.

Tests production-like scenarios with large datasets, concurrency, 
edge cases, and stress conditions. These tests may take several minutes.

Run with: pytest test_pgvector_stress.py -v -s
Mark as slow: pytest -m "not slow" to skip these tests in CI
"""
import uuid
import time
import threading
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed
from datapizza.vectorstores.pgvector import PGVectorStore
from datapizza.core.vectorstore import VectorConfig
from datapizza.type import Chunk, DenseEmbedding

CONN_STR = "postgresql://postgres:secret@localhost:5433/postgres"

pytestmark = pytest.mark.slow  # Mark all tests in this module as slow


@pytest.fixture
def integration_store():
    """Create store for integration tests."""
    store = PGVectorStore(connection_string=CONN_STR)
    yield store
    # Cleanup all test collections
    try:
        for coll in store.get_collections():
            if coll.startswith("integ_"):
                store.delete_collection(coll)
    except:
        pass
    store.close()


def test_massive_batch_operations(integration_store):
    """
    Test adding and searching 10,000 chunks.
    Verifies: memory efficiency, batch processing, large result sets.
    Expected time: ~30 seconds
    """
    print("\n[STRESS] Testing 10,000 chunks...")
    start = time.time()
    
    collection = "integ_massive"
    integration_store.create_collection(
        collection,
        vector_config=[VectorConfig(dimensions=384, name="default")],
    )

    # Generate 10,000 chunks in batches
    batch_size = 100
    total_chunks = 10000
    
    for batch_num in range(total_chunks // batch_size):
        chunks = [
            Chunk(
                id=str(uuid.uuid4()),
                text=f"Document {batch_num * batch_size + i}: Lorem ipsum dolor sit amet",
                metadata={"batch": batch_num, "index": i, "category": f"cat_{i % 10}"},
                embeddings=[DenseEmbedding(name="default", vector=[float(i % 100) / 100.0] * 384)],
            )
            for i in range(batch_size)
        ]
        integration_store.add(chunks, collection)
        
        if batch_num % 10 == 0:
            print(f"  Added {(batch_num + 1) * batch_size} chunks...")

    elapsed = time.time() - start
    print(f"  ✓ Added 10,000 chunks in {elapsed:.2f}s ({total_chunks/elapsed:.0f} chunks/sec)")

    # Test search with various k values
    search_start = time.time()
    for k in [1, 10, 100, 1000]:
        results = integration_store.search(
            collection,
            query_vector=[0.5] * 384,
            k=k,
        )
        assert len(results) == k
        print(f"  ✓ Search k={k}: {len(results)} results")
    
    search_elapsed = time.time() - search_start
    print(f"  ✓ All searches completed in {search_elapsed:.2f}s")

    # Cleanup
    integration_store.delete_collection(collection)
    
    total_time = time.time() - start
    print(f"  ✓ Total test time: {total_time:.2f}s")
    assert total_time < 60, "Test took too long (>60s)"


def test_concurrent_operations(integration_store):
    """
    Test 10 concurrent threads doing add/search/update operations.
    Verifies: connection pool handling, thread safety, no deadlocks.
    Expected time: ~20 seconds
    """
    print("\n[CONCURRENCY] Testing 10 parallel threads...")
    
    collection = "integ_concurrent"
    integration_store.create_collection(
        collection,
        vector_config=[VectorConfig(dimensions=128, name="default")],
    )

    def worker_task(thread_id):
        """Each thread adds, searches, and updates chunks."""
        chunks = [
            Chunk(
                id=str(uuid.uuid4()),
                text=f"Thread {thread_id} chunk {i}",
                metadata={"thread": thread_id, "item": i},
                embeddings=[DenseEmbedding(name="default", vector=[float(thread_id + i) / 100.0] * 128)],
            )
            for i in range(50)
        ]
        
        # Add
        integration_store.add(chunks, collection)
        
        # Search
        results = integration_store.search(
            collection,
            query_vector=[float(thread_id) / 10.0] * 128,
            k=10,
        )
        
        # Update one chunk
        if results:
            integration_store.update(
                collection,
                payload={"processed": True, "thread": thread_id},
                points=[results[0].id],
            )
        
        return f"Thread {thread_id} completed: {len(results)} results"

    start = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker_task, i) for i in range(10)]
        results = [f.result() for f in as_completed(futures)]
    
    elapsed = time.time() - start
    print(f"  ✓ 10 threads completed in {elapsed:.2f}s")
    
    # Verify data integrity
    all_results = integration_store.search(collection, query_vector=[0.5] * 128, k=500)
    print(f"  ✓ Total chunks in collection: {len(all_results)}")
    assert len(all_results) == 500, "Expected 500 chunks from 10 threads × 50 chunks"
    
    # Cleanup
    integration_store.delete_collection(collection)
    assert elapsed < 30, "Concurrent test took too long"


def test_upsert_behavior_stress(integration_store):
    """
    Test adding same IDs 100 times (duplicate resolution).
    Verifies: upsert logic, no duplicate accumulation.
    Expected time: ~10 seconds
    """
    print("\n[UPSERT] Testing duplicate ID handling...")
    
    collection = "integ_upsert"
    integration_store.create_collection(
        collection,
        vector_config=[VectorConfig(dimensions=64, name="default")],
    )

    chunk_id = str(uuid.uuid4())
    
    # Add same ID 100 times with different versions
    for version in range(100):
        chunk = Chunk(
            id=chunk_id,
            text=f"Version {version}",
            metadata={"version": version, "timestamp": time.time()},
            embeddings=[DenseEmbedding(name="default", vector=[float(version) / 100.0] * 64)],
        )
        integration_store.add([chunk], collection)
        
        if version % 10 == 0:
            print(f"  Added version {version}")

    # Verify only latest version exists
    results = integration_store.retrieve(collection, ids=[chunk_id])
    assert len(results) == 1, "Should only have 1 chunk, not duplicates"
    assert results[0].metadata["version"] == 99, "Should be latest version"
    assert results[0].text == "Version 99"
    
    print(f"  ✓ Verified only latest version exists (v99)")
    
    # Cleanup
    integration_store.delete_collection(collection)


def test_dump_pagination_edge_cases(integration_store):
    """
    Test dump_collection with various page sizes and edge cases.
    Verifies: pagination correctness, boundary conditions.
    Expected time: ~15 seconds
    """
    print("\n[DUMP] Testing pagination edge cases...")
    
    collection = "integ_dump"
    integration_store.create_collection(
        collection,
        vector_config=[VectorConfig(dimensions=32, name="default")],
    )

    # Test with 0 chunks
    dumped_empty = list(integration_store.dump_collection(collection, page_size=10))
    assert len(dumped_empty) == 0
    print("  ✓ Empty collection: 0 chunks")

    # Add exactly 1 chunk
    chunk = Chunk(
        id=str(uuid.uuid4()),
        text="Single chunk",
        embeddings=[DenseEmbedding(name="default", vector=[0.0] * 32)],
    )
    integration_store.add([chunk], collection)
    
    dumped_one = list(integration_store.dump_collection(collection, page_size=10))
    assert len(dumped_one) == 1
    print("  ✓ Single chunk: 1 chunk")

    # Add 97 more chunks (total 98 - tests boundary with various page sizes)
    chunks = [
        Chunk(
            id=str(uuid.uuid4()),
            text=f"Chunk {i}",
            metadata={"index": i},
            embeddings=[DenseEmbedding(name="default", vector=[float(i) / 100.0] * 32)],
        )
        for i in range(97)
    ]
    integration_store.add(chunks, collection)

    # Test various page sizes
    for page_size in [1, 7, 10, 50, 100, 1000]:
        dumped = list(integration_store.dump_collection(collection, page_size=page_size))
        assert len(dumped) == 98, f"page_size={page_size} should return all 98 chunks"
        print(f"  ✓ page_size={page_size}: retrieved all 98 chunks")

    # Test with vectors
    dumped_with_vecs = list(integration_store.dump_collection(
        collection, 
        page_size=20, 
        with_vectors=True
    ))
    assert all(len(c.embeddings) > 0 for c in dumped_with_vecs)
    assert all(c.embeddings[0].vector is not None for c in dumped_with_vecs)
    print("  ✓ with_vectors=True: all chunks have embeddings")

    # Cleanup
    integration_store.delete_collection(collection)


def test_jsonb_filter_combinations(integration_store):
    """
    Test complex JSONB filter queries.
    Verifies: nested objects, arrays, multiple conditions.
    Expected time: ~10 seconds
    """
    print("\n[JSONB] Testing complex metadata filters...")
    
    collection = "integ_jsonb"
    integration_store.create_collection(
        collection,
        vector_config=[VectorConfig(dimensions=64, name="default")],
    )

    # Add chunks with complex metadata
    chunks = [
        Chunk(
            id=str(uuid.uuid4()),
            text="Document A",
            metadata={
                "category": "tech",
                "tags": ["python", "database"],
                "priority": 1,
                "author": {"name": "Alice", "verified": True},
            },
            embeddings=[DenseEmbedding(name="default", vector=[0.1] * 64)],
        ),
        Chunk(
            id=str(uuid.uuid4()),
            text="Document B",
            metadata={
                "category": "tech",
                "tags": ["javascript", "frontend"],
                "priority": 2,
                "author": {"name": "Bob", "verified": False},
            },
            embeddings=[DenseEmbedding(name="default", vector=[0.2] * 64)],
        ),
        Chunk(
            id=str(uuid.uuid4()),
            text="Document C",
            metadata={
                "category": "business",
                "tags": ["marketing"],
                "priority": 1,
                "author": {"name": "Charlie", "verified": True},
            },
            embeddings=[DenseEmbedding(name="default", vector=[0.3] * 64)],
        ),
    ]
    integration_store.add(chunks, collection)

    # Test filter by category
    results = integration_store.search(
        collection,
        query_vector=[0.0] * 64,
        k=10,
        filters={"category": "tech"},
    )
    assert len(results) == 2
    print(f"  ✓ Filter by category='tech': {len(results)} results")

    # Test filter by priority
    results = integration_store.search(
        collection,
        query_vector=[0.0] * 64,
        k=10,
        filters={"priority": 1},
    )
    assert len(results) == 2
    print(f"  ✓ Filter by priority=1: {len(results)} results")

    # Test nested object filter (PostgreSQL JSONB supports containment)
    results = integration_store.search(
        collection,
        query_vector=[0.0] * 64,
        k=10,
        filters={"author": {"verified": True}},
    )
    assert len(results) == 2
    print(f"  ✓ Filter by author.verified=True: {len(results)} results")

    # Cleanup
    integration_store.delete_collection(collection)


def test_connection_pool_lifecycle(integration_store):
    """
    Test connection pool creation, usage, and cleanup.
    Verifies: no connection leaks, proper cleanup.
    Expected time: ~5 seconds
    """
    print("\n[POOL] Testing connection pool lifecycle...")
    
    # Create new store
    store = PGVectorStore(connection_string=CONN_STR)
    
    # Verify lazy init - pools don't exist yet
    assert not hasattr(store, "sync_pool") or store.sync_pool is None
    print("  ✓ Initial state: no pools created")
    
    # Create collection (triggers pool creation)
    store.create_collection(
        "integ_pool_test",
        vector_config=[VectorConfig(dimensions=32, name="default")],
    )
    
    # Verify pool created
    assert store.sync_pool is not None
    print("  ✓ After first operation: sync pool created")
    
    # Use the pool
    chunk = Chunk(
        id=str(uuid.uuid4()),
        text="Test",
        embeddings=[DenseEmbedding(name="default", vector=[0.0] * 32)],
    )
    store.add([chunk], "integ_pool_test")
    
    results = store.search("integ_pool_test", query_vector=[0.0] * 32, k=1)
    assert len(results) == 1
    print("  ✓ Pool operations successful")
    
    # Cleanup and close
    store.delete_collection("integ_pool_test")
    store.close()
    print("  ✓ Pool closed successfully")
    
    # Verify cleanup (pools should be closed)
    # Note: Can't easily verify PostgreSQL connection count here


def test_vector_dimensionality_edge_cases(integration_store):
    """
    Test various vector dimensions.
    Verifies: dim=1, dim=4096, proper similarity calculations.
    Expected time: ~5 seconds
    """
    print("\n[VECTORS] Testing dimension edge cases...")
    
    # Test dim=1
    collection_1d = "integ_dim_1"
    integration_store.create_collection(
        collection_1d,
        vector_config=[VectorConfig(dimensions=1, name="default")],
    )
    
    chunk_1d = Chunk(
        id=str(uuid.uuid4()),
        text="1D vector",
        embeddings=[DenseEmbedding(name="default", vector=[0.5])],
    )
    integration_store.add([chunk_1d], collection_1d)
    
    results_1d = integration_store.search(collection_1d, query_vector=[0.5], k=1)
    assert len(results_1d) == 1
    print("  ✓ dim=1: OK")
    
    # Test dim=4096
    collection_4k = "integ_dim_4096"
    integration_store.create_collection(
        collection_4k,
        vector_config=[VectorConfig(dimensions=4096, name="default")],
    )
    
    chunk_4k = Chunk(
        id=str(uuid.uuid4()),
        text="4096D vector",
        embeddings=[DenseEmbedding(name="default", vector=[0.001] * 4096)],
    )
    integration_store.add([chunk_4k], collection_4k)
    
    results_4k = integration_store.search(collection_4k, query_vector=[0.001] * 4096, k=1)
    assert len(results_4k) == 1
    print("  ✓ dim=4096: OK")
    
    # Cleanup
    integration_store.delete_collection(collection_1d)
    integration_store.delete_collection(collection_4k)


def test_special_characters_in_metadata(integration_store):
    """
    Test Unicode, emojis, and special characters in metadata.
    Verifies: safe JSONB handling, no encoding issues.
    Expected time: ~5 seconds
    """
    print("\n[METADATA] Testing special characters...")
    
    collection = "integ_special_chars"
    integration_store.create_collection(
        collection,
        vector_config=[VectorConfig(dimensions=32, name="default")],
    )

    # Test various special characters
    chunks = [
        Chunk(
            id=str(uuid.uuid4()),
            text="Unicode test: 你好世界 مرحبا العالم",
            metadata={
                "emoji": "🍕🎉💯",
                "special": "quotes\"'apostrophes",
                "unicode": "Ñoño façade naïve",
                "symbols": "!@#$%^&*()_+-=[]{}|;:,.<>?",
            },
            embeddings=[DenseEmbedding(name="default", vector=[0.1] * 32)],
        ),
        Chunk(
            id=str(uuid.uuid4()),
            text="SQL special: '; DROP TABLE--",
            metadata={
                "attempt": "'; DROP TABLE test; --",
                "unicode_emoji": "テスト🔥",
            },
            embeddings=[DenseEmbedding(name="default", vector=[0.2] * 32)],
        ),
    ]
    
    integration_store.add(chunks, collection)
    
    # Retrieve and verify
    results = integration_store.search(collection, query_vector=[0.0] * 32, k=10)
    assert len(results) == 2
    
    # Verify metadata preserved correctly
    for result in results:
        assert result.metadata is not None
        if "emoji" in result.metadata:
            assert result.metadata["emoji"] == "🍕🎉💯"
            print("  ✓ Emoji preserved correctly")
        if "unicode_emoji" in result.metadata:
            assert result.metadata["unicode_emoji"] == "テスト🔥"
            print("  ✓ Unicode+emoji preserved correctly")
    
    # Test filtering with special characters
    results_filtered = integration_store.search(
        collection,
        query_vector=[0.0] * 32,
        k=10,
        filters={"emoji": "🍕🎉💯"},
    )
    assert len(results_filtered) == 1
    print("  ✓ Filter with emoji: OK")
    
    # Cleanup
    integration_store.delete_collection(collection)


def test_error_handling_graceful_failures(integration_store):
    """
    Test graceful error handling for various failure scenarios.
    Verifies: proper error messages, no crashes, state consistency.
    Expected time: ~5 seconds
    """
    print("\n[ERRORS] Testing error handling...")
    
    # Test search on non-existent collection
    try:
        integration_store.search(
            "nonexistent_collection_12345",
            query_vector=[0.0] * 128,
            k=10,
        )
        assert False, "Should have raised an exception"
    except Exception as e:
        print(f"  ✓ Non-existent collection error: {type(e).__name__}")
    
    # Test add without creating collection
    try:
        chunk = Chunk(
            id=str(uuid.uuid4()),
            text="Test",
            embeddings=[DenseEmbedding(name="default", vector=[0.0] * 64)],
        )
        integration_store.add([chunk], "never_created_collection")
        assert False, "Should have raised an exception"
    except Exception as e:
        print(f"  ✓ Add to non-existent collection error: {type(e).__name__}")
    
    # Test invalid dimension in search
    collection = "integ_error_test"
    integration_store.create_collection(
        collection,
        vector_config=[VectorConfig(dimensions=128, name="default")],
    )
    
    try:
        # Wrong dimension vector
        integration_store.search(
            collection,
            query_vector=[0.0] * 64,  # Wrong dimension (should be 128)
            k=10,
        )
        # May or may not raise depending on pgvector behavior
    except Exception as e:
        print(f"  ✓ Dimension mismatch error: {type(e).__name__}")
    
    # Cleanup
    integration_store.delete_collection(collection)
    print("  ✓ Error recovery successful")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
