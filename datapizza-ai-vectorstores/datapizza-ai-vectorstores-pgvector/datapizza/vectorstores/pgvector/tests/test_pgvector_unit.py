"""
Unit Tests for PGVectorStore.

Mock-based tests for specific methods without database dependencies.
Fast and isolated testing of core functionality.

Run with: pytest test_pgvector_unit.py -v
"""
import pytest
from unittest.mock import MagicMock, patch
import sys

# Define mocks globally so we can configure them if needed, 
# but we'll apply them via fixture
mock_psycopg = MagicMock()
mock_sql = MagicMock()
mock_psycopg.sql = mock_sql
mock_retriever_module = MagicMock()

@pytest.fixture(autouse=True)
def patch_dependencies():
    """Patch sys.modules for all tests in this file."""
    with patch.dict(sys.modules, {
        "psycopg": mock_psycopg, 
        "psycopg.sql": mock_sql,
        "datapizza.retriever": mock_retriever_module
    }):
        # Ensure submodules are accessible
        sys.modules["psycopg.sql"] = mock_sql
        yield

# Import the class under test. 
# Note: If these imports fail because of missing dependencies on the system,
# we might need to wrap them in a try/except or move them inside tests/fixtures
# BUT, since we are patching sys.modules in the fixture, we need to make sure
# the import happens AFTER the patch if the module itself has top-level imports that fail.
# However, PGVectorStore has try/except blocks for imports, so it should be safe to import.
from datapizza.vectorstores.pgvector.pgvector_vectorstore import PGVectorStore
from datapizza.type import Chunk, DenseEmbedding

@pytest.fixture
def mock_pg_store():
    # We need to patch the module-level psycopg in pgvector as well
    # And 'sql' since it's imported directly
    with patch("datapizza.vectorstores.pgvector.pgvector_vectorstore.psycopg", mock_psycopg), \
         patch("datapizza.vectorstores.pgvector.pgvector_vectorstore.sql", mock_sql), \
         patch("datapizza.vectorstores.pgvector.pgvector_vectorstore.ConnectionPool") as MockPool, \
         patch("datapizza.vectorstores.pgvector.pgvector_vectorstore.register_vector"):
        
        store = PGVectorStore(connection_string="mock_conn_str")
        store.sync_pool = MockPool.return_value
        yield store

def test_get_collections(mock_pg_store):
    # Setup mock return
    mock_conn = mock_pg_store.sync_pool.connection.return_value.__enter__.return_value
    mock_conn.execute.return_value.fetchall.return_value = [("collection1",), ("collection2",)]
    
    collections = mock_pg_store.get_collections()
    
    assert collections == ["collection1", "collection2"]
    mock_conn.execute.assert_called_once()
    assert "SELECT table_name" in mock_conn.execute.call_args[0][0]

def test_dump_collection(mock_pg_store):
    # Setup mock return
    mock_conn = mock_pg_store.sync_pool.connection.return_value.__enter__.return_value
    # Mock row: id, text, metadata, embedding
    mock_rows = [
        ("id1", "text1", {"meta": "data1"}, [0.1, 0.2]),
        ("id2", "text2", {"meta": "data2"}, [0.3, 0.4])
    ]
    # First call returns rows, second call returns empty (end of loop)
    mock_conn.execute.return_value.fetchall.side_effect = [mock_rows, []]
    
    # Reset mocks to clear previous calls
    mock_sql.SQL.reset_mock()
    
    # It returns a generator now
    chunks_gen = mock_pg_store.dump_collection("test_coll", page_size=10, with_vectors=True)
    chunks = list(chunks_gen)
    
    assert len(chunks) == 2
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].id == "id1"
    assert chunks[0].text == "text1"
    assert chunks[0].metadata == {"meta": "data1"}
    
    # Verify SQL construction
    mock_sql.SQL.assert_called()
    # Check execute was called
    assert mock_conn.execute.call_count >= 1

def test_as_retriever(mock_pg_store):
    # Now uses base implementation which takes kwargs
    retriever = mock_pg_store.as_retriever(some_arg="value")
    
    # Should return a Retriever instance from datapizza.core.vectorstore
    # Since we didn't patch datapizza.core.vectorstore.Retriever, it might be the real one or fail if not found.
    # But Vectorstore imports Retriever from datapizza.core.vectorstore (which is in the same file usually or imported)
    
    # The base Vectorstore.as_retriever returns Retriever(self, **kwargs)
    # We can check if it has the vectorstore attached
    assert retriever.vectorstore == mock_pg_store
    assert retriever.kwargs == {"some_arg": "value"}
