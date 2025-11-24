# Datapizza AI - PostgreSQL (pgvector) Vector Store

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A production-ready PostgreSQL vector store integration for the **Datapizza AI** framework. Use PostgreSQL as your vector database for RAG applications without managing separate infrastructure like Qdrant or Milvus.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
  - [Initialization](#initialization)
  - [Creating Collections](#creating-collections)
  - [Adding Documents](#adding-documents)
  - [Searching with Filters](#searching-with-filters)
  - [Async Operations](#async-operations)
  - [Managing Collections](#managing-collections)
- [Advanced Features](#advanced-features)
  - [Connection Pooling](#connection-pooling)
  - [Metadata Filtering](#metadata-filtering)
  - [Resource Management](#resource-management)
- [Testing](#testing)
- [API Reference](#api-reference)
- [License](#license)

---

## Features

- 🐘 **Native PostgreSQL Support** – Uses `psycopg` (v3) drivers with the `pgvector` extension for efficient vector operations
- ⚡ **High-Performance Connection Pooling** – Implements `psycopg_pool` for both sync and async connection pooling to handle high-traffic scenarios
- 🔄 **True Async Support** – Native `async`/`await` implementation that won't block your event loop
- 🔍 **Advanced Metadata Filtering** – Leverage PostgreSQL JSONB columns with `@>` operators for powerful metadata queries
- 🛡️ **Production-Ready Robustness** – Automatic schema creation, extension management, upsert logic for duplicate handling, and graceful resource cleanup
- 🔌 **Seamless Integration** – Fully compatible with Datapizza AI framework components like `DagPipeline`
- 📦 **Export & Backup** – Built-in collection dumping via generators for memory-efficient data export

---

## Installation

```bash
pip install datapizza-ai-vectorstores-pgvector
```

### Prerequisites

You need a PostgreSQL database (version 11+) with the `pgvector` extension installed:

```sql
CREATE EXTENSION vector;
```

For local development, you can run PostgreSQL with Docker:

```bash
docker run -d \
  --name postgres-vector \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

---

## Quick Start

```python
from datapizza.vectorstores.pgvector import PGVectorStore
from datapizza.core.vectorstore import VectorConfig
from datapizza.type import Chunk, DenseEmbedding

# Initialize the vector store
vectorstore = PGVectorStore(
    connection_string="postgresql://user:password@localhost:5432/dbname",
    schema="public",
    min_size=2,
    max_size=10
)

# Create a collection
vectorstore.create_collection(
    collection_name="documents",
    vector_config=[VectorConfig(name="default", dimensions=384)]
)

# Add documents
chunk = Chunk(
    id="doc_1",
    text="PostgreSQL is a powerful database",
    metadata={"category": "database", "year": 2024},
    embeddings=[DenseEmbedding(name="default", vector=[0.1] * 384)]
)
vectorstore.add(chunk, collection_name="documents")

# Search
results = vectorstore.search(
    collection_name="documents",
    query_vector=[0.1] * 384,
    k=5,
    filters={"category": "database"}
)

# Clean up
vectorstore.close()
```

---

## Usage Examples

### Initialization

```python
from datapizza.vectorstores.pgvector import PGVectorStore

# Basic initialization
store = PGVectorStore(
    connection_string="postgresql://localhost/mydb"
)

# Advanced configuration with connection pooling
store = PGVectorStore(
    connection_string="postgresql://user:pass@localhost:5432/mydb",
    schema="custom_schema",
    min_size=5,      # Minimum pool connections
    max_size=20      # Maximum pool connections
)

# Context manager for automatic resource cleanup
with PGVectorStore(connection_string="postgresql://localhost/mydb") as store:
    # Your operations here
    pass  # Pool closes automatically
```

### Creating Collections

```python
from datapizza.core.vectorstore import VectorConfig

# Create a collection with specific vector dimensions
store.create_collection(
    collection_name="my_documents",
    vector_config=[
        VectorConfig(name="default", dimensions=768)  # e.g., for BERT embeddings
    ]
)
```

### Adding Documents

```python
from datapizza.type import Chunk, DenseEmbedding

# Single document
chunk = Chunk(
    id="article_001",
    text="Artificial Intelligence is transforming industries",
    metadata={"category": "tech", "author": "John Doe"},
    embeddings=[DenseEmbedding(name="default", vector=[0.2] * 768)]
)
store.add(chunk, collection_name="my_documents")

# Multiple documents (batch)
chunks = [
    Chunk(
        id=f"doc_{i}",
        text=f"Document content {i}",
        metadata={"batch": 1, "index": i},
        embeddings=[DenseEmbedding(name="default", vector=[0.1 * i] * 768)]
    )
    for i in range(100)
]
store.add(chunks, collection_name="my_documents", batch_size=50)
```

**Note:** The `add` method automatically handles duplicates via upsert logic—existing documents with the same ID will be replaced.

### Searching with Filters

```python
# Simple similarity search
results = store.search(
    collection_name="my_documents",
    query_vector=[0.15] * 768,
    k=10
)

# Search with metadata filters (PostgreSQL JSONB operators)
results = store.search(
    collection_name="my_documents",
    query_vector=[0.15] * 768,
    k=5,
    filters={"category": "tech"}  # Only return documents with category="tech"
)

# Complex filters (nested metadata)
results = store.search(
    collection_name="my_documents",
    query_vector=[0.15] * 768,
    k=5,
    filters={"author": "John Doe", "metadata.verified": True}
)

for chunk in results:
    print(f"ID: {chunk.id}, Score: {chunk.metadata}, Text: {chunk.text[:50]}...")
```

### Async Operations

```python
import asyncio
from datapizza.vectorstores.pgvector import PGVectorStore

async def main():
    store = PGVectorStore(
        connection_string="postgresql://localhost/mydb",
        min_size=3,
        max_size=15
    )
    
    # Async add
    await store.a_add(
        chunk=Chunk(id="async_doc", text="Async example", embeddings=[...]),
        collection_name="my_documents"
    )
    
    # Async search
    results = await store.a_search(
        collection_name="my_documents",
        query_vector=[0.2] * 768,
        k=10,
        filters={"category": "news"}
    )
    
    # Async cleanup
    await store.aclose()

asyncio.run(main())
```

### Managing Collections

```python
# List all collections
collections = store.get_collections()
print(f"Available collections: {collections}")

# Async version
collections = await store.a_get_collections()

# Export/dump a collection (memory-efficient generator)
for chunk in store.dump_collection(
    collection_name="my_documents",
    page_size=100,
    with_vectors=True
):
    print(f"Exported: {chunk.id}")

# Delete a collection
store.delete_collection("old_collection")

# Remove specific documents
store.remove(collection_name="my_documents", ids=["doc_1", "doc_2", "doc_3"])

# Async remove
await store.a_remove(collection_name="my_documents", ids=["doc_4"])
```

---

## Advanced Features

### Connection Pooling

The vector store uses `psycopg_pool` to manage database connections efficiently:

- **Sync Pool**: Automatically initialized on first use
- **Async Pool**: Separate pool for async operations, opened on-demand
- **Configuration**: Control pool size via `min_size` and `max_size` parameters

```python
store = PGVectorStore(
    connection_string="postgresql://localhost/mydb",
    min_size=5,   # Keep 5 connections ready
    max_size=50   # Max 50 concurrent connections
)
```

### Metadata Filtering

Leverage PostgreSQL's powerful JSONB operators for advanced filtering:

```python
# Exact match
filters = {"status": "published"}

# Multiple conditions (implicit AND)
filters = {"category": "tech", "verified": True}

# The implementation uses the @> (contains) operator
# metadata @> '{"category": "tech"}' in SQL
```

### Resource Management

```python
# Manual cleanup
store.close()        # Close sync pool
await store.aclose() # Close async pool

# Context manager (recommended)
with PGVectorStore(connection_string="...") as store:
    store.add(...)   # Automatic cleanup on exit

# Destructor handles cleanup automatically
store = PGVectorStore(connection_string="...")
# ... use store ...
# Pool closes when object is garbage collected
```

---

## Testing

The package includes a comprehensive test suite:

- **Unit Tests**: Core functionality and edge cases
- **Functional Tests**: End-to-end workflows
- **Smoke Tests**: Quick validation of critical paths
- **Stress Tests**: Performance and concurrency testing

Run tests with:

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest datapizza/vectorstores/pgvector/tests/

# Run with coverage
pytest --cov=datapizza.vectorstores.pgvector
```

---

## API Reference

### Main Class: `PGVectorStore`

#### Initialization

```python
PGVectorStore(
    connection_string: str,
    schema: str = "public",
    min_size: int = 1,
    max_size: int = 10,
    **kwargs
)
```

#### Core Methods

| Method | Description | Async Version |
|--------|-------------|---------------|
| `create_collection(collection_name, vector_config)` | Create a new collection | N/A |
| `add(chunk, collection_name, batch_size)` | Add documents | `a_add()` |
| `search(collection_name, query_vector, k, filters)` | Search similar documents | `a_search()` |
| `remove(collection_name, ids)` | Remove documents by ID | `a_remove()` |
| `get_collections()` | List all collections | `a_get_collections()` |
| `dump_collection(collection_name, page_size, with_vectors)` | Export collection data | N/A |
| `delete_collection(collection_name)` | Drop entire collection | N/A |
| `retrieve(collection_name, ids)` | Get specific documents | N/A |
| `update(collection_name, payload, points)` | Update metadata | N/A |
| `close()` | Close sync pool | `aclose()` |

---

## License

MIT License - see LICENSE file for details.

---

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass (`pytest`)
2. Code follows PEP 8 style guidelines
3. New features include tests and documentation

---

## Support

For issues and questions:

- GitHub Issues: [datapizza-ai](https://github.com/your-org/datapizza-ai/issues)
- Documentation: [Full Datapizza AI Docs](https://docs.datapizza.ai)

---

**Built with ❤️ for the Datapizza AI ecosystem**
