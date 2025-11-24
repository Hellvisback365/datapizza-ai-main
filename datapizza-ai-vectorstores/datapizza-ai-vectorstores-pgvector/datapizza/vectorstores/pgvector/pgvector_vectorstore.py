import json
import logging
from collections.abc import Generator
from typing import Any, TYPE_CHECKING

try:
    import psycopg
    from psycopg import sql
except ImportError:
    raise ImportError(
        "Missing dependencies! Install: pip install psycopg[binary] pgvector psycopg-pool"
    )

try:
    from pgvector.psycopg import register_vector
except ImportError:
    register_vector = None

try:
    from psycopg_pool import ConnectionPool, AsyncConnectionPool
except ImportError:
    ConnectionPool = None
    AsyncConnectionPool = None

if TYPE_CHECKING:
    from psycopg import Connection

from datapizza.core.vectorstore import Vectorstore, VectorConfig
from datapizza.type import Chunk, DenseEmbedding

log = logging.getLogger(__name__)


class PGVectorStore(Vectorstore):
    """
    Enterprise PostgreSQL implementation for Datapizza.
    Features: Connection Pooling, Native Async, and JSONB Metadata Filters.
    """

    def __init__(
        self,
        connection_string: str,
        schema: str = "public",
        min_size: int = 1,
        max_size: int = 10,
        **kwargs: Any,
    ):
        """
        Initialize the PGVectorStore.

        Args:
            connection_string (str): PostgreSQL connection string.
            schema (str, optional): Database schema to use. Defaults to "public".
            min_size (int, optional): Minimum pool size. Defaults to 1.
            max_size (int, optional): Maximum pool size. Defaults to 10.
            **kwargs: Additional keyword arguments.
        """
        # Store connection parameters for lazy initialization
        self.connection_string = connection_string
        self.schema = schema
        self.min_size = min_size
        self.max_size = max_size
        self.kwargs = kwargs
        self.batch_size: int = 100

        # Connection pools will be initialized lazily
        # Type hints for IDE support
        self.sync_pool: ConnectionPool  # type: ignore
        self.async_pool: AsyncConnectionPool  # type: ignore

    def get_client(self) -> ConnectionPool:  # type: ignore
        """Get or initialize the synchronous connection pool."""
        if not hasattr(self, "sync_pool") or self.sync_pool is None:
            self._init_client()
        return self.sync_pool

    def _get_a_client(self) -> AsyncConnectionPool:  # type: ignore
        """Get or initialize the asynchronous connection pool."""
        if not hasattr(self, "async_pool") or self.async_pool is None:
            self._init_a_client()
        return self.async_pool

    def _init_client(self) -> None:
        """Initialize the synchronous connection pool and database schema."""
        self.sync_pool = ConnectionPool(  # type: ignore
            self.connection_string,
            min_size=self.min_size,
            max_size=self.max_size,
            kwargs={"autocommit": True},
            open=True,
        )

        # Initialize database schema on first connection
        with self.get_client().connection() as conn:
            register_vector(conn)  # type: ignore
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")  # type: ignore
                cur.execute(
                    sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                        sql.Identifier(self.schema)
                    )
                )  # type: ignore

    def _init_a_client(self) -> None:
        """Initialize the asynchronous connection pool."""
        self.async_pool = AsyncConnectionPool(  # type: ignore
            self.connection_string,
            min_size=self.min_size,
            max_size=self.max_size,
            kwargs={"autocommit": True},
            open=False,
        )

    def create_collection(
        self, collection_name: str, vector_config: list[VectorConfig], **kwargs: Any
    ) -> None:
        """
        Create a new collection if it doesn't exist.

        Args:
            collection_name (str): Name of the collection to create.
            vector_config (list[VectorConfig]): List of vector configurations.
            **kwargs: Additional arguments (unused).
        """
        try:
            dim = vector_config[0].dimensions
            
            # Use JSONB for metadata to filter quickly
            query = sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {table} (
                    id TEXT PRIMARY KEY,
                    text TEXT,
                    metadata JSONB, 
                    embedding vector({dim})
                );
                CREATE INDEX IF NOT EXISTS {idx} ON {table} USING gin (metadata);
                """
            ).format(
                table=sql.Identifier(self.schema, collection_name),
                dim=sql.Literal(dim),
                idx=sql.Identifier(f"idx_{collection_name}_meta"),
            )
            with self.get_client().connection() as conn:
                conn.execute(query)  # type: ignore
        except Exception as e:
            log.error(f"Failed to create collection '{collection_name}': {e!s}")
            raise

    def add(
        self,
        chunk: Chunk | list[Chunk],
        collection_name: str | None = None,
        batch_size: int = 100,
        **kwargs: Any,
    ) -> None:
        """
        Add chunks to the collection.

        Args:
            chunk (Chunk | list[Chunk]): Single chunk or list of chunks to add.
            collection_name (str, optional): Name of the collection. Required.
            batch_size (int, optional): Batch size for processing (unused).
            **kwargs: Additional arguments.

        Raises:
            ValueError: If collection_name is not provided.
        """
        if collection_name is None:
            raise ValueError("collection_name is required")

        try:
            nodes = [chunk] if isinstance(chunk, Chunk) else chunk

            # --- FIX DUPLICATES: Collect IDs and delete them before inserting ---
            ids_to_remove = [n.id for n in nodes]
            self.remove(collection_name, ids_to_remove)
            # -------------------------------------------------------------------

            with self.get_client().connection() as conn:
                for node in nodes:
                    vector = self._extract_vector(node)
                    if vector:
                        query = sql.SQL(
                            "INSERT INTO {table} (id, text, metadata, embedding) VALUES (%s, %s, %s, %s)"
                        ).format(table=sql.Identifier(self.schema, collection_name))
                        conn.execute(  # type: ignore
                            query,
                            (
                                node.id,
                                node.text,
                                json.dumps(node.metadata) if node.metadata else "{}",
                                vector,
                            ),
                        )
        except Exception as e:
            log.error(f"Failed to add chunks to collection '{collection_name}': {e!s}")
            raise

    def search(
        self,
        collection_name: str,
        query_vector: list[float] | Any,
        k: int = 10,
        vector_name: str | None = None,
        filters: dict[str, Any] | None = None,  # <--- PRO FEATURE 2: FILTERS
        **kwargs: Any,
    ) -> list[Chunk]:
        """
        Search for similar chunks using vector similarity.

        Args:
            collection_name (str): Name of the collection to search.
            query_vector (list[float] | Any): Query vector for similarity search.
            k (int, optional): Number of results to return. Defaults to 10.
            vector_name (str | None, optional): Name of the vector field (unused).
            filters (dict[str, Any] | None, optional): Optional JSONB metadata filters.
            **kwargs: Additional arguments.

        Returns:
            list[Chunk]: List of similar chunks.
        """
        try:
            # Handle case where query_vector is an Embedding object or list
            if hasattr(query_vector, "values") and not isinstance(query_vector, list):
                query_vector = query_vector.values  # type: ignore

            # Dynamic Query Construction with Filters
            where_clause = ""
            sql_params = []

            if filters:
                # Postgres JSONB syntax: metadata @> '{"key": "value"}'
                where_clause = "WHERE metadata @> %s"
                sql_params.append(json.dumps(filters))

            # Convert vector to pgvector string format
            vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"

            query = sql.SQL(
                """
                SELECT id, text, metadata, embedding 
                FROM {table}
                {where}
                ORDER BY embedding <=> %s::vector 
                LIMIT %s
                """
            ).format(
                table=sql.Identifier(self.schema, collection_name),
                where=sql.SQL(where_clause),
            )
            sql_params.extend([vector_str, k])
            results = []
            with self.get_client().connection() as conn:
                rows = conn.execute(query, sql_params).fetchall()  # type: ignore
                for row in rows:
                    results.append(self._row_to_chunk(row))
            return results
        except Exception as e:
            log.error(f"Failed to search in collection '{collection_name}': {e!s}")
            raise

    # --- ASYNC VERSION (TRUE) ---
    # PRO FEATURE 3: NATIVE ASYNC
    # Here we use true 'async def' and 'await', not calling the synchronous version.

    async def a_add(
        self,
        chunk: Chunk | list[Chunk],
        collection_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Add chunks to the collection (Async).

        Args:
            chunk (Chunk | list[Chunk]): Single chunk or list of chunks to add.
            collection_name (str, optional): Name of the collection. Required.
            **kwargs: Additional arguments.

        Raises:
            ValueError: If collection_name is not provided.
        """
        if collection_name is None:
            raise ValueError("collection_name required")

        try:
            nodes = [chunk] if isinstance(chunk, Chunk) else chunk

            # --- FIX DUPLICATES: Collect IDs and delete them before inserting ---
            ids_to_remove = [n.id for n in nodes]
            await self.a_remove(collection_name, ids_to_remove)
            # -------------------------------------------------------------------

            # Ensure async pool is open
            async_pool = self._get_a_client()
            if not async_pool._opened:  # type: ignore
                await async_pool.open()  # type: ignore

            async with self._get_a_client().connection() as conn:
                for node in nodes:
                    vector = self._extract_vector(node)
                    if vector:
                        query = sql.SQL(
                            "INSERT INTO {table} (id, text, metadata, embedding) VALUES (%s, %s, %s, %s)"
                        ).format(table=sql.Identifier(self.schema, collection_name))
                        await conn.execute(  # type: ignore
                            query,
                            (
                                node.id,
                                node.text,
                                json.dumps(node.metadata) if node.metadata else "{}",
                                vector,
                            ),
                        )
        except Exception as e:
            log.error(
                f"Failed to add chunks async to collection '{collection_name}': {e!s}"
            )
            raise

    async def a_search(
        self,
        collection_name: str,
        query_vector: list[float] | Any,
        k: int = 10,
        vector_name: str | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Chunk]:
        """
        Search for similar chunks using vector similarity (Async).

        Args:
            collection_name (str): Name of the collection to search.
            query_vector (list[float] | Any): Query vector for similarity search.
            k (int, optional): Number of results to return. Defaults to 10.
            vector_name (str | None, optional): Name of the vector field (unused).
            filters (dict[str, Any] | None, optional): Optional JSONB metadata filters.
            **kwargs: Additional arguments.

        Returns:
            list[Chunk]: List of similar chunks.
        """
        try:
            if hasattr(query_vector, "values") and not isinstance(query_vector, list):
                query_vector = query_vector.values  # type: ignore

            where_clause = "WHERE metadata @> %s" if filters else ""

            sql_params = []
            if filters:
                sql_params.append(json.dumps(filters))

            vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"

            query = sql.SQL(
                """
                SELECT id, text, metadata, embedding 
                FROM {table}
                {where}
                ORDER BY embedding <=> %s::vector 
                LIMIT %s
                """
            ).format(
                table=sql.Identifier(self.schema, collection_name),
                where=sql.SQL(where_clause),
            )
            sql_params.extend([vector_str, k])
            results = []
            # Ensure async pool is open
            async_pool = self._get_a_client()
            if not async_pool._opened:  # type: ignore
                await async_pool.open()  # type: ignore
            async with self._get_a_client().connection() as conn:
                try:
                    register_vector(conn)  # type: ignore
                except Exception:
                    pass
                cur = await conn.execute(query, sql_params)  # type: ignore
                rows = await cur.fetchall()  # type: ignore
                for row in rows:
                    results.append(self._row_to_chunk(row))
            return results
        except Exception as e:
            log.error(f"Failed to search async in collection '{collection_name}': {e!s}")
            raise

    # --- HELPER UTILS ---
    def _extract_vector(self, node: Chunk) -> list[float] | None:
        """Extract the first dense vector found."""
        for emb in node.embeddings:
            if isinstance(emb, DenseEmbedding):
                return emb.vector
        return None

    def _row_to_chunk(self, row) -> Chunk:
        """Convert DB row to Chunk."""
        chunk_id, text, metadata, vec = row
        # Convert numpy array or similar to list if necessary
        vector_list = vec.tolist() if hasattr(vec, "tolist") else vec
        return Chunk(
            id=chunk_id,
            text=text,
            metadata=metadata,
            embeddings=[DenseEmbedding(name="default", vector=vector_list)],
        )

    def update(
        self,
        collection_name: str,
        payload: dict,
        points: list[int],
        **kwargs: Any,
    ) -> None:
        """
        Update metadata of existing records.
        
        Args:
            collection_name (str): Name of the collection.
            payload (dict): New metadata payload.
            points (list[int]): List of point IDs to update.
            **kwargs: Additional arguments.
        """
        query = sql.SQL("UPDATE {table} SET metadata = %s WHERE id = %s").format(
            table=sql.Identifier(self.schema, collection_name)
        )

        with self.get_client().connection() as conn:
            for point_id in points:
                conn.execute(query, (json.dumps(payload), str(point_id)))  # type: ignore

    def remove(self, collection_name: str, ids: list[str], **kwargs: Any) -> None:
        """
        Remove specific chunks by ID.
        
        Args:
            collection_name (str): Name of the collection.
            ids (list[str]): List of IDs to remove.
            **kwargs: Additional arguments.
        """
        query = sql.SQL("DELETE FROM {table} WHERE id = ANY(%s)").format(
            table=sql.Identifier(self.schema, collection_name)
        )

        with self.get_client().connection() as conn:
            conn.execute(query, (ids,))

    async def a_remove(
        self, collection_name: str, ids: list[str], **kwargs: Any
    ) -> None:
        """
        Remove specific chunks by ID (Async).
        
        Args:
            collection_name (str): Name of the collection.
            ids (list[str]): List of IDs to remove.
            **kwargs: Additional arguments.
        """
        query = sql.SQL("DELETE FROM {table} WHERE id = ANY(%s)").format(
            table=sql.Identifier(self.schema, collection_name)
        )

        # Ensure async pool is open
        async_pool = self._get_a_client()
        if not async_pool._opened:
            await async_pool.open()

        async with self._get_a_client().connection() as conn:
            await conn.execute(query, (ids,))

    def delete_collection(self, collection_name: str, **kwargs: Any) -> None:
        """
        Delete the entire collection (table).
        
        Args:
            collection_name (str): Name of the collection to delete.
            **kwargs: Additional arguments.
        """
        query = sql.SQL("DROP TABLE IF EXISTS {table}").format(
            table=sql.Identifier(self.schema, collection_name)
        )

        with self.get_client().connection() as conn:
            conn.execute(query)  # type: ignore

    def retrieve(
        self, collection_name: str, ids: list[str], **kwargs: Any
    ) -> list[Chunk]:
        """
        Retrieve specific records by ID.
        
        Args:
            collection_name (str): Name of the collection.
            ids (list[str]): List of IDs to retrieve.
            **kwargs: Additional arguments.
            
        Returns:
            list[Chunk]: List of retrieved chunks.
        """
        query = sql.SQL(
            "SELECT id, text, metadata, embedding FROM {table} WHERE id = ANY(%s)"
        ).format(table=sql.Identifier(self.schema, collection_name))

        results = []
        with self.get_client().connection() as conn:
            rows = conn.execute(query, (ids,)).fetchall()  # type: ignore
            for row in rows:
                results.append(self._row_to_chunk(row))
        return results

    def dump_collection(
        self,
        collection_name: str,
        page_size: int = 100,
        with_vectors: bool = False,
        **kwargs: Any,
    ) -> Generator[Chunk, None, None]:
        """
        Dump all chunks from a collection in a chunk-wise manner.

        Args:
            collection_name (str): Name of the collection to dump.
            page_size (int, optional): Number of points to retrieve per batch. Defaults to 100.
            with_vectors (bool, optional): Whether to include vectors. Defaults to False.
            **kwargs: Additional arguments.

        Yields:
            Chunk: A chunk object from the collection.
        """
        offset = 0
        while True:
            # Construct query based on with_vectors
            if with_vectors:
                query = sql.SQL(
                    "SELECT id, text, metadata, embedding FROM {table} LIMIT %s OFFSET %s"
                ).format(table=sql.Identifier(self.schema, collection_name))
            else:
                query = sql.SQL(
                    "SELECT id, text, metadata, NULL as embedding FROM {table} LIMIT %s OFFSET %s"
                ).format(table=sql.Identifier(self.schema, collection_name))

            with self.sync_pool.connection() as conn:
                rows = conn.execute(query, (page_size, offset)).fetchall()
            
            if not rows:
                break
                
            for row in rows:
                yield self._row_to_chunk(row)
            
            if len(rows) < page_size:
                break
                
            offset += page_size

    def get_collections(self) -> list[str]:
        """
        Get all collections (tables) in the schema.

        Returns:
            list[str]: List of collection names in the current schema.
        """
        query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = %s 
            AND table_type = 'BASE TABLE'
        """

        with self.get_client().connection() as conn:
            rows = conn.execute(query, (self.schema,)).fetchall()

        return [row[0] for row in rows]

    async def a_get_collections(self) -> list[str]:
        """
        Get all collections (tables) in the schema (Async).

        Returns:
            list[str]: List of collection names in the current schema.
        """
        query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = %s 
            AND table_type = 'BASE TABLE'
        """

        # Ensure async pool is open
        async_pool = self._get_a_client()
        if not async_pool._opened:
            await async_pool.open()

        async with self._get_a_client().connection() as conn:
            cur = await conn.execute(query, (self.schema,))
            rows = await cur.fetchall()

        return [row[0] for row in rows]

    # --- RESOURCE MANAGEMENT (CLEANUP) ---
    def close(self) -> None:
        """Close the synchronous pool."""
        if hasattr(self, "sync_pool"):
            self.sync_pool.close()

    async def aclose(self) -> None:
        """Close the asynchronous pool."""
        if hasattr(self, "async_pool"):
            await self.async_pool.close()

    # Support for 'with' statement (Context Manager)
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        """Destructor - close pools when object is destroyed."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during destruction
