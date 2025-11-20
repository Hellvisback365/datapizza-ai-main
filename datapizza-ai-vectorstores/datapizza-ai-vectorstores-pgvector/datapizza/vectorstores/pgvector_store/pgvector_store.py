import json
from typing import Any, List, Optional, Dict, TYPE_CHECKING
import logging

try:
    import psycopg
except ImportError:
    psycopg = None

try:
    from pgvector.psycopg import register_vector
    print(f"register_vector imported: {register_vector}")
except ImportError as e:
    print(f"Import error register_vector: {e}")
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

logger = logging.getLogger(__name__)

class PGVectorStore(Vectorstore):
    """
    Implementazione Enterprise di Postgres per Datapizza.
    Include: Connection Pooling, Async nativo e Filtri Metadata JSONB.
    """

    def __init__(
        self,
        connection_string: str,
        schema: str = "public",
        min_size: int = 1,
        max_size: int = 10,
        **kwargs: Any,
    ):
        if psycopg is None:
            raise ImportError(
                "Mancano le dipendenze! Installa: pip install psycopg[binary] pgvector psycopg-pool"
            )
        
        self.connection_string = connection_string
        self.schema = schema
        
        # --- PRO FEATURE 1: CONNECTION POOLING ---
        # Invece di una singola connessione, ne creiamo una "piscina" (pool).
        # Il database prenderà e rilascerà connessioni in automatico.
        
        # Pool per le chiamate sincrone (normali)
        self.sync_pool = ConnectionPool(  # type: ignore
            connection_string,
            min_size=min_size,
            max_size=max_size,
            kwargs={"autocommit": True},
            open=True # Apre subito le connessioni
        )
        
        # Pool per le chiamate asincrone (async/await)
        # NON apriamo subito perché non c'è un event loop attivo
        self.async_pool = AsyncConnectionPool(  # type: ignore
            connection_string,
            min_size=min_size,
            max_size=max_size,
            kwargs={"autocommit": True},
            open=False  # Apriremo on-demand
        )

        # Inizializziamo il DB (schema ed estensione) usando il pool sincrono
        self._init_db()

    def _init_db(self):
        """Prepara il database al primo avvio."""
        with self.sync_pool.connection() as conn:
            register_vector(conn)  # type: ignore
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")  # type: ignore
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")  # type: ignore

    def create_collection(self, collection_name: str, vector_config: List[VectorConfig]) -> None:
        dim = vector_config[0].dimensions
        table_name = f"{self.schema}.{collection_name}"

        # Usiamo JSONB per i metadati per poterli filtrare velocemente
        from psycopg import sql
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
            idx=sql.Identifier(f"idx_{collection_name}_meta")
        )
        with self.sync_pool.connection() as conn:
            conn.execute(query)  # type: ignore

    # --- VERSIONE SINCRONA ---
    def add(
        self,
        chunk: Chunk | List[Chunk],
        collection_name: str | None = None,
        batch_size: int = 100,
        **kwargs: Any,
    ) -> None:
        if collection_name is None:
            raise ValueError("collection_name is required")
        
        nodes = [chunk] if isinstance(chunk, Chunk) else chunk
        
        # --- FIX DUPLICATI: Raccogli gli ID e cancellali prima di inserire ---
        ids_to_remove = [n.id for n in nodes]
        self.remove(collection_name, ids_to_remove) 
        # -------------------------------------------------------------------
        
        table_name = f"{self.schema}.{collection_name}"
        
        from psycopg import sql
        with self.sync_pool.connection() as conn:
            for node in nodes:
                vector = self._extract_vector(node)
                if vector:
                    query = sql.SQL("INSERT INTO {table} (id, text, metadata, embedding) VALUES (%s, %s, %s, %s)").format(
                        table=sql.Identifier(self.schema, collection_name)
                    )
                    conn.execute(  # type: ignore
                        query,
                        (
                            node.id,
                            node.text,
                            json.dumps(node.metadata) if node.metadata else "{}",
                            vector
                        )
                    )

    def search(
        self,
        collection_name: str,
        query_vector: List[float] | Any, 
        k: int = 10,
        vector_name: str | None = None,
        filters: Optional[Dict[str, Any]] = None,  # <--- PRO FEATURE 2: FILTRI
        **kwargs: Any,
    ) -> List[Chunk]:
        
        # Gestiamo il caso in cui query_vector sia un oggetto Embedding o lista
        if hasattr(query_vector, "values") and not isinstance(query_vector, list):
             query_vector = query_vector.values  # type: ignore

        table_name = f"{self.schema}.{collection_name}"
        
        # Costruzione Query Dinamica con Filtri
        where_clause = ""
        sql_params = []
        
        if filters:
            # Sintassi Postgres JSONB: metadata @> '{"key": "value"}'
            where_clause = "WHERE metadata @> %s"
            sql_params.append(json.dumps(filters))

        # Convertiamo il vettore in stringa formato pgvector
        vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"
        
        from psycopg import sql
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
            where=sql.SQL(where_clause)
        )
        sql_params.extend([vector_str, k])
        results = []
        with self.sync_pool.connection() as conn:
            rows = conn.execute(query, sql_params).fetchall()  # type: ignore
            for row in rows:
                results.append(self._row_to_chunk(row))
        return results

    # --- VERSIONE ASYNC (VERA) ---
    # PRO FEATURE 3: ASYNC NATIVO
    # Qui usiamo 'async def' e 'await' veri, non chiamiamo la versione sincrona.
    
    async def a_add(
        self,
        chunk: Chunk | List[Chunk],
        collection_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        if collection_name is None: 
            raise ValueError("collection_name required")
        nodes = [chunk] if isinstance(chunk, Chunk) else chunk
        
        # --- FIX DUPLICATI: Raccogli gli ID e cancellali prima di inserire ---
        ids_to_remove = [n.id for n in nodes]
        await self.a_remove(collection_name, ids_to_remove)
        # -------------------------------------------------------------------

        table_name = f"{self.schema}.{collection_name}"

        # Assicuriamoci che il pool async sia aperto
        if not self.async_pool._opened:  # type: ignore
            await self.async_pool.open()  # type: ignore
        
        from psycopg import sql
        async with self.async_pool.connection() as conn:
            for node in nodes:
                vector = self._extract_vector(node)
                if vector:
                    query = sql.SQL("INSERT INTO {table} (id, text, metadata, embedding) VALUES (%s, %s, %s, %s)").format(
                        table=sql.Identifier(self.schema, collection_name)
                    )
                    await conn.execute(  # type: ignore
                        query,
                        (
                            node.id,
                            node.text,
                            json.dumps(node.metadata) if node.metadata else "{}",
                            vector
                        )
                    )

    async def a_search(
        self,
        collection_name: str,
        query_vector: List[float] | Any,
        k: int = 10,
        vector_name: str | None = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Chunk]:
        if hasattr(query_vector, "values") and not isinstance(query_vector, list): 
            query_vector = query_vector.values  # type: ignore
        
        table_name = f"{self.schema}.{collection_name}"
        where_clause = "WHERE metadata @> %s" if filters else ""
        
        sql_params = []
        if filters:
            sql_params.append(json.dumps(filters))
        
        vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"
        
        from psycopg import sql
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
            where=sql.SQL(where_clause)
        )
        sql_params.extend([vector_str, k])
        results = []
        # Assicuriamoci che il pool async sia aperto
        if not self.async_pool._opened:  # type: ignore
            await self.async_pool.open()  # type: ignore
        async with self.async_pool.connection() as conn:
            try:
                register_vector(conn)  # type: ignore
            except:
                pass
            cur = await conn.execute(query, sql_params)  # type: ignore
            rows = await cur.fetchall()  # type: ignore
            for row in rows:
                results.append(self._row_to_chunk(row))
        return results

    # --- HELPER UTILS ---
    def _extract_vector(self, node: Chunk) -> Optional[List[float]]:
        """Estrae il primo vettore denso trovato."""
        for emb in node.embeddings:
            if isinstance(emb, DenseEmbedding):
                return emb.vector
        return None

    def _row_to_chunk(self, row) -> Chunk:
        """Converte riga DB in Chunk."""
        chunk_id, text, metadata, vec = row
        # Convertiamo numpy array o simili in lista se necessario
        vector_list = vec.tolist() if hasattr(vec, "tolist") else vec
        return Chunk(
            id=chunk_id,
            text=text,
            metadata=metadata,
            embeddings=[DenseEmbedding(name="default", vector=vector_list)]
        )

    def update(
        self, 
        collection_name: str, 
        payload: dict, 
        points: List[int], 
        **kwargs: Any
    ) -> None:
        """Aggiorna metadati di record esistenti."""
        table_name = f"{self.schema}.{collection_name}"
        with self.sync_pool.connection() as conn:
            for point_id in points:
                query_str = f"UPDATE {table_name} SET metadata = %s WHERE id = %s"
                conn.execute(query_str, (json.dumps(payload), str(point_id)))  # type: ignore
    
    def remove(self, collection_name: str, ids: List[str], **kwargs: Any) -> None:
        """Rimuove chunk specifici per ID."""
        table_name = f"{self.schema}.{collection_name}"
        from psycopg import sql
        
        query = sql.SQL("DELETE FROM {table} WHERE id = ANY(%s)").format(
            table=sql.Identifier(self.schema, collection_name)
        )
        
        with self.sync_pool.connection() as conn:
            conn.execute(query, (ids,))

    async def a_remove(self, collection_name: str, ids: List[str], **kwargs: Any) -> None:
        """Rimuove chunk specifici per ID (Async)."""
        from psycopg import sql
        
        query = sql.SQL("DELETE FROM {table} WHERE id = ANY(%s)").format(
            table=sql.Identifier(self.schema, collection_name)
        )
        
        # Assicuriamoci che il pool async sia aperto
        if not self.async_pool._opened:
            await self.async_pool.open()

        async with self.async_pool.connection() as conn:
            await conn.execute(query, (ids,))

    def delete_collection(self, collection_name: str) -> None:
        """Elimina l'intera collection (tabella)."""
        with self.sync_pool.connection() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {self.schema}.{collection_name}")  # type: ignore
            
    def retrieve(self, collection_name: str, ids: List[str], **kwargs: Any) -> List[Chunk]:
        """Recupera record specifici per ID."""
        table_name = f"{self.schema}.{collection_name}"
        query = f"SELECT id, text, metadata, embedding FROM {table_name} WHERE id = ANY(%s)"
        results = []
        with self.sync_pool.connection() as conn:
            rows = conn.execute(query, (ids,)).fetchall()  # type: ignore
            for row in rows:
                results.append(self._row_to_chunk(row))
        return results

    # --- GESTIONE RISORSE (CLEANUP) ---
    def close(self) -> None:
        """Chiude il pool sincrono."""
        if hasattr(self, 'sync_pool'):
            self.sync_pool.close()

    async def aclose(self) -> None:
        """Chiude il pool asincrono."""
        if hasattr(self, 'async_pool'):
            await self.async_pool.close()

    # Supporto per il 'with' statement (Context Manager)
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        """Destructor - chiude i pool quando l'oggetto viene distrutto."""
        try:
            self.close()
        except Exception:
            pass  # Ignoriamo errori durante la distruzione
