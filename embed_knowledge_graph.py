from neo4j import GraphDatabase
from langchain_huggingface import HuggingFaceEmbeddings

uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("XXXX", "XXXXX"))
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def embed_and_store(tx, external_id, vector):
    tx.run(
        """
        MATCH (n)
        WHERE elementId(n) = $node_id
        SET n.embedding = $vector
        """,
        external_id=external_id,
        vector=vector
    )

with driver.session() as session:
    result = session.run("""
        MATCH (d:Document)
        WHERE d.embedding IS NULL AND d.text IS NOT NULL
        RETURN d.external_id AS ext_id, d.text AS text
        LIMIT 500
    """)

    for record in result:
        external_id = record["ext_id"]
        text = record["text"]
        vector = embedder.embed_query(text)
        session.execute_write(embed_and_store, external_id, vector)

print("Embeddings stored in Neo4j!")