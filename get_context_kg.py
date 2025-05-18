from neo4j import GraphDatabase
from langchain_huggingface import HuggingFaceEmbeddings

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "XXXX"
NEO4J_PASSWORD = "XXXX"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_relevant_nodes(embedded_query, top_k):
    cypher_query = f"""
    CALL db.index.vector.queryNodes('relevant_entity_embeddings_index', $k, $embedding)
    YIELD node, score
    RETURN elementId(node) AS node_id, node.name AS name, node.description AS description, labels(node) AS labels, score
    ORDER BY score DESC
    """
    with driver.session() as session:
        results = session.run(cypher_query, embedding=embedded_query, k=top_k)
        return [{"id": r["node_id"], "name": r["name"], "description": r.get("description", ""), "labels": r["labels"], "score": r["score"]} for r in results]
    
def expand_neighbors(node_ids):
    cypher_query = """
    MATCH (n)-[r]-(m)
    WHERE elementId(n) IN $node_ids
    RETURN n.name AS left_node, n.description AS left_desc, type(r) AS relationship, m.name AS right_node, m.description AS right_desc
    LIMIT 100
    """
    with driver.session() as session:
        results = session.run(cypher_query, node_ids=node_ids)
        neighbour_nodes = []
        for r in results:
            left_desc = r["left_desc"]
            right_desc = r["right_desc"]
            neighbour_nodes.append(f"{r['left_node']} ({left_desc}) --[{r['relationship']}]-- {r['right_node']} ({right_desc})")
        return neighbour_nodes

def get_full_context(pdf_context):
    embedded_query = embedding_model.embed_query(pdf_context)
    top_matching_nodes = get_relevant_nodes(embedded_query, top_k = 5)
    
    if not top_matching_nodes:
        print("No relevant nodes found.")
        return
    
    print("Top Retrieved Nodes:")
    for node in top_matching_nodes:
        print(f"{node['name']} (Similarity Score: {node['score']})")
        print(f"Description: {node.get('description', '')}")

    node_ids = [node['id'] for node in top_matching_nodes]
    neighbor_information = expand_neighbors(node_ids)
    
    graph_contexts = []
    
    for node in top_matching_nodes:
        description = node.get('description', '')
        graph_contexts.append(f"{node['name']} (Labels: {', '.join(node['labels'])}) {description}")
    
    if neighbor_information:
        for neighbour in neighbor_information:
            graph_contexts.append(neighbour)
    
    context_block = "\n".join(graph_contexts)
    
    return context_block