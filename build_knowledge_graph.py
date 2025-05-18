import time
import re
import wikipediaapi
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
import json
import os
from neo4j import GraphDatabase
from langchain_huggingface import HuggingFaceEmbeddings

GROQ_API_KEY = "XXXXX"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "XXXX"
NEO4J_PASSWORD = "XXXX"

WIKI_TITLES = [
    "Supreme Court of the United States",
    "Constitution of the United States",
    "Geneva Conventions",
    "Universal Declaration of Human Rights",
    "International Criminal Court",
    "Brown v. Board of Education",
    "Fourth Amendment to the United States Constitution"
]

wiki = wikipediaapi.Wikipedia(user_agent="aayush_wiki",language="en")

visited = set()
articles = {}
depth = 1

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="llama3-70b-8192")

graph_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Law", "Court", "Case", "Person", "Institution", "Treaty", "Amendment", "Document", "Country", "Government", "Article", "Organization"],
    allowed_relationships=["decided_by", "ratified_by", "interprets", "applies_to", "established_by", "includes", "part_of", "describes", "has", "superseded"],
    node_properties=True,
    relationship_properties=True,
    strict_mode=False
)

def recurse_sections(sections):
    content = ""
    references = []
    for section in sections:
        content += f"\n\n== {section.title} ==\n{section.text}"
        contents, refs = recurse_sections(section.sections)
        content += contents
        references.extend(refs)
        if "reference" in section.title.lower():
            urls = re.findall(r'https?://\S+', section.text)
            references.extend(urls)
    return content, references
    
def get_full_sections_with_citations(title):
    page = wiki.page(title)
    content, references = recurse_sections(page.sections)
    text = page.summary + content
    return text, list(set(references))

def fetch_articles(titles, current_depth=0, depth=1):
    for title in titles:
        if title in visited or current_depth > depth:
            return
        
        visited.add(title)
        cache_path = f"cache/{title.replace('/', '_').replace(' ', '_')}.json"
        
        if os.path.exists(cache_path):
            print(f"Loaded from cache: {title}")
            with open(cache_path, "r", encoding="utf-8") as f:
                articles[title] = json.load(f)
        else:
            page = wiki.page(title)
            if not page.exists():
                print(f"Page not found: {title}")
                return

            text, citations = get_full_sections_with_citations(title)
            articles[title] = {"text": text, "citations": citations}
            
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(articles[title], f, ensure_ascii=False, indent=2)

            time.sleep(1.5)

        for linked_page in page.links.keys():
            fetch_articles(linked_page, current_depth + 1)
    
    return articles

print("Fetching Wikipedia Articles")
articles = fetch_articles(WIKI_TITLES, current_depth = 0, depth = 1)

documents = []
for title, data in articles.items():
    documents.append(Document(
        page_content=data["text"][:4000],
        metadata={
            "source": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "citations": data.get("citations", [])
        }
    ))

print("Transforming into Graph")
graph_documents = graph_transformer.convert_to_graph_documents(documents)

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

graph.add_graph_documents(graph_documents, include_source=True)