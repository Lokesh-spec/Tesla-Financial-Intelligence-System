import yaml
import re
from pathlib import Path
from retrieval.retriever import load_vector_db
from generation.generator import generate_answer
from dotenv import load_dotenv

load_dotenv()


def load_config(config_path):
    config_path = Path(config_path)

    if not config_path.exists():
        print(f"Looking for file at: {config_path.absolute()}")
        raise FileNotFoundError("Config file not found")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_years(query: str) -> list:
    """
    Extract all years mentioned in the query.
    Returns list like ['2022', '2023']
    """
    return re.findall(r"\b(20\d{2})\b", query)


def deduplicate_chunks(docs: list) -> list:
    """
    Remove duplicate chunks using chunk_id.
    """
    seen = set()
    unique_docs = []

    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique_docs.append(doc)

    return unique_docs


def build_enhanced_query(query: str, years: list) -> str:
    """
    Improve query for better retrieval.
    """
    if years:
        return f"{query}. Tesla {years[0]} annual report total revenues consolidated financial statements"
    return f"{query}. Tesla financial statements revenue details"


def retrieve_chunks(vectordb, enhanced_query: str, years: list, top_k: int) -> list:
    """
    High-recall retrieval + soft filtering
    """

    # Increase recall pool
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k * 3})
    docs = retriever.invoke(enhanced_query)

    if not years:
        return docs[:top_k]

    print(f"Applying soft year filter: {years}")

    filtered_docs = []

    for doc in docs:
        source = doc.metadata.get("source", "").lower()

        for year in years:
            if year in source:
                filtered_docs.append(doc)
                break

    # fallback if filtering removes everything
    if len(filtered_docs) == 0:
        print("No exact year match found, returning top results")
        return docs[:top_k]

    return filtered_docs[:top_k]


def classify_query(query: str) -> str:
    query_lower =query.lower()

    if any(word in query_lower for word in 
           ["compare", "vs", "versus", "difference", 
            "between", "contrast", "better", "worse"]):
        return {"k": 15, "type": "comparison"}
    
    elif any(word in query_lower for word in 
             ["summary", "overview", "overall", "explain", 
              "describe", "tell me about", "what happened"]):
        return {"k": 10, "type": "summary"}
    
    else:
        return {"k": 5, "type": "specific"}

# def run_query_engine(config_path):
#     config = load_config(config_path)
    
#     vectordb = load_vector_db(config)

#     print("Vector count:", vectordb._collection.count())

#     while True:
#         query = input("\nAsk a question (or 'exit'): ")

#         search_params = classify_query(query)
#         top_k = search_params["k"]
#         query_type = search_params["type"]

#         if query.lower() == "exit":
#             break

#         years = extract_years(query)

#         enhanced_query = build_enhanced_query(query, years)

#         raw_docs = retrieve_chunks(vectordb, enhanced_query, years, top_k)

#         docs = deduplicate_chunks(raw_docs)

#         print("\n--- Retrieved Chunks ---")
#         for doc in docs:
#             print(doc.metadata)
#             print(doc.page_content[:300])
#             print("-----")

#         print(f"\nTotal unique chunks retrieved: {len(docs)}")

#         context = "\n\n".join([doc.page_content for doc in docs])

#         answer = generate_answer(context, query, query_type)

#         print("\nAnswer:")
#         print(answer)

def run_query_engine(query: str, config_path: str):
    config = load_config(config_path)
    
    vectordb = load_vector_db(config)

    print("Vector count:", vectordb._collection.count())

    search_params = classify_query(query)
    top_k = search_params["k"]
    query_type = search_params["type"]

    years = extract_years(query)

    enhanced_query = build_enhanced_query(query, years)

    raw_docs = retrieve_chunks(vectordb, enhanced_query, years, top_k)

    docs = deduplicate_chunks(raw_docs)

    print("\n--- Retrieved Chunks ---")
    for doc in docs:
        print(doc.metadata)
        print(doc.page_content[:300])
        print("-----")

    print(f"\nTotal unique chunks retrieved: {len(docs)}")

    context = "\n\n".join([doc.page_content for doc in docs])

    answer = generate_answer(context, query, query_type)

    print("\nAnswer:")
    print(answer)

