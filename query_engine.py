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


def extract_year(query: str):
    match = re.search(r"\b(20\d{2})\b", query)
    return match.group(1) if match else None


def run_query_engine(config_path):
    config = load_config(config_path)

    vectordb = load_vector_db(config)

    print("Vector count:", vectordb._collection.count())

    while True:
        query = input("\nAsk a question (or 'exit'): ")

        if query.lower() == "exit":
            break

        # 🔴 Step 1: Extract year (if present)
        year = extract_year(query)

        # 🔴 Step 2: Query augmentation
        enhanced_query = f"{query}. Tesla financial statements revenue details"

        # 🔴 Step 3: Apply filter if year exists
        if year:
            print(f"Applying year filter: {year}")
            retriever = vectordb.as_retriever(
                search_kwargs={
                    "k": 10,
                    "filter": {"source": f"NASDAQ_TSLA_{year}.txt"}
                }
            )
        else:
            retriever = vectordb.as_retriever(search_kwargs={"k": 10})

        docs = retriever.invoke(enhanced_query)

        print("\n--- Retrieved Chunks ---")
        for doc in docs:
            print(doc.metadata)
            print(doc.page_content[:300])
            print("-----")

        context = "\n\n".join([doc.page_content for doc in docs])

        answer = generate_answer(context, query)

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    run_query_engine(
        "/Users/lokeshkv/data-engineering/Tesla_Financial_Document_Q_and_A_System_using_RAG/config/parameters.yaml"
    )