import yaml
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

def create_vector_db(config_path: [str, Path]) -> Chroma:
    """
    Loads JSON chunk files, converts them to documents, embeds them using 
    HuggingFace, and persists the database.
    
    Args:
        config_path (str or Path): Path to the parameters.yaml file.
        
    Returns:
        Chroma: The persisted Chroma vector database instance.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        print(f"Looking for file at: {config_path.absolute()}")
        raise FileNotFoundError("Config file not found")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    chunked_data_path = Path(config['pipeline']['chunked_data_path'])
    vector_db_path = Path(config['pipeline']['vector_db_path'])

    # Ensure vector DB directory exists
    vector_db_path.mkdir(parents=True, exist_ok=True)

    # Load JSON → Documents
    documents = []
    json_files = list(chunked_data_path.glob("*.json"))

    print(f"Total JSON files: {len(json_files)}")
    for f in json_files:
        print(f.name)

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            documents.append(
                Document(
                    page_content=item["text"],
                    metadata=item["metadata"]
                )
            )

    print(f"Loaded {len(documents)} documents")

    # Initialize Embedding Model (LOCAL)
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1024
    )

    # Create / Persist Chroma DB
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=str(vector_db_path),
        collection_name="tesla_financials"
    )

    # Persist DB
    vectordb.persist()

    print("Chroma DB created and persisted successfully.")
    return vectordb

if __name__ == "__main__":
    config_file_path = "/Users/lokeshkv/data-engineering/Tesla_Financial_Document_Q_and_A_System_using_RAG/config/parameters.yaml"
    vectordb = create_vector_db(config_file_path)