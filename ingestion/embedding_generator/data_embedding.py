import yaml
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Union, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

def create_vector_db(config_path: Union[str, Path]) -> Optional[Chroma]:
    """
    Loads JSON chunk files, converts them to documents, embeds them using 
    OpenAIEmbeddings, and persists the database. If no files exist, prints a message and moves on.
    
    Args:
        config_path (str or Path): Path to the parameters.yaml file.
        
    Returns:
        Chroma or None: The persisted Chroma vector database instance, or None if no files are found.
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

    print(f"Total JSON files found: {len(json_files)}")
    for f in json_files:
        print(f"Found: {f.name}")

    # Check: If no files exist, print and return
    if not json_files:
        print(
            f"No JSON files found at {chunked_data_path}. "
            "Skipping Chroma DB creation and moving on."
        )
        return None

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

    print(f"Loaded {len(documents)} documents successfully.")

    # Initialize Embedding Model
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
    # Note: Depending on your Chroma version, persist() might be deprecated in favor of automatic persistence.
    if hasattr(vectordb, 'persist'):
        vectordb.persist()

    print("Chroma DB created and persisted successfully.")
    return vectordb