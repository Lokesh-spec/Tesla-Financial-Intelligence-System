import yaml
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

config_path = Path("/Users/lokeshkv/data-engineering/Tesla_Financial_Document_Q_and_A_System_using_RAG/config/parameters.yaml")

if not config_path.exists():
    print(f"Looking for file at: {config_path.absolute()}")
    raise FileNotFoundError(f"Configuration file not found")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

cleaned_text_path = Path(config['pipeline']['cleaned_text_path'])
chunked_data_path = Path(config['pipeline']['chunked_data_path'])
chunk_size = config['ingestion']['chunk_size']
chunk_overlap = config['ingestion']['chunk_overlap']

# Ensure output directory exists
chunked_data_path.mkdir(parents=True, exist_ok=True)

# Initialize splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

# Process all cleaned text files
text_files = list(cleaned_text_path.glob("*.txt"))

for text_file in text_files:
    with open(text_file, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = splitter.create_documents(
        [content],
        metadatas=[{"source": text_file.name}]
    )

    # Add chunk_id
    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = f"chunk_{i}"

        chunk_data.append({
            "text": chunk.page_content,
            "metadata": chunk.metadata
        })

    # Save as JSON
    output_file = chunked_data_path / f"{text_file.stem}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, indent=2)

    print(f"Processed {text_file.name} → {len(chunk_data)} chunks")

print(f"Total files processed: {len(text_files)}")