import yaml
import json
import shutil
from typing import Union
from query_engine import run_query_engine
from ingestion.document_loader.extracting_pdf_file import extract_text_from_pdfs
from ingestion.text_cleaning.data_cleaning import clean_text
from ingestion.chunking_strategy.data_chunking import process_and_chunk_documents
from ingestion.embedding_generator.data_embedding import create_vector_db
from pathlib import Path


def archive_processed_files(checkpoint: dict, config: dict) -> None:
    """
    Move files listed in checkpoint to their respective archive folders.
    Archives are created inside each stage folder: raw/archive, extracted/archive, etc.
    """

    # Map checkpoint keys to respective stage directories from config
    stage_map = {
        "raw_pdf_file":      Path(config['pipeline']['raw_pdf_path']),
        "extracted_text_file": Path(config['pipeline']['extracted_text_path']),
        "cleaned_text_file":  Path(config['pipeline']['cleaned_text_path']),
        "chunked_data_file":  Path(config['pipeline']['chunked_data_path']),
    }
    # Note: vector_db_file is a folder not a file — skip archiving it here

    for stage_key, stage_dir in stage_map.items():
        files_to_archive = checkpoint.get(stage_key, [])

        if not files_to_archive:
            continue

        # Create archive subfolder inside the stage directory
        archive_dir = stage_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)

        for filename in files_to_archive:
            source = stage_dir / filename

            if source.exists():
                destination = archive_dir / filename

                # If file already exists in archive — overwrite it
                if destination.exists():
                    destination.unlink()

                shutil.move(str(source), str(destination))
                print(f"  Archived: {stage_key} -> {filename}")
            else:
                print(f"  File not found, skipping: {source}")


def main(config_path: Union[str, Path]):
    print("=" * 60)
    print("Tesla Financial Intelligence System — RAG Pipeline")
    print("=" * 60)

    # Load configuration
    print("\n[Step 0] Loading configuration...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pdf_path            = config['pipeline']['raw_pdf_path']
    extracted_path      = config['pipeline']['extracted_text_path']
    cleaned_path        = config['pipeline']['cleaned_text_path']
    chunked_path        = config['pipeline']['chunked_data_path']
    vector_db_path      = config['pipeline']['vector_db_path']

    # Load or create checkpoint
    print("\n[Step 1] Loading checkpoint...")
    checkpoint_file = Path("check_point.json")

    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            checkpoint = json.load(f)
        print(f"  Checkpoint loaded: {checkpoint_file}")
    else:
        checkpoint = {
            "raw_pdf_file":        [],
            "extracted_text_file": [],
            "cleaned_text_file":   [],
            "chunked_data_file":   [],
            "vector_db_file":      []
        }
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=4)
        print(f"  New checkpoint created: {checkpoint_file}")

    # Archive already processed files
    print("\n[Step 2] Archiving previously processed files...")

    has_files_to_archive = any(
        len(checkpoint.get(key, [])) > 0
        for key in ["raw_pdf_file", "extracted_text_file",
                    "cleaned_text_file", "chunked_data_file"]
    )

    if has_files_to_archive:
        archive_processed_files(checkpoint, config)
        print("  Archiving complete.")
    else:
        print("  No previously processed files found — skipping archive.")

    # Run ingestion pipeline
    print("\n[Step 3] Running ingestion pipeline...")

    # 3a. Extract text from PDFs
    print("  Extracting text from PDFs...")
    extract_text_from_pdfs(config_path)

    # 3b. Clean extracted text
    print("  Cleaning extracted text...")
    clean_text(config_path)

    # 3c. Chunk cleaned text
    print("  Chunking cleaned text...")
    process_and_chunk_documents(config_path)

    # 3d. Generate embeddings and store in vector DB
    print("  Generating embeddings and storing in ChromaDB...")
    create_vector_db(config_path)

    print("  Ingestion pipeline complete.")

    # Update checkpoint with new files
    print("\n[Step 4] Updating checkpoint...")

    new_checkpoint = {
        "raw_pdf_file":        [f.name for f in Path(pdf_path).glob("*.pdf")],
        "extracted_text_file": [f.name for f in Path(extracted_path).glob("*.txt")],
        "cleaned_text_file":   [f.name for f in Path(cleaned_path).glob("*.txt")],
        "chunked_data_file":   [f.name for f in Path(chunked_path).glob("*.json")],
        "vector_db_file":      [vector_db_path]
    }

    with open(checkpoint_file, "w") as f:
        json.dump(new_checkpoint, f, indent=4)

    print(f"  Checkpoint updated: {checkpoint_file}")

    # Run query engine
    print("\n[Step 5] Starting query engine...")
    print("=" * 60)
    run_query_engine(config_path)


if __name__ == "__main__":
    main('config/parameters.yaml')