import os
import yaml
from pathlib import Path
from typing import Union
from langchain_community.document_loaders import PyPDFLoader

def extract_text_from_pdfs(config_path: Union[str, Path]) -> None:
    """
    Extracts text from PDF files based on parameters in the YAML configuration 
    file and saves the text files to the extracted_text_path.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        print(f"Looking for file at: {config_path.absolute()}")
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pdf_path = config['pipeline']['raw_pdf_path']
    extracted_text_path = config['pipeline']['extracted_text_path']

    # Locate and process PDFs
    pdf_files = list(Path(pdf_path).glob("*.pdf"))

    for pdf_file in pdf_files:
        loader = PyPDFLoader(
            file_path=str(pdf_file),
            mode="page"
        )
        pdf_content = loader.load()

        if not pdf_content:
            print(f"Warning: No content extracted from {pdf_file.name}")
            continue

        output_file = Path(extracted_text_path) / f"{pdf_file.stem}.txt"
        
        # Ensure the output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            for i, page in enumerate(pdf_content, start=1):
                f.write(f"--- Page {i} ---\n")
                f.write(page.page_content)
                f.write("\n\n")

    print(f"Successfully processed {len(pdf_files)} PDF files.")


if __name__ == "__main__":
    config_file = Path("/Users/lokeshkv/data-engineering/Tesla_Financial_Document_Q_and_A_System_using_RAG/config/parameters.yaml")
    extract_text_from_pdfs(config_file)