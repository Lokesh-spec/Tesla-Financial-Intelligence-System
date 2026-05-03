import re
import yaml
from pathlib import Path
from typing import Union

def clean_text(config_path: Union[Path, str]) -> None:
    """
    Cleans extracted text files and saves cleaned versions.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Looking for file at: {config_path.absolute()}")
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    extracted_path = Path(config['pipeline']['extracted_text_path'])
    cleaned_path = Path(config['pipeline']['cleaned_text_path'])

    cleaned_path.mkdir(parents=True, exist_ok=True)

    text_files = list(extracted_path.glob("*.txt"))

    for text_file in text_files:
        with open(text_file, "r", encoding="utf-8") as f:
            content = f.read()

        # --- Cleaning ---
        content = content.replace('\t', ' ')
        content = re.sub(r'\s+', ' ', content) 

        # fix broken words like "develop\ned"
        content = re.sub(r'(\w)-\s+(\w)', r'\1\2', content)

        # normalize newlines (optional: keep structure)
        content = re.sub(r'\n+', '\n', content).strip()

        # Save cleaned file
        output_file = cleaned_path / text_file.name
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)

    print(f"Cleaned {len(text_files)} files successfully.")

