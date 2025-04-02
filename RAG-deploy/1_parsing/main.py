import logging
import json
import os
from pathlib import Path
import warnings
import torch

# Ensure PyTorch uses GPU if available
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA Available:", torch.cuda.is_available())
print("Using Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import ImageRefMode
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.datamodel.settings import settings

warnings.filterwarnings("ignore")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_RESOLUTION_SCALE = 2.0
INPUT_DIR = r"C:\\Users\\LENOVO\\Desktop\\DOCKER-TRY\\1_parsing\\input_pdfs"  # Folder containing PDFs/DOCs
OUTPUT_DIR = r"C:\\Users\\LENOVO\\Desktop\\DOCKER-TRY\\1_parsing\\output_md"   # Folder for Markdown outputs

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_pipeline_options(input_format):
    """Creates dynamic pipeline options based on the input format."""
    if input_format == InputFormat.PDF:
        return PdfFormatOption(
            pipeline_options=PdfPipelineOptions(
                do_table_structure=True,
                generate_page_images=True,
                generate_picture_images=True,
                images_scale=IMAGE_RESOLUTION_SCALE,
            )
        )
    elif input_format == InputFormat.DOCX:
        return WordFormatOption(pipeline_cls=SimplePipeline)
    return None

def initialize_converter():
    """Initializes the document converter with multiformat support."""
    allowed_formats = [InputFormat.PDF, InputFormat.DOCX]
    format_options = {fmt: create_pipeline_options(fmt) for fmt in allowed_formats if create_pipeline_options(fmt)}
    return DocumentConverter(allowed_formats=allowed_formats, format_options=format_options)

def convert_and_save():
    """Converts documents to Markdown and saves the output."""
    input_paths = list(Path(INPUT_DIR).glob("*.pdf")) + list(Path(INPUT_DIR).glob("*.docx"))
    if not input_paths:
        logger.warning("No input files found in the directory.")
        return
    
    converter = initialize_converter()
    results = converter.convert_all(input_paths)
    
    for res in results:
        file_name = res.input.file.stem
        md_path = Path(OUTPUT_DIR) / f"{file_name}.md"
        res.document.save_as_markdown(md_path, image_mode=ImageRefMode.REFERENCED)
        logger.info(f"Markdown saved to {md_path}")

def extract_nodes(md_file_path):
    """Extracts nodes from a markdown file, including image references."""
    output_path = Path(OUTPUT_DIR) / f"{md_file_path.stem}_nodes.json"
    
    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
    except Exception as e:
        logger.error(f"Error reading {md_file_path}: {e}")
        return
    
    nodes, text_block = [], ""
    for line in markdown_content.split('\n'):
        if '![' in line and '(' in line and ')' in line:
            parts = line.split('(')
            image_path = parts[1].split(')')[0] if len(parts) > 1 else None
            node_text = parts[0].split('[')[1].split(']')[0] if '[' in parts[0] else ""
            if text_block.strip():
                nodes.append({"index": len(nodes) + 1, "text": text_block.strip(), "image_path": None})
            nodes.append({"index": len(nodes) + 1, "text": node_text, "image_path": image_path})
            text_block = ""
        else:
            text_block += line + "\n"
    
    if text_block.strip():
        nodes.append({"index": len(nodes) + 1, "text": text_block.strip(), "image_path": None})
    
    with open(output_path, "w") as fp:
        json.dump({"file_name": md_file_path.name, "nodes": nodes}, fp, indent=4)
    logger.info(f"Extracted {len(nodes)} nodes from {md_file_path} to {output_path}")

def main():
    settings.debug.profile_pipeline_timings = True
    convert_and_save()
    for md_file in Path(OUTPUT_DIR).glob("*.md"):
        extract_nodes(md_file)

if __name__ == "__main__":
    main()