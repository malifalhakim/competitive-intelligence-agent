import argparse
import logging
import time
import json
from pathlib import Path

import pandas as pd
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    TableFormerMode,
    ThreadedPdfPipelineOptions,
    smolvlm_picture_description,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("docling").setLevel(logging.INFO)

_log = logging.getLogger(__name__)

def get_pipeline_options() -> ThreadedPdfPipelineOptions:
    """Configures the PDF pipeline with OCR and table structure options."""
    pipeline_options = ThreadedPdfPipelineOptions(
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.AUTO,
        ),
        ocr_batch_size=2,
        layout_batch_size=16,
        table_batch_size=2,
    )
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(
        mode=TableFormerMode.ACCURATE, do_cell_matching=True
    )
    pipeline_options.ocr_options.lang = ["en"]
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2
    pipeline_options.do_picture_classification = True
    pipeline_options.do_picture_description = True
    pipeline_options.picture_description_options = smolvlm_picture_description
    return pipeline_options

def export_results(conv_result, output_dir: Path):
    """Exports conversion results to JSON format."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        _log.error(f"Failed to create output directory {output_dir}: {e}", exc_info=True)
        return

    doc_filename = conv_result.input.file.stem

    json_path = output_dir / f"{doc_filename}.json"
    try:
        _log.info(f"Saving JSON output to {json_path}")
        with json_path.open("w", encoding="utf-8") as fp:
            doc_dict = conv_result.document.export_to_dict()
            json.dump(doc_dict, fp, ensure_ascii=False)
        _log.info(f"Successfully saved {json_path}")
    except Exception as e:
        _log.error(f"Failed to export JSON for {doc_filename}: {e}", exc_info=True)

def main(input_folder: Path, output_base_dir: Path):
    _log.setLevel(logging.INFO)
    if not input_folder.is_dir():
        _log.error(f"Input folder not found or is not a directory: {input_folder}")
        return

    pdf_files = list(input_folder.glob("*.pdf"))
    if not pdf_files:
        _log.info(f"No PDF files found in {input_folder}")
        return

    try:
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=get_pipeline_options())
            }
        )

        start_time = time.time()
        doc_converter.initialize_pipeline(InputFormat.PDF)
        _log.info(f"Pipeline initialized in {time.time() - start_time:.2f} seconds for {len(pdf_files)} files.")
    except Exception as e:
        _log.error(f"Failed to initialize Docling pipeline: {e}", exc_info=True)
        return

    for pdf_path in pdf_files:
        _log.info(f"Processing: {pdf_path.name}")
        start_time = time.time()
        
        try:
            conv_result = doc_converter.convert(pdf_path)
            pipeline_runtime = time.time() - start_time
            
            if conv_result.status != ConversionStatus.SUCCESS:
                _log.error(f"Conversion failed for {pdf_path.name} with status: {conv_result.status}")
                continue

            _log.info(f"Document {pdf_path.name} converted in {pipeline_runtime:.2f} seconds.")
            
            try:
                pages_count = len(conv_result.pages)
                if pipeline_runtime > 0:
                    _log.info(f"Rate: {pages_count / pipeline_runtime:.2f} pages/second.")
            except Exception:
                pass

            export_results(conv_result, output_base_dir)
            
        except Exception as e:
            pipeline_runtime = time.time() - start_time
            _log.error(f"Unexpected error processing {pdf_path.name} after {pipeline_runtime:.2f}s: {e}", exc_info=True)
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process PDF documents using Docling.")
    parser.add_argument("input_folder", type=Path, help="Path to the folder containing PDF files.")
    parser.add_argument("--output_dir", type=Path, default=Path("scratch"), help="Base directory for output.")
    
    args = parser.parse_args()
    main(args.input_folder, args.output_dir)