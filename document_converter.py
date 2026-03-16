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
        ocr_batch_size=4,
        layout_batch_size=64,
        table_batch_size=4,
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
    """Exports conversion results to various formats including images, CSVs, and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_result.input.file.stem

    # --- Save page images ---
    for page_no, page in conv_result.document.pages.items():
        page_no = page.page_no
        page_image_filename = output_dir / f"{doc_filename}-{page_no}.png"
        with page_image_filename.open("wb") as fp:
            page.image.pil_image.save(fp, format="PNG")

    # --- Save images of figures and tables ---
    table_counter = 0
    picture_counter = 0
    for element, _level in conv_result.document.iterate_items():
        if isinstance(element, TableItem):
            table_counter += 1
            element_image_filename = (
                output_dir / f"{doc_filename}-table-{table_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_result.document).save(fp, "PNG")

        if isinstance(element, PictureItem):
            picture_counter += 1
            element_image_filename = (
                output_dir / f"{doc_filename}-picture-{picture_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_result.document).save(fp, "PNG")
    
    # --- Export tables ---
    for table_ix, table in enumerate(conv_result.document.tables):
        table_df: pd.DataFrame = table.export_to_dataframe(doc=conv_result.document)
        _log.info(f"Processing table {table_ix}...")

        # Save the table as CSV
        element_csv_filename = output_dir / f"{doc_filename}-table-{table_ix + 1}.csv"
        _log.info(f"Saving CSV table to {element_csv_filename}")
        table_df.to_csv(element_csv_filename)

        # Save the table as HTML
        element_html_filename = output_dir / f"{doc_filename}-table-{table_ix + 1}.html"
        _log.info(f"Saving HTML table to {element_html_filename}")
        with element_html_filename.open("w") as fp:
            fp.write(table.export_to_html(doc=conv_result.document))


    # --- Export Docling document JSON format: ---
    with (output_dir / f"{doc_filename}.json").open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(conv_result.document.export_to_dict()))

    # --- Export Text format (plain text via Markdown export): ---
    with (output_dir / f"{doc_filename}.txt").open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_markdown(strict_text=True))

    # --- Export Markdown format: ---
    with (output_dir / f"{doc_filename}.md").open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_markdown())

    # --- Export Document Tags format: ---
    with (output_dir / f"{doc_filename}.doctags").open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_doctags())
    
    # --- Save markdown with embedded pictures ---
    md_filename = output_dir / f"{doc_filename}-with-images.md"
    conv_result.document.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)

    # --- Save markdown with externally referenced pictures ---
    md_filename = output_dir / f"{doc_filename}-with-image-refs.md"
    conv_result.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)

def main(input_folder: Path, output_base_dir: Path):
    _log.setLevel(logging.INFO)
    if not input_folder.is_dir():
        _log.error(f"Input folder not found or is not a directory: {input_folder}")
        return

    pdf_files = list(input_folder.glob("*.pdf"))
    if not pdf_files:
        _log.info(f"No PDF files found in {input_folder}")
        return

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=get_pipeline_options())
        }
    )

    start_time = time.time()
    doc_converter.initialize_pipeline(InputFormat.PDF)
    _log.info(f"Pipeline initialized in {time.time() - start_time:.2f} seconds for {len(pdf_files)} files.")

    for pdf_path in pdf_files:
        _log.info(f"Processing: {pdf_path.name}")
        start_time = time.time()
        conv_result = doc_converter.convert(pdf_path)
        pipeline_runtime = time.time() - start_time
        
        if conv_result.status != ConversionStatus.SUCCESS:
            _log.error(f"Conversion failed for {pdf_path.name} with status: {conv_result.status}")
            continue

        _log.info(f"Document {pdf_path.name} converted in {pipeline_runtime:.2f} seconds.")
        _log.info(f"Rate: {len(conv_result.pages) / pipeline_runtime:.2f} pages/second.")

        doc_output_dir = output_base_dir / pdf_path.stem
        export_results(conv_result, doc_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process PDF documents using Docling.")
    parser.add_argument("input_folder", type=Path, help="Path to the folder containing PDF files.")
    parser.add_argument("--output_dir", type=Path, default=Path("scratch"), help="Base directory for output.")
    
    args = parser.parse_args()
    main(args.input_folder, args.output_dir)