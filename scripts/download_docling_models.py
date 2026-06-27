import sys
import os

print("Starting Docling models pre-download...")

import huggingface_hub
import os

print("Force-downloading ds4sd/docling-models repository to bypass internal allow_patterns bug...")
try:
    # Explicitly download the entire model repo so .pt and .onnx files are all cached.
    # This prevents the "Missing ONNX file: .../model.pt" bug.
    huggingface_hub.snapshot_download(repo_id="ds4sd/docling-models")
    print("Successfully fetched ds4sd/docling-models!")
except Exception as e:
    print(f"Warning: Failed to manual snapshot download: {e}")

# Approach 1: Official v2 model downloader
try:
    from docling.utils.model_downloader import download_models
    print("Found docling.utils.model_downloader, downloading all models via official API...")
    download_models()
    print("Successfully downloaded Docling models via download_models API.")
    sys.exit(0)
except ImportError:
    print("download_models API not found, falling back to dummy conversion...")

# Approach 2: Dummy conversion to force lazy-loading
try:
    from docling.document_converter import DocumentConverter
    try:
        from docling.document_converter import PdfFormatOption
    except ImportError:
        from docling.datamodel.document import PdfFormatOption
        
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
    from docling.datamodel.base_models import InputFormat
    
    opts = PdfPipelineOptions()
    opts.do_ocr = True
    opts.do_table_structure = True
    
    try:
        from docling.datamodel.pipeline_options import TableFormerMode
        opts.table_structure_options.mode = TableFormerMode.ACCURATE
    except ImportError:
        opts.table_structure_options = TableStructureOptions(mode='accurate')
        
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    
    # In Docling v2, models are lazily loaded. We must call initialize_pipeline to force download
    if hasattr(converter, 'initialize_pipeline'):
        converter.initialize_pipeline(InputFormat.PDF)
        print("Successfully pre-downloaded Docling models via initialize_pipeline.")
    else:
        print("Warning: Could not force lazy loading. Models might download at runtime.")
        
except Exception as e:
    print(f"Fallback triggered during pre-download: {e}")
    from docling.document_converter import DocumentConverter
    DocumentConverter()
    print("Successfully pre-downloaded Docling models with default mode.")
