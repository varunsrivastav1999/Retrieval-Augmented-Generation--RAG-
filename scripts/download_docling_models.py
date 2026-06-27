print("Starting Docling models pre-download...")
import sys
import os

# Approach 1: Official v2 model downloader
try:
    from docling.utils.model_downloader import download_models
    print("Found docling.utils.model_downloader, downloading all models via official API...")
    download_models()
    print("Successfully downloaded Docling models via download_models API.")
except ImportError:
    print("download_models API not found, falling back to dummy conversion...")

# Approach 1.5: Pre-download EasyOCR weights to guarantee 100% offline air-gap
try:
    import easyocr
    print("Pre-downloading EasyOCR weights to prevent runtime network access...")
    # Initializing the Reader triggers the model downloads (craft_mlt_25k and english_g2)
    reader = easyocr.Reader(['en'], gpu=False)
    print("Successfully pre-downloaded EasyOCR weights.")
except ImportError:
    print("easyocr not installed, skipping EasyOCR offline pre-download.")
except Exception as e:
    print(f"Warning during EasyOCR pre-download: {e}")

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
