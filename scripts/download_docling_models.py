import sys

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
        
    DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    print("Successfully pre-downloaded Docling models with ACCURATE table mode.")
except Exception as e:
    print(f"Fallback triggered during pre-download: {e}")
    from docling.document_converter import DocumentConverter
    DocumentConverter()
    print("Successfully pre-downloaded Docling models with default mode.")
