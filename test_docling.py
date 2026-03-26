import pprint

from docling.document_converter import (
    DocumentConverter,
    WordFormatOption,
    AsciiDocFormatOption,
)
from docling.datamodel.base_models import InputFormat


source = "data/BM.10.2.01.BISO - Bao cao HDKP va BHKN-AUTH_20250623.docx"
converter = DocumentConverter()
converter.initialize_pipeline(InputFormat.DOCX)
doc = converter.convert(source)

print(
    doc.document.export_to_text(
        # included_content_layers=["body", "furniture"],
    ),
)

# print(
#     doc.export_to_text(
#         labels=["page_header", "table", "reference"],
#         included_content_layers=["body", "furniture"],
#     )
# )
