import pprint

from langchain_unstructured import UnstructuredLoader
from unstructured_client.models.shared import Strategy
from unstructured.cleaners.core import *
from unstructured.partition.docx import partition_docx
from unstructured.staging.base import elements_to_json


elements = partition_docx(filename="data/BM.10.2.01.BISO - Bao cao HDKP va BHKN_29753.docx", strategy=Strategy.FAST)
elements_to_json(elements=elements, filename="data/BM.10.2.01.BISO - Bao cao HDKP va BHKN_29753.json")
exit(0)


loader = UnstructuredLoader(
    file_path="data/BM.10.2.01.BISO - Bao cao HDKP va BHKN_29753.docx",
    post_processors=[],
    strategy=Strategy.FAST,
    # chunking options
    chunking_strategy="basic",
    max_characters=1000,
    new_after_n_characters=1000,
    chunk_overlap=0,
    chunk_overlap_all=True,
    multipage_sections=True,
    combine_text_under_n_chars=1000,
)
docs = loader.load()
for d in docs:
    pprint.pprint(f"{d.metadata['category']} === {d.page_content}", width=1000000)
    print("\n\n" + "-" * 80)
exit(0)
