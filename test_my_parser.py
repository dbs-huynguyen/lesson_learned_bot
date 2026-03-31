import json
import pprint

from lib.parser.base import LessonsLearnedParser

parser = LessonsLearnedParser()
documents = parser.parser(
    file_path="data/BM.10.2.01.BISO - Bao cao HDKP va BHKN - 32045.docx"
)

for doc in documents:
    # print("Content:")
    print(f"{doc.page_content}")
    # print()
    # print("Metadata:")
    # print(json.dumps(doc.metadata, ensure_ascii=False, indent=4))
    # print("-" * 80)
