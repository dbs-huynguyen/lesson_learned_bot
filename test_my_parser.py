import json
from pathlib import Path
import pprint

from lib.parser.base import LessonsLearnedParser

parser = LessonsLearnedParser()
documents = parser.parser(
    # file_path=Path("data/BM.10.2.01.BISO - Bao cao HDKP va BHKN-Precision_monshinApp_9999-05.docx")
    # file_path=Path("data/BM.10.2.01.BISO - Bao cao HDKP va BHKN-Precision_monshinApp_9999-04.docx")
    # file_path=Path("data/BM.10.2.01.BISO - Bao cao HDKP va BHKN-Precision_monshinApp_9999-03.docx")
    file_path=Path("data/BM.10.2.01.BISO - Bao cao HDKP va BHKN-CCJ_PASS_20260122-04.docx")
)

for doc in documents:
    # print("Content:")
    print(f"{doc.page_content}")
    print()
    print("Metadata:")
    print(json.dumps(doc.metadata, ensure_ascii=False, indent=4))
    print("-" * 80)
