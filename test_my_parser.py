import pprint

from lib.parser.base import LessonsLearnedParser

parser = LessonsLearnedParser()
documents = parser.parser(
    file_path="data/BM.10.2.01.BISO - Bao cao HDKP va BHKN - 32045.docx"
)

for doc in documents:
    pprint.pprint(doc.page_content, width=100000)
    pprint.pprint(doc.metadata, width=100000)
    print()
