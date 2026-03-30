from dataclasses import dataclass
import json
import pprint
import re
from typing import Union
import unicodedata
from enum import Enum

from docx2python import docx2python
from langchain_core.documents import Document


class ContentTypeEnum(Enum):
    INITIAL_CAUSE = "Nguyên nhân ban đầu"
    ROOT_CAUSE = "Nguyên nhân gốc"
    SOLUTION = "Giải pháp"
    LESSON_LEARNED = "Bài học"
    IMPROVEMENT_PLAN = "Cải tiến"


class ProcessRoleEnum(Enum):
    REVIEWER = "Người xem xét"
    PERFORMER = "Người thực hiện"
    REPORTER = "Người báo cáo"


@dataclass
class LessonsLearnedRaw:
    title: str | None
    content: str | None
    content_type: ContentTypeEnum | None
    process_role: ProcessRoleEnum | None
    owner: str | None
    date: str | None
    urls: dict[str, tuple[str, str]] | None = None


FIELD_MAP = {
    "Title": "title",
    "Content": "content",
    "Content Type": "content_type",
    "Process Role": "process_role",
    "Owner": "owner",
    "Date": "date",
}


class Utils:

    @staticmethod
    def clean_text(text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = text.replace("\xa0", " ")
        text = re.sub(r"\s\s+", " ", text)
        text = text.replace("\t", " ")
        return text.strip()


class LessonsLearnedParser:

    def get_tables(self, body: list[list[list[list[str]]]]) -> list[list[list[str]]]:

        tables: list[list[list[str]]] = []

        for section in body:
            for block in section:
                if isinstance(block, list) and isinstance(block[0], list):
                    tables.append(block)

        return tables

    def flat_3d_to_2d(self, tables: list[list[list[str]]]) -> list[list[str]]:
        if not (
            isinstance(tables[0], list)
            and isinstance(tables[0][0], list)
            and isinstance(tables[0][0][0], str)
        ):
            raise ValueError("Input must be a 3D list of strings")

        flat_table: list[list[str]] = []

        for row in tables:
            flat_row = []
            for cell in row:
                texts = []
                for p in cell:
                    p = unicodedata.normalize("NFC", p)
                    p = re.sub(r"\t", " ", p)
                    p = re.sub(r"^-- ", "- ", p)
                    p = re.sub(r"^ -- ", " - ", p)
                    p = re.sub(r"^  -- ", "  - ", p)
                    if p.strip():
                        if not re.search(r"^- |^ - |^  - ", p):
                            p = p.strip()
                            if not re.search(r"^Ngày   /", p):
                                p = re.sub(r"\s\s+", " ", p)
                            else:
                                p = re.sub(r"/ (\d{2,4})", r"/\1", p)
                        texts.append(p)
                text = "\n".join(texts)
                if text:
                    flat_row.append(text)
                    # flat_row.append(Utils.clean_text(text))
            if flat_row:
                flat_table.append(flat_row)

        return flat_table

    def mapping_data(self, flat_table: list[list[str]]) -> list[LessonsLearnedRaw]:

        tables: list[LessonsLearnedRaw] = []

        for row in flat_table:
            data = {FIELD_MAP[key]: None for key in FIELD_MAP}

            match = re.match(
                r"^([IVXLCDM]+\.\s*[^:]+):\s*(.*)$", row[0].replace("\n", "\\n")
            )

            if match and len(match.groups()) == 2:
                data["title"] = match.group(1)
                content = match.group(2).replace("\\n", "\n")

                urls: dict[str, str] = {}

                def to_snake_case(text: str) -> str:
                    for viet_char, replacement in {"đ": "d", "Đ": "D"}.items():
                        text = text.replace(viet_char, replacement)
                    text = unicodedata.normalize("NFKD", text)
                    text = "".join(c for c in text if not unicodedata.combining(c))
                    text = re.sub(r"[^a-z0-9]+", "_", text.lower())
                    return text.strip("_")

                def repl(match: re.Match[str]) -> str:
                    key = f"#{to_snake_case(match.group(2))}"
                    urls[key] = (match.group(1), match.group(2))
                    return key

                content = re.sub(r'<a[^>]*href="(.*?)"[^>]*>(.*?)</a>', repl, content)

                data["content"] = content
                data["urls"] = urls

            if len(row) > 1 and row[1]:
                match = re.match(
                    r"^Ngày\s+(\d{2}\/\d{2}\/\d{4})\s+(.+?)\s+([A-ZÀ-Ỹ][A-Za-zÀ-ỹ\s]+)$",
                    row[1],
                )
                if match and len(match.groups()) == 3:
                    data["date"] = match.group(1)
                    data["process_role"] = match.group(2)
                    data["owner"] = match.group(3)

            if data["title"] and data["content"]:
                tables.append(LessonsLearnedRaw(**data))

        for item in tables:
            if re.search(r"^I\.", item.title):
                item.content_type = ContentTypeEnum.INITIAL_CAUSE.value
            elif re.search(r"^II\.", item.title):
                item.content_type = ContentTypeEnum.ROOT_CAUSE.value
            elif re.search(r"^III\.", item.title):
                item.content_type = ContentTypeEnum.SOLUTION.value
            elif re.search(r"^IV\.", item.title):
                item.content_type = ContentTypeEnum.LESSON_LEARNED.value
            else:
                raise ValueError(f"Unknown content type for title: {item.title}")

        return tables

    def parser(
        self,
        file_path: str,
        image_folder: str | None = None,
        *,
        duplicate_merged_cells: bool = False,
    ) -> list[Document]:
        docs: list[Document] = []
        with docx2python(
            file_path,
            image_folder=image_folder,
            html=False,
            duplicate_merged_cells=duplicate_merged_cells,
        ) as docx_content:
            tables_3d = self.get_tables(docx_content.body)
            tables_2d = self.flat_3d_to_2d(tables_3d)
            raw_objects = self.mapping_data(tables_2d)

            for item in raw_objects:
                if item.title and item.content:
                    docs.append(
                        Document(
                            page_content=f"{item.title}{item.content}",
                            metadata=dict(
                                content_type=item.content_type,
                                title=item.title,
                                date=item.date,
                                owner=item.owner,
                                urls=item.urls,
                            ),
                        )
                    )

        return docs
