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
    INITIAL_CAUSE = "initial_cause"
    ROOT_CAUSE = "root_cause"
    SOLUTION = "solution"
    LESSON_LEARNED = "lesson_learned"


class ContentTypeRawEnum(Enum):
    INITIAL_CAUSE = "Nội dung khắc phục"
    ROOT_CAUSE = "Xác định nguyên nhân gốc"
    SOLUTION = "Biện pháp khắc phục"
    LESSON_LEARNED = "Bài học kinh nghiệm ngăn ngừa phát sinh vấn đề tương tự"


class RoleEnum(Enum):
    REVIEWER = "reviewer"
    PERFORMER = "performer"
    REPORTER = "reporter"


class RoleRawEnum(Enum):
    REVIEWER = "Người xem xét"
    PERFORMER = "Người thực hiện"
    REPORTER = "Người báo cáo"


@dataclass
class LessonsLearnedRaw:
    title: str | None = None
    content: str | None = None
    content_type: ContentTypeEnum | None = None
    role: RoleEnum | None = None
    owner: str | None = None
    date: str | None = None
    urls: dict[str, tuple[str, str]] | None = None


def to_snake_case(text: str) -> str:
    for viet_char, replacement in {"đ": "d", "Đ": "D"}.items():
        text = text.replace(viet_char, replacement)
    # text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9]+", "_", text.lower())
    return text.strip("_")


def repl(match: re.Match[str], urls: dict[str, tuple[str, str]]) -> str:
    key = f"#{to_snake_case(match.group(2))}"
    urls[key] = (match.group(1), match.group(2))
    return key


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
                    p = unicodedata.normalize("NFKD", p)

                    if re.search(r"^\t+$", p):
                        p = re.sub(r"^\t+$", "", p)
                    elif re.search(r"^--\t", p):
                        p = re.sub(r"^--\t", "* ", p)
                    elif re.search(r"^\t--\t", p):
                        p = re.sub(r"^\t--\t", "  * ", p)
                    elif re.search(r"^\t\t--\t", p):
                        p = re.sub(r"^\t\t--\t", "    * ", p)

                    if re.search(r"^\s*\* ", p):
                        texts.append(p)
                        continue

                    p = re.sub(r"\s\s+", " ", p.strip())

                    # Replace numbered or lettered list with markdown headers
                    if re.search(r"^[IVXLCDM]+[\./\)]\s*", p):
                        p = re.sub(
                            r"^[IVXLCDM]+[\./\)]\s*([^:\n]*?)\s*(?::\s*(.*))?$",
                            lambda m: f"## {m.group(1)}:"
                            + (f"\n{m.group(2)}" if m.group(2) else ""),
                            p,
                        )
                    elif re.search(r"^\d[\./\)]\s*", p):
                        p = re.sub(
                            r"^\d[\./\)]\s*([^:\n]*?)\s*(?::\s*(.*))?$",
                            lambda m: f"### {m.group(1)}:"
                            + (f"\n{m.group(2)}" if m.group(2) else ""),
                            p,
                        )
                    elif re.search(r"^[a-zA-Z][\./\)]\s*", p):
                        p = re.sub(
                            r"^[a-zA-Z][\./\)]\s*([^:\n]*?)\s*(?::\s*(.*))?$",
                            lambda m: f"#### {m.group(1)}:"
                            + (f"\n{m.group(2)}" if m.group(2) else ""),
                            p,
                        )

                    # Handle case "Ngày   /"
                    if re.search(r"\s*\d{2}/\s*\d{2}/\s*\d{4}", p):
                        p = re.sub(r"\s*(\d{2})/\s*(\d{2})/\s*(\d{4})", r" \1/\2/\3", p)
                    # Handles case unchecked or checked "Không"
                    elif re.search(r"^\u2610|^\u2612 Không", p):
                        continue
                    # Handles case checked "Có"
                    elif match := re.match(
                        r"^\u2612 Có[\s\n:]*[\(\[]*(.+?)[\)\]]*$", p, re.DOTALL
                    ):
                        if not match:
                            continue
                        p = match.group(1).strip()

                    texts.append(p)

                text = "\n".join(p for p in texts if p)
                if text:
                    flat_row.append(text)
            if flat_row:
                flat_table.append(flat_row)

        return flat_table

    def mapping_data(self, flat_table: list[list[str]]) -> list[LessonsLearnedRaw]:

        tables: list[LessonsLearnedRaw] = []
        for row in flat_table:
            raw_data = {}
            match = re.match(r"^##\s+([^:]+):\s*(.*)$", row[0], re.DOTALL)
            if match:
                raw_data["title"] = match.group(1)

                if re.search(
                    rf"^{ContentTypeRawEnum.INITIAL_CAUSE.value}", raw_data["title"]
                ):
                    raw_data["content_type"] = ContentTypeEnum.INITIAL_CAUSE.value
                elif re.search(
                    rf"^{ContentTypeRawEnum.ROOT_CAUSE.value}", raw_data["title"]
                ):
                    raw_data["content_type"] = ContentTypeEnum.ROOT_CAUSE.value
                elif re.search(
                    rf"^{ContentTypeRawEnum.SOLUTION.value}", raw_data["title"]
                ):
                    raw_data["content_type"] = ContentTypeEnum.SOLUTION.value
                elif re.search(
                    rf"^{ContentTypeRawEnum.LESSON_LEARNED.value}", raw_data["title"]
                ):
                    raw_data["content_type"] = ContentTypeEnum.LESSON_LEARNED.value

                content = match.group(2)

                urls: dict[str, tuple[str, str]] = {}
                content = re.sub(
                    r'<a[^>]*href="(.*?)"[^>]*>(.*?)</a>',
                    lambda m: repl(m, urls),
                    content,
                )

                raw_data["content"] = content
                raw_data["urls"] = urls or None

            if len(row) > 1:
                match = re.match(
                    r"^\S+(\s*\d{0,2}/\s*\d{0,2}/\s*\d{0,4})\n(.+?)(?:\n(.+))?$", row[1]
                )
                if match:
                    raw_data["date"] = match.group(1)
                    raw_data["owner"] = (
                        match.group(3).replace("\n", ", ") if match.group(3) else None
                    )

                    if re.search(rf"{RoleRawEnum.REVIEWER.value}", match.group(2)):
                        raw_data["role"] = RoleEnum.REVIEWER.value
                    elif re.search(rf"{RoleRawEnum.PERFORMER.value}", match.group(2)):
                        raw_data["role"] = RoleEnum.PERFORMER.value
                    elif re.search(rf"{RoleRawEnum.REPORTER.value}", match.group(2)):
                        raw_data["role"] = RoleEnum.REPORTER.value

            if raw_data.get("title") and raw_data.get("content"):
                tables.append(LessonsLearnedRaw(**raw_data))

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
            # print(json.dumps(tables_2d, ensure_ascii=False, indent=2))
            raw_objects = self.mapping_data(tables_2d)

            for item in raw_objects:
                if item.title and item.content:
                    docs.append(
                        Document(
                            page_content=f"## {item.title}\n{item.content}",
                            metadata=dict(
                                content_type=item.content_type,
                                title=item.title,
                                date=item.date,
                                owner=item.owner,
                                role=item.role,
                                urls=item.urls,
                            ),
                        )
                    )

        return docs
