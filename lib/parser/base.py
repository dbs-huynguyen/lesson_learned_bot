import json
from pathlib import Path
import pprint
import re
from typing import Any
import unicodedata
from collections.abc import Callable
from enum import Enum
from dataclasses import dataclass

from docx2python import docx2python
from langchain_core.documents import Document


class TypeEnum(Enum):
    PROBLEM = "Problem Description"
    ROOT_CAUSE = "Root Cause"
    SOLUTION = "Solution"
    LESSON_LEARNED = "Lesson Learned"
    IMPROVEMENT = "Improvement"
    EVALUATION = "Evaluation"
    DATE = "Date"


class RoleEnum(Enum):
    REVIEWER = "Người xem xét"
    PERFORMER = "Người thực hiện"
    REPORTER = "Người báo cáo"


class RoleRawEnum(Enum):
    REVIEWER = "Người xem xét"
    PERFORMER = "Người thực hiện"
    REPORTER = "Người báo cáo"


@dataclass
class LessonsLearned:
    page_content: str | None = None
    type: TypeEnum | None = None
    role: RoleEnum | None = None
    owner: str | None = None
    date: str | None = None
    urls: dict[str, tuple[str, str]] | None = None


VIETNAMESE_MAP = {
    "á": "a",
    "Á": "A",
    "à": "a",
    "À": "A",
    "ả": "a",
    "Ả": "A",
    "ã": "a",
    "Ã": "A",
    "ạ": "a",
    "Ạ": "A",
    "ă": "a",
    "Ă": "A",
    "ắ": "a",
    "Ắ": "A",
    "ằ": "a",
    "Ằ": "A",
    "ẳ": "a",
    "Ẳ": "A",
    "ẵ": "a",
    "Ẵ": "A",
    "ặ": "a",
    "Ặ": "A",
    "â": "a",
    "Â": "A",
    "ấ": "a",
    "Ấ": "A",
    "ầ": "a",
    "Ầ": "A",
    "ậ": "a",
    "Ậ": "A",
    "ẫ": "a",
    "Ẫ": "A",
    "é": "e",
    "É": "E",
    "è": "e",
    "È": "E",
    "ẻ": "e",
    "Ẻ": "E",
    "ẽ": "e",
    "Ẽ": "E",
    "ẹ": "e",
    "Ẹ": "E",
    "ê": "e",
    "Ê": "E",
    "ế": "e",
    "Ế": "E",
    "ề": "e",
    "Ề": "E",
    "ể": "e",
    "Ể": "E",
    "ễ": "e",
    "Ễ": "E",
    "ệ": "e",
    "Ệ": "E",
    "ó": "o",
    "Ó": "O",
    "ò": "o",
    "Ò": "O",
    "ỏ": "o",
    "Ỏ": "O",
    "õ": "o",
    "Õ": "O",
    "ọ": "o",
    "Ọ": "O",
    "ơ": "o",
    "Ơ": "O",
    "ớ": "o",
    "Ớ": "O",
    "ờ": "o",
    "Ờ": "O",
    "ở": "o",
    "Ở": "O",
    "ỡ": "o",
    "Ỡ": "O",
    "ợ": "o",
    "Ợ": "O",
    "ô": "o",
    "Ô": "O",
    "ố": "o",
    "Ố": "O",
    "ồ": "o",
    "Ồ": "O",
    "ổ": "o",
    "Ổ": "O",
    "ỗ": "o",
    "Ỗ": "O",
    "ộ": "o",
    "Ộ": "O",
    "ú": "u",
    "Ú": "U",
    "ù": "u",
    "Ù": "U",
    "ủ": "u",
    "Ủ": "U",
    "ũ": "u",
    "Ũ": "U",
    "ụ": "u",
    "Ụ": "U",
    "ư": "u",
    "Ư": "U",
    "ứ": "u",
    "Ứ": "U",
    "ừ": "u",
    "Ừ": "U",
    "ử": "u",
    "Ử": "U",
    "ữ": "u",
    "Ữ": "U",
    "ự": "u",
    "Ự": "U",
    "í": "i",
    "Í": "I",
    "ì": "i",
    "Ì": "I",
    "ỉ": "i",
    "Ỉ": "I",
    "ĩ": "i",
    "Ĩ": "I",
    "ị": "i",
    "Ị": "I",
    "ý": "y",
    "Ý": "Y",
    "ỳ": "y",
    "Ỳ": "Y",
    "ỷ": "y",
    "Ỷ": "Y",
    "ỹ": "y",
    "Ỹ": "Y",
    "ỵ": "y",
    "Ỵ": "Y",
    "đ": "d",
    "Đ": "D",
}


def to_snake_case(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for viet_char, replacement in VIETNAMESE_MAP.items():
        text = text.replace(viet_char, replacement)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9]+", "_", text.lower())
    return text.strip("_")


def handle_link(match: re.Match[str], urls: dict[str, list[str]]) -> str:
    key = f"#{to_snake_case(match.group(2))}"
    urls[key] = [match.group(1), match.group(2)]
    return key


def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"___+", "", s)
    s = s.lstrip("\n").rstrip()
    s = re.sub(r"^([IVXLCDM\da-z]\s*[\./\)])\s*(.+)", r"\1 \2", s)
    if match := re.match(r"^(\s*)--\s*(.+)", s):
        s = f"{match.group(1)}* {match.group(2)}"
    else:
        s = s.lstrip("\t").lstrip(" ")
        if match := re.match(r"^\u2610", s):
            s = ""
        elif match := re.match(r"^\u2612\s*Có\s*(?:(.+))?", s):
            s = match.group(1) or ""
        elif re.search(r"^Ngày", s):
            s = re.sub(r"\s+", " ", s)
    return s

class LessonsLearnedParser:

    def get_table(self, body: list[list[list[list[str]]]], clean_fn: Callable[[str], str] = lambda s: s) -> list[list[list[list[str]]]]:
        if not (
            isinstance(body, list)
            and isinstance(body[0], list)
            and isinstance(body[0][0], list)
            and isinstance(body[0][0][0], list)
            and isinstance(body[0][0][0][0], str)
        ):
            raise ValueError("Input must be a 4D list of strings")

        body[:] = [
            [
                row
                for raw_row in table
                if (
                    row := [
                        [cleaned for s in c if (cleaned := clean_fn(s))]
                        for c in raw_row
                    ]
                )
            ]
            for table in body
            if all(len(row) >= 2 for row in table)
        ]

        return body

    def transform(self, table: list[list[list[str]]], file_path: Path | None = None) -> list[dict[str, Any]]:
        if not (
            isinstance(table[0], list)
            and isinstance(table[0][0], list)
            and isinstance(table[0][0][0], str)
        ):
            raise ValueError("Input must be a 3D list of strings")

        company, project, task = None, None, None
        if match := re.match(
            r"^BM.10.2.01.BISO - Bao cao HDKP va BHKN-(.+[^_]*)_(.+[^_]*)_(.+[^_]*)",
            file_path.stem,
        ):
            company = match.group(1)
            project = match.group(2)
            task = match.group(3)

        # print(json.dumps(table, ensure_ascii=False, indent=4))
        # exit()

        results: list[dict[str, Any]] = []
        for row in table:
            if len(row[0]) == 0:
                continue

            obj = {
                "company": company,
                "project": project,
                "task": task,
            }

            heading1 = row[0].pop(0)
            if re.search(r"^I[\./\)]", heading1):
                obj["type"] = TypeEnum.PROBLEM.value
            elif re.search(r"^II[\./\)]", heading1):
                obj["type"] = TypeEnum.ROOT_CAUSE.value
            elif re.search(r"^III[\./\)]", heading1):
                obj["type"] = TypeEnum.SOLUTION.value
            elif re.search(r"^IV[\./\)]", heading1):
                obj["type"] = TypeEnum.LESSON_LEARNED.value
            elif re.search(r"^V[\./\)]", heading1):
                obj["type"] = TypeEnum.EVALUATION.value

            sentences = []
            urls = {}
            for i, p in enumerate(row[0][::-1]):
                if (not sentences and re.search(r"^\d[\./\)]", p)) or (
                    sentences
                    and re.search(r"^\d[\./\)]", sentences[-1])
                    and re.search(r"^\d[\./\)]", p)
                ):
                    continue

                if re.search(r"^[\da-z][\./\)]", p) and i + 1 < len(row[0]):
                    p = f"\n{p}"

                elif re.search(r"""<a[^>]*.+</a>""", p):
                    p = re.sub(
                        r"""<a[^>]*href=["'](.*?)["'][^>]*>(.*?)</a>""",
                        lambda m: handle_link(m, urls),
                        p,
                    )

                sentences.append(f"{p}  ")

            date, role, owner = None, None, None
            if len(row) == 2 and len(row[1]) > 0:

                if match := re.match(r"^Ngày\s*(\d{2}/\s*\d{2}/\s*\d{4})", row[1][0]):
                    date = match.group(1)

                if re.search(rf"{RoleEnum.REVIEWER.value}", row[1][1]):
                    role = RoleEnum.REVIEWER.name.lower()
                elif re.search(rf"{RoleEnum.PERFORMER.value}", row[1][1]):
                    role = RoleEnum.PERFORMER.name.lower()
                elif re.search(rf"{RoleEnum.REPORTER.value}", row[1][1]):
                    role = RoleEnum.REPORTER.name.lower()

                if len(row[1]) == 3:
                    owner = row[1][2]

            obj["page_content"] = "\n".join(sentences[::-1])
            obj["urls"] = urls or None
            obj["date"] = date
            obj["role"] = role
            obj["redactor"] = owner

            results.append(obj)

        return results

    def parser(
        self,
        file_path: Path,
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
            body_cleaned = self.get_table(docx_content.body, clean_text)
            raw_objs = self.transform(body_cleaned[0], file_path=file_path)
            print(json.dumps(raw_objs, ensure_ascii=False, indent=4))
            exit()

            for obj in raw_objs:
                if obj["page_content"]:
                    docs.append(
                        Document(
                            page_content=f"{obj['page_content']}",
                            metadata=dict(
                                company=obj['company'],
                                project=obj['project'],
                                task=obj['task'],
                                type=obj['type'],
                                urls=obj['urls'],
                                date=obj['date'],
                                role=obj['role'],
                                redactor=obj['redactor'],
                            ),
                        )
                    )

        return docs
