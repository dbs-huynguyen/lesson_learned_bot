import json
from pathlib import Path
import re
from typing import Any, Union
import unicodedata
from datetime import datetime

from docx2python import docx2python
from docx2python.iterators import enum_at_depth
from langchain_core.documents import Document


VIETNAMESE_MAP = {"đ": "d", "Đ": "D"}
REGEX_HEADING_1 = r"(?:^)[IVXLCDM]+[\)\.\/]"
REGEX_HEADING_2 = r"(?:^)\d+[\)\.\/]"
REGEX_HEADING_3 = r"(?:^)[a-zA-Z]+[\)\.\/]"
REGEX_CROSSED_CHECKBOX = r"(?:^|\s)[\u2612][\s\w]*"
REGEX_ANCHOR = r'<a[^>]*href="(.*?)"[^>]*>(.*?)</a>'


class DocxExtractor:

    def __init__(self):
        self._file_path: Path | None = None
        self._data: list = []
        self._ext_data: list = []
        self._properties: dict[str, Any] = {}
        self._tree: dict[str, Any] = {}
        self._documents: list[Document] = []

    # ------------------------------------------------------------------ #
    # Static / class-level helpers                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"__+|\u2612\s*C[oó]|\u2610\s*Kh[oô]ng", "", text)
        text = re.sub(r" / ", "/", text)
        text = re.sub(r"^--+", "- ", text)
        text = re.sub(r"\s\s+", " ", text)
        return text.strip()

    @classmethod
    def is_meaningful(cls, text: str) -> bool:
        cleaned = text.strip()
        if cleaned:
            return True
        if re.search(REGEX_CROSSED_CHECKBOX, cleaned):
            return True
        return False

    @classmethod
    def flatten(cls, data: Union[list, str]) -> list:
        result = []
        if isinstance(data, list):
            for item in data:
                result.extend(cls.flatten(item))
        else:
            result.append(data)
        return result

    @classmethod
    def get_level(cls, text: str) -> int | None:
        text = text.strip()
        if re.search(REGEX_HEADING_1, text):
            return 1
        if re.search(REGEX_HEADING_2, text):
            return 2
        if re.search(REGEX_HEADING_3, text):
            return 3
        return None

    @classmethod
    def to_snake_case(cls, text: str) -> str:
        for viet_char, replacement in VIETNAMESE_MAP.items():
            text = text.replace(viet_char, replacement)
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
        text = re.sub(r"[^a-z0-9]+", "_", text.lower())
        return text.strip("_")

    @classmethod
    def replace_anchor_and_collect(cls, text: str) -> tuple[str, dict[str, str]]:
        urls: dict[str, str] = {}

        def repl(match: re.Match[str]) -> str:
            url = match.group(1)
            content = match.group(2)
            key = f"#{cls.to_snake_case(content)}"
            urls[key] = url
            return key

        new_text = re.sub(REGEX_ANCHOR, repl, text)
        return new_text, urls

    # ------------------------------------------------------------------ #
    # Pipeline steps                                                       #
    # ------------------------------------------------------------------ #

    def extract_docs(self) -> None:
        with docx2python(self._file_path, duplicate_merged_cells=False) as docx_content:
            print(docx_content.header)
            print(docx_content.footer)

            self._properties = docx_content.core_properties
            for key, value in self._properties.items():
                if key in ("lastPrinted", "created", "modified"):
                    if value is None:
                        continue
                    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    self._properties[key] = dt.date().isoformat()
            for _, sections in enum_at_depth(docx_content.body, 2):
                title = sections[0][0].strip()
                if re.search(REGEX_HEADING_1, title):
                    for text in sections[0]:
                        self._data.extend(text.split("\n"))
                    self._ext_data.append(sections[1])

    def build_tree(self) -> None:
        flat = self.flatten(self._data)

        root: dict[str, Any] = {"level": 0, "content": "ROOT", "children": []}
        stack: list[tuple[int, dict[str, Any]]] = [(0, root)]

        for item in flat:
            text = self.clean_text(item)
            level = self.get_level(text)

            if not self.is_meaningful(text):
                continue

            if level is None:
                parent = stack[-1][1]
                node: dict[str, Any] = {"level": None, "content": text, "children": []}
                parent["children"].append(node)
                continue

            node = {"level": level, "content": text, "children": []}

            while stack and stack[-1][0] >= level:
                stack.pop()

            parent = stack[-1][1]
            parent["children"].append(node)
            stack.append((level, node))

        self._tree = self._merge_node(root, self._clean_metadata(self._ext_data))

    def build_documents(self) -> None:
        for section in self._tree["children"]:
            text, urls = self._build_text(section)
            text += self._format_ext(section.get("ext_data", []))
            if text:
                self._documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": self._file_path.stem,
                            "reference": self._file_path.absolute().as_uri(),
                            "section": section["content"],
                            "doc_type": "lesson-learned",
                            "created_date": self._properties.get("modified"),
                            "urls": urls,
                        },
                    )
                )

    def run(self, file_path: Path) -> list[Document]:
        self._file_path = file_path
        self.extract_docs()
        self.build_tree()
        self.build_documents()
        return self._documents

    def clear(self) -> None:
        self._file_path: Path | None = None
        self._data: list = []
        self._ext_data: list = []
        self._tree: dict[str, Any] = {}
        self._documents: list[Document] = []
        self._properties: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _clean_metadata(self, metadata: list) -> list[list[str]]:
        tables: list[list[str]] = [[] for _ in range(len(metadata))]
        for (i, _), item in enum_at_depth(metadata, 2):
            item = self.clean_text(item)
            if not self.is_meaningful(item):
                continue
            tables[i].append(item)
        return tables

    def _merge_node(
        self, node_1: dict[str, Any], node_2: list[list[str]]
    ) -> dict[str, Any]:
        """Merge node_2 into node_1 based on the order of children in node_1."""
        for idx, node in enumerate(node_1["children"]):
            ext_data: list = []
            if len(node["children"]) > 0:
                ext_data = node_2[idx]
            node["ext_data"] = ext_data
        return node_1

    def _build_text(self, node: dict[str, Any]) -> tuple[str, dict[str, str]]:
        if not node.get("children", []) and node.get("level") is not None:
            return "", {}

        lines: list[str] = []
        urls: dict[str, str] = {}

        content = node["content"]
        content, local_urls = self.replace_anchor_and_collect(content)
        urls.update(local_urls)

        if content != "ROOT":
            lines.append(content)

        for child in node.get("children", []):
            child_text, child_urls = self._build_text(child)
            if child_text:
                lines.append(child_text)
            urls.update(child_urls)

        return "\n".join(lines), urls

    @staticmethod
    def _format_ext(ext_data: list) -> str:
        if not ext_data:
            return ""
        if len(ext_data) == 3:
            return f"\n_{ext_data[0]} - {ext_data[1]}: {ext_data[2]}_"
        return ""


if __name__ == "__main__":
    extractor = DocxExtractor()
    docs = extractor.run("./lesson_learned_05.docx")
    for doc in docs:
        print(doc.page_content)
        print(doc.metadata)
        print("-" * 50)
