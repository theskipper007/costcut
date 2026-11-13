"""Dataset: HotpotQA, distractor setting, validation split (CC BY-SA 4.0).

Source : https://hotpotqa.github.io/  (Yang et al., EMNLP 2018)
Mirror : https://huggingface.co/datasets/hotpotqa/hotpot_qa  (config "distractor")
File   : data/hotpotqa_distractor_validation_head.jsonl = the first 100 validation rows in
         published order (copied byte-for-byte from the sibling project sabot).

Questions: rows [0, n) in order, no filtering.
Corpus   : EVERY context paragraph of ALL 100 rows (gold + HotpotQA's distractors), one document
           per (row, paragraph), NOT de-duplicated, so each question is retrieved against ~990
           paragraphs. Duplicate paragraphs across rows are kept on purpose: removing them is
           part of the context-budget lever, not of the corpus.
Gold     : a question's supporting titles (HotpotQA supporting_facts), usually 2.
Intent   : HotpotQA's own question `type` (bridge | comparison) is the gold intent label.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "hotpotqa_distractor_validation_head.jsonl"
LICENSE = "CC BY-SA 4.0"
SOURCE = "https://huggingface.co/datasets/hotpotqa/hotpot_qa (distractor, validation)"


@dataclass
class Question:
    idx: int
    id: str
    question: str
    answer: str
    qtype: str
    gold_titles: list[str]


@dataclass
class Doc:
    id: int
    title: str
    text: str

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(f"{self.title}\n{self.text}".encode("utf-8")).hexdigest()


@dataclass
class Data:
    questions: list[Question]
    docs: list[Doc]
    sha256: str
    n_rows_file: int


def load(n: int, path: Path = DATA_FILE) -> Data:
    raw = path.read_bytes()
    rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    if n > len(rows):
        raise ValueError(f"asked for n={n} questions, file has {len(rows)} rows")
    qs = [Question(idx=i, id=r["id"], question=r["question"], answer=r["answer"], qtype=r["type"],
                   gold_titles=list(r["supporting_titles"])) for i, r in enumerate(rows[:n])]
    docs: list[Doc] = []
    for r in rows:
        for title, sents in r["context"]:
            docs.append(Doc(id=len(docs), title=title, text="".join(sents).strip()))
    return Data(questions=qs, docs=docs, sha256=hashlib.sha256(raw).hexdigest(), n_rows_file=len(rows))
