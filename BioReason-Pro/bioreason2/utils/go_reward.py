from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple

import obonet

GO_ID_RE = re.compile(r"\bGO:\d{7}\b")

GO_SUMMARY_START = "<|GO_SUMMARY_START|>"
GO_SUMMARY_END = "<|GO_SUMMARY_END|>"
_ASPECT_BLOCK_DELIMS: Tuple[Tuple[str, str], ...] = (
    ("<|MF_START|>", "<|MF_END|>"),
    ("<|BP_START|>", "<|BP_END|>"),
    ("<|CC_START|>", "<|CC_END|>"),
)


@dataclass
class RewardBreakdown:
    predicted_leaf_terms: Set[str]
    predicted_propagated_terms: Set[str]
    gold_leaf_terms: Set[str]
    gold_propagated_terms: Set[str]
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float


# Hierarchical edges used for CAFA-style term closure (not all obonet edges, e.g. regulates).
_PROPAGATION_EDGE_KEYS = frozenset({"is_a", "part_of"})


class GeneOntology:
    """
    GO hierarchy for reward propagation.

    Loaded with ``obonet.read_obo`` (same OBO intake as ``go_graph_encoder``), so stanza
    parsing and obsolete handling match the rest of BioReason-Pro. Propagation follows
    only ``is_a`` and ``part_of`` edges—the same closure family as before this module used
    a hand-rolled parser; other relationship types stay out of ancestor expansion.
    """

    def __init__(self, parent_map: Dict[str, Set[str]]):
        self.parent_map = parent_map
        self._ancestor_cache: Dict[str, Set[str]] = {}

    @classmethod
    def from_obo(cls, obo_path: str | Path, *, ignore_obsolete: bool = True) -> "GeneOntology":
        path = str(Path(obo_path).expanduser().resolve())
        graph = obonet.read_obo(path, ignore_obsolete=ignore_obsolete)
        parent_map: Dict[str, Set[str]] = defaultdict(set)
        for u, v, key in graph.edges(keys=True):
            if key not in _PROPAGATION_EDGE_KEYS:
                continue
            if not isinstance(u, str) or not u.startswith("GO:"):
                continue
            if not isinstance(v, str) or not v.startswith("GO:"):
                continue
            parent_map[u].add(v)
        return cls(dict(parent_map))

    def ancestors(self, term: str) -> Set[str]:
        if term in self._ancestor_cache:
            return self._ancestor_cache[term]

        seen: Set[str] = set()
        stack = [term]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            for parent in self.parent_map.get(node, ()):
                if parent not in seen:
                    stack.append(parent)

        self._ancestor_cache[term] = seen
        return seen

    def propagate(self, leaf_terms: Iterable[str]) -> Set[str]:
        expanded: Set[str] = set()
        for term in leaf_terms:
            if term.startswith("GO:"):
                expanded.update(self.ancestors(term))
        return expanded


def extract_go_ids(text: str) -> Set[str]:
    return set(GO_ID_RE.findall(text or ""))


def extract_go_ids_from_final_answer(text: str) -> Set[str]:
    """
    Conservative extractor:
    1. Prefer content after 'Final Answer' if present.
    2. Else use entire text.
    """
    if not text:
        return set()

    lowered = text.lower()
    marker_positions = [
        lowered.find("final answer"),
        lowered.find("answer:"),
        lowered.find("final response"),
    ]
    marker_positions = [p for p in marker_positions if p >= 0]
    if marker_positions:
        start = min(marker_positions)
        text = text[start:]
    return extract_go_ids(text)


def _segments_between(text: str, start_token: str, end_token: str) -> List[str]:
    """Return content inside each start…end pair; if end missing, tail from last start."""
    segments: List[str] = []
    i = 0
    while True:
        s = text.find(start_token, i)
        if s < 0:
            break
        s += len(start_token)
        e = text.find(end_token, s)
        if e < 0:
            segments.append(text[s:])
            break
        segments.append(text[s:e])
        i = e + len(end_token)
    return segments


def extract_go_ids_sft_aligned(text: str) -> Set[str]:
    """
    Match BioReason-Pro output structure:
    1. GO IDs inside <|GO_SUMMARY_START|>…<|GO_SUMMARY_END|> (preferred).
    2. Else GO IDs inside MF/BP/CC delimiter blocks.
    3. Else same heuristics as extract_go_ids_from_final_answer.
    """
    if not text:
        return set()

    if GO_SUMMARY_START in text:
        ids: Set[str] = set()
        for seg in _segments_between(text, GO_SUMMARY_START, GO_SUMMARY_END):
            ids.update(extract_go_ids(seg))
        if ids:
            return ids

    aspect_ids: Set[str] = set()
    for start_t, end_t in _ASPECT_BLOCK_DELIMS:
        for seg in _segments_between(text, start_t, end_t):
            aspect_ids.update(extract_go_ids(seg))
    if aspect_ids:
        return aspect_ids

    return extract_go_ids_from_final_answer(text)


def normalize_go_field(value) -> Set[str]:
    """
    Handles common HF dataset encodings:
      - list[str]
      - "GO:0000001; GO:0000002"
      - "['GO:0000001', 'GO:0000002']"
      - None
    """
    if value is None:
        return set()

    if isinstance(value, list):
        return {str(x).strip() for x in value if str(x).strip().startswith("GO:")}

    if isinstance(value, tuple):
        return {str(x).strip() for x in value if str(x).strip().startswith("GO:")}

    if isinstance(value, str):
        matches = GO_ID_RE.findall(value)
        return set(matches)

    return set()


def load_ia_weights(path: str | Path) -> Dict[str, float]:
    path = Path(path)

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): float(v) for k, v in data.items()}

    weights: Dict[str, float] = {}
    with path.open("r", encoding="utf-8") as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        reader = csv.reader(f, dialect=dialect)
        for row in reader:
            if len(row) < 2:
                continue
            go_id = row[0].strip()
            if not go_id.startswith("GO:"):
                continue
            try:
                weights[go_id] = float(row[1])
            except ValueError:
                continue
    return weights


def estimate_ia_from_annotations(
    dataset_rows: Iterable[dict],
    ontology: GeneOntology,
    gold_field_priority: Sequence[str] = (
        "go_ids",
        "go_pred",
        "go_bp",
        "go_mf",
        "go_cc",
        "go_bp_leaf",
        "go_mf_leaf",
        "go_cc_leaf",
    ),
    smoothing: float = 1.0,
) -> Dict[str, float]:
    """
    Approximate IA if official CAFA IA weights are unavailable.
    We use:
        ia(t) = -log( (count(t) + s) / (N + s) )
    over propagated training annotations.
    """
    counts: Counter[str] = Counter()
    n = 0

    for row in dataset_rows:
        leaf_terms: Set[str] = set()
        for field in gold_field_priority:
            if field in row:
                leaf_terms.update(normalize_go_field(row[field]))
        if not leaf_terms:
            continue

        propagated = ontology.propagate(leaf_terms)
        if not propagated:
            continue

        counts.update(propagated)
        n += 1

    if n == 0:
        return {}

    denom = n + smoothing
    ia: Dict[str, float] = {}
    for term, c in counts.items():
        p = (c + smoothing) / denom
        ia[term] = -math.log(p)
    return ia


def weighted_precision_recall_f1(
    predicted_terms: Set[str],
    gold_terms: Set[str],
    ia_weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, float, float]:
    if ia_weights is None:
        ia_weights = {}

    def w(term: str) -> float:
        return float(ia_weights.get(term, 1.0))

    overlap = predicted_terms & gold_terms
    pred_mass = sum(w(t) for t in predicted_terms)
    gold_mass = sum(w(t) for t in gold_terms)
    overlap_mass = sum(w(t) for t in overlap)

    precision = overlap_mass / pred_mass if pred_mass > 0 else 0.0
    recall = overlap_mass / gold_mass if gold_mass > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return precision, recall, f1


def reward_from_text(
    generated_text: str,
    gold_leaf_terms: Iterable[str],
    ontology: GeneOntology,
    ia_weights: Optional[Dict[str, float]] = None,
    extraction_mode: Literal["sft_aligned", "final_answer", "full"] = "sft_aligned",
) -> RewardBreakdown:
    if extraction_mode == "sft_aligned":
        pred_leaf = extract_go_ids_sft_aligned(generated_text)
    elif extraction_mode == "final_answer":
        pred_leaf = extract_go_ids_from_final_answer(generated_text)
    elif extraction_mode == "full":
        pred_leaf = extract_go_ids(generated_text)
    else:
        raise ValueError(
            f"extraction_mode must be 'sft_aligned', 'final_answer', or 'full', got {extraction_mode!r}"
        )
    gold_leaf = {t for t in gold_leaf_terms if t.startswith("GO:")}

    pred_prop = ontology.propagate(pred_leaf)
    gold_prop = ontology.propagate(gold_leaf)

    pr, rc, f1 = weighted_precision_recall_f1(
        predicted_terms=pred_prop,
        gold_terms=gold_prop,
        ia_weights=ia_weights,
    )

    return RewardBreakdown(
        predicted_leaf_terms=pred_leaf,
        predicted_propagated_terms=pred_prop,
        gold_leaf_terms=gold_leaf,
        gold_propagated_terms=gold_prop,
        weighted_precision=pr,
        weighted_recall=rc,
        weighted_f1=f1,
    )


def resolve_gold_terms_from_row(
    row: dict,
    preferred_leaf_fields: Sequence[str] = (
        "go_bp_leaf",
        "go_mf_leaf",
        "go_cc_leaf",
        "go_pred_leaf",
        "go_ids",
        "go_bp",
        "go_mf",
        "go_cc",
    ),
) -> Set[str]:
    gold: Set[str] = set()
    for field in preferred_leaf_fields:
        if field in row:
            gold.update(normalize_go_field(row[field]))
    return gold