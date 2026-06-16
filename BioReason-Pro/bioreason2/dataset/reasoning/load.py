import os
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset


def _first_present(example: Dict[str, Any], candidates: List[str], default: Any = None) -> Any:
    for key in candidates:
        if key in example and example[key] is not None:
            return example[key]
    return default


def _build_chat_prompt(system_text: str, user_text: str, reasoning_text: str, answer_text: str, include_go_graph: bool = False) -> List[Dict[str, Any]]:
    user_content = [{"type": "protein", "text": None}]
    if include_go_graph:
        user_content.append({"type": "go_graph", "text": None})
    user_content.append({"type": "text", "text": f"{system_text}\n\n{user_text}".strip()})
    return [
        {
            "role": "user",
            "content": user_content,
        },
        {
            "role": "assistant",
            "reasoning_content": reasoning_text.strip(),
            "content": [
                {"type": "text", "text": answer_text.strip()},
            ],
        },
    ]


def _normalize_reasoning_example(example: Dict[str, Any], max_length: int, include_go_graph: bool = False) -> Dict[str, Any]:
    # If already in training format, keep as-is with lightweight safety defaults.
    if isinstance(example.get("prompt"), list) and example.get("protein_sequences") is not None:
        normalized = dict(example)
        if "structure_path" not in normalized:
            normalized["structure_path"] = None
        if "answer" not in normalized:
            normalized["answer"] = _first_present(example, ["answer", "final_answer"], default="").strip()
        return normalized

    sequence = _first_present(
        example,
        ["sequence", "protein_sequence", "protein_seq", "seq", "amino_acid_sequence"],
        default="",
    )
    if sequence is None:
        sequence = ""
    sequence = str(sequence).strip()
    if max_length is not None and max_length > 0:
        sequence = sequence[:max_length]

    system_text = _first_present(
        example,
        ["system", "system_prompt", "instruction"],
        default="You are an expert in protein function annotation.",
    )
    user_text = _first_present(
        example,
        ["user", "user_prompt", "input", "question"],
        default=(
            "Analyze this protein and provide a concise functional annotation with molecular evidence."
        ),
    )
    reasoning_text = _first_present(
        example,
        ["reasoning", "assistant_reasoning", "rationale", "cot"],
        default="",
    )
    answer_text = _first_present(
        example,
        ["final_answer", "assistant_answer", "answer", "output", "response"],
        default="",
    )

    if sequence == "":
        # Keep examples valid for downstream collate even if a row is malformed.
        # This mirrors existing behavior that tolerates missing structure paths.
        sequence = "X"

    prompt = _build_chat_prompt(
        system_text=str(system_text),
        user_text=str(user_text),
        reasoning_text=str(reasoning_text),
        answer_text=str(answer_text),
        include_go_graph=include_go_graph,
    )

    normalized = dict(example)
    normalized["prompt"] = prompt
    normalized["protein_sequences"] = [sequence]
    normalized["structure_path"] = _first_present(example, ["structure_path", "pdb_path", "cif_path"], default=None)
    normalized["go_aspect"] = _first_present(example, ["go_aspect"], default="all")
    normalized["answer"] = str(answer_text).strip()
    return normalized


def _process_split(dataset_split, max_length: int, debug: bool, include_go_graph: bool = False):
    if debug and len(dataset_split) > 50:
        dataset_split = dataset_split.select(range(50))

    dataset_split = dataset_split.map(
        lambda row: _normalize_reasoning_example(row, max_length=max_length, include_go_graph=include_go_graph),
        desc="Normalizing reasoning SFT examples",
    )
    return dataset_split


def load_reasoning_sft_dataset(
    dataset: str = "wanglab/bioreason-pro-sft-reasoning-data",
    dataset_name: Optional[str] = None,
    max_length: int = 2048,
    val_split_ratio: float = 0.1,
    seed: int = 23,
    cache_dir: Optional[str] = None,
    debug: bool = False,
    include_go_graph: bool = False,
) -> Tuple[Any, Any, Any]:
    normalized_name = dataset_name
    if isinstance(normalized_name, str) and normalized_name.strip().lower() in {"", "none", "null"}:
        normalized_name = None

    # Local CSV directory support:
    #   /path/to/data/train.csv
    #   /path/to/data/id-test.csv    -> validation
    #   /path/to/data/ood-test.csv   -> test
    # (or validation.csv / test.csv if present)
    if os.path.isdir(dataset):
        train_csv = os.path.join(dataset, "train.csv")
        val_csv = os.path.join(dataset, "validation.csv")
        test_csv = os.path.join(dataset, "test.csv")
        id_test_csv = os.path.join(dataset, "id-test.csv")
        ood_test_csv = os.path.join(dataset, "ood-test.csv")

        data_files = {"train": train_csv}
        if os.path.exists(val_csv):
            data_files["validation"] = val_csv
        elif os.path.exists(id_test_csv):
            data_files["validation"] = id_test_csv

        if os.path.exists(test_csv):
            data_files["test"] = test_csv
        elif os.path.exists(ood_test_csv):
            data_files["test"] = ood_test_csv

        datasets = load_dataset("csv", data_files=data_files, cache_dir=cache_dir)
        normalized_name = None
    else:
        try:
            if normalized_name:
                datasets = load_dataset(dataset, name=normalized_name, cache_dir=cache_dir)
            else:
                datasets = load_dataset(dataset, cache_dir=cache_dir)
        except ValueError as e:
            error_text = str(e)
            # Be forgiving when users accidentally pass the dataset repo ID as config name.
            # If only `default` exists, retry with that config.
            if (
                normalized_name is not None
                and "BuilderConfig" in error_text
                and "not found" in error_text
                and "Available: ['default']" in error_text
            ):
                print(
                    f"Warning: dataset config '{normalized_name}' not found for {dataset}. "
                    "Falling back to config 'default'."
                )
                normalized_name = "default"
                datasets = load_dataset(dataset, name=normalized_name, cache_dir=cache_dir)
            else:
                raise

    if "train" in datasets and ("validation" in datasets or "test" in datasets):
        train_dataset = _process_split(datasets["train"], max_length=max_length, debug=debug, include_go_graph=include_go_graph)
        val_source = datasets["validation"] if "validation" in datasets else datasets["test"]
        test_source = datasets["test"] if "test" in datasets else datasets["validation"]
        val_dataset = _process_split(val_source, max_length=max_length, debug=debug, include_go_graph=include_go_graph)
        test_dataset = _process_split(test_source, max_length=max_length, debug=debug, include_go_graph=include_go_graph)
    else:
        train_source = datasets["train"] if "train" in datasets else datasets[list(datasets.keys())[0]]
        train_source = _process_split(train_source, max_length=max_length, debug=debug, include_go_graph=include_go_graph)

        val_size = max(1, int(len(train_source) * val_split_ratio))
        val_size = min(val_size, max(1, len(train_source) - 1))
        split = train_source.train_test_split(test_size=val_size, seed=seed)
        train_dataset = split["train"]
        val_dataset = split["test"]
        test_dataset = val_dataset

    print("Reasoning SFT dataset loaded successfully:")
    print(f"  - Dataset: {dataset}" + (f" (config: {normalized_name})" if normalized_name else ""))
    print(f"  - Training: {len(train_dataset)} samples")
    print(f"  - Validation: {len(val_dataset)} samples")
    print(f"  - Test: {len(test_dataset)} samples")

    return train_dataset, val_dataset, test_dataset
