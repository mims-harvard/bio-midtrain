from itertools import islice
from typing import Iterable, List, Optional, Union

from datasets import Dataset, IterableDataset, load_dataset

FINEFINEWEB_DATASET = "m-a-p/FineFineWeb"
BIOLOGY_DOMAIN = "biology"
DOCS_PER_FILE = 100_000


def _biology_urls(num_files: int, revision: str = "main") -> List[str]:
    base = (
        f"https://huggingface.co/datasets/{FINEFINEWEB_DATASET}/resolve/"
        f"{revision}/{BIOLOGY_DOMAIN}"
    )
    return [f"{base}/{BIOLOGY_DOMAIN}_{idx:06d}.jsonl" for idx in range(num_files)]


def load_finefineweb_biology(
    num_samples: int,
    skip_samples: int = 5_000,
    extra_files: int = 1,
    streaming: bool = True,
    return_iterable: bool = False,
    revision: str = "main",
    cache_dir: Optional[str] = None,
) -> Union[Dataset, IterableDataset]:
    """
    Load FineFineWeb biology-only corpus quickly.

    Args:
        num_samples: Number of biology samples to load.
        skip_samples: Skip first N rows (e.g., to avoid ppl eval overlap).
        extra_files: Add extra file(s) as safety for short final shards.
        streaming: Use streaming mode for faster startup and lower memory.
        return_iterable: Return IterableDataset instead of materialized Dataset.
        revision: HF revision/branch.
        cache_dir: Optional Hugging Face cache directory.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    if skip_samples < 0:
        raise ValueError("skip_samples must be >= 0")

    total_needed = skip_samples + num_samples
    num_files = max(1, (total_needed + DOCS_PER_FILE - 1) // DOCS_PER_FILE + extra_files)

    data_files = {"train": _biology_urls(num_files=num_files, revision=revision)}

    if streaming:
        ds = load_dataset(
            "json",
            data_files=data_files,
            split="train",
            streaming=True,
            cache_dir=cache_dir,
        )
        sliced = ds.skip(skip_samples).take(num_samples)
        if return_iterable:
            return sliced

        rows = list(sliced)
        if len(rows) < num_samples:
            raise ValueError(
                f"Only loaded {len(rows)} samples, requested {num_samples}. "
                f"Try increasing extra_files (current={extra_files})."
            )
        return Dataset.from_list(rows)

    split = f"train[{skip_samples}:{skip_samples + num_samples}]"
    return load_dataset(
        "json",
        data_files=data_files,
        split=split,
        cache_dir=cache_dir,
    )


def iter_finefineweb_biology_text(
    num_samples: int,
    skip_samples: int = 5_000,
    text_key: str = "text",
    revision: str = "main",
    cache_dir: Optional[str] = None,
) -> Iterable[str]:
    """
    Very light text iterator for training pipelines that only need raw text.
    """
    ds = load_finefineweb_biology(
        num_samples=num_samples,
        skip_samples=skip_samples,
        streaming=True,
        return_iterable=True,
        revision=revision,
        cache_dir=cache_dir,
    )
    return (row[text_key] for row in islice(ds, num_samples))
