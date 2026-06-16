from __future__ import annotations

"""
GRPO on a standalone ``AutoModelForCausalLM`` (optional PEFT). Checkpoints under
``output_dir`` are Hugging Face adapter or full-text saves; inject them into
``ProteinLLMModel`` with ``python -m bioreason2.utils.save_grpo_ckpt`` (use
``--checkpoint_path`` = ``final/`` or ``checkpoint-*``; layout is auto-detected).
"""

import json
import os
import random
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Match ``train_protein_llm.py`` before importing torch/transformers/datasets.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_CUMEM_ENABLE"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging

from bioreason2.dataset.prompts.cafa5 import (
    CAFA5_REASONING_TEMPLATE,
    CAFA5_REASONING_TEMPLATE_WITH_CONTEXT,
    CAFA5_REASONING_TEMPLATE_WITH_CONTEXT_PPI,
)
from bioreason2.utils import str2bool
from bioreason2.utils.go_reward import (
    GeneOntology,
    estimate_ia_from_annotations,
    GO_SUMMARY_END,
    GO_SUMMARY_START,
    load_ia_weights,
    resolve_gold_terms_from_row,
    reward_from_text,
)

torch.multiprocessing.set_sharing_strategy("file_system")
hf_logging.set_verbosity_error()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class RolloutExample:
    row: dict
    prompt: str
    gold_leaf_terms: set[str]


@dataclass
class GrpoPromptConfig:
    """Mirrors bioreason2.dataset.cafa5.load._format_reasoning_prompt (text-only)."""

    go_gpt_predictions_column: str | None
    interpro_in_prompt: bool
    ppi_in_prompt: bool
    ask_all_go_aspects: bool
    append_uniprot_suffix: bool
    go_summary_tags_in_prompt: bool


_GO_SUMMARY_TAG_INSTRUCTION = (
    f" When you summarize your conclusions, list every Gene Ontology identifier you endorse "
    f"between {GO_SUMMARY_START} and {GO_SUMMARY_END} "
    f"(for example: {GO_SUMMARY_START} GO:0005515 GO:0006915 {GO_SUMMARY_END})."
)


def maybe_get(row: dict, key: str, default: str = "") -> str:
    value = row.get(key, default)
    if value is None:
        return default
    if isinstance(value, list):
        return "; ".join(map(str, value))
    return str(value)


def _go_aspects_suffix(row: dict, ask_all_go_aspects: bool) -> str:
    if ask_all_go_aspects:
        return (
            " and focus more on its Molecular Function, Biological Process, and Cellular Component."
        )
    go_aspects: List[str] = []
    if row.get("go_mf"):
        go_aspects.append("Molecular Function")
    if row.get("go_cc"):
        go_aspects.append("Cellular Component")
    if row.get("go_bp"):
        go_aspects.append("Biological Process")
    if go_aspects:
        return f" and focus more on its {', '.join(go_aspects)}."
    return "."


def build_prompt(row: dict, cfg: GrpoPromptConfig) -> str:
    """
    Text-only prompt aligned with bioreason2.dataset.cafa5.load._format_reasoning_prompt.
    Embeds the same system/user strings as reasoning SFT; no multimodal placeholders.
    """
    organism = maybe_get(row, "organism") or "Unknown"

    interpro_data = ""
    if cfg.interpro_in_prompt:
        interpro_data = (maybe_get(row, "interpro_formatted") or maybe_get(row, "interpro")).strip()

    ppi_data = ""
    if cfg.ppi_in_prompt:
        ppi_data = (maybe_get(row, "ppi_formatted") or maybe_get(row, "ppi")).strip()

    go_speculations = ""
    if cfg.go_gpt_predictions_column:
        go_speculations = maybe_get(row, cfg.go_gpt_predictions_column).strip()

    go_aspects_suffix = _go_aspects_suffix(row, cfg.ask_all_go_aspects)
    uniprot_summary = " Summarize in UniProt format."

    if cfg.ppi_in_prompt and (interpro_data or go_speculations):
        system_prompt = CAFA5_REASONING_TEMPLATE_WITH_CONTEXT_PPI["system_prompt"]
        user_prompt = CAFA5_REASONING_TEMPLATE_WITH_CONTEXT_PPI["user_prompt"].format(
            organism=organism,
            interpro_data=interpro_data if interpro_data else "None",
            ppi_data=ppi_data if ppi_data else "None",
            go_speculations=go_speculations if go_speculations else "None",
            go_aspects_suffix=go_aspects_suffix,
        )
    elif interpro_data or go_speculations:
        system_prompt = CAFA5_REASONING_TEMPLATE_WITH_CONTEXT["system_prompt"]
        user_prompt = CAFA5_REASONING_TEMPLATE_WITH_CONTEXT["user_prompt"].format(
            organism=organism,
            interpro_data=interpro_data if interpro_data else "",
            go_speculations=go_speculations if go_speculations else "",
        )
    else:
        system_prompt = CAFA5_REASONING_TEMPLATE["system_prompt"]
        user_prompt = CAFA5_REASONING_TEMPLATE["user_prompt"].format(organism=organism)

    user_prompt = user_prompt.rstrip(".") + go_aspects_suffix
    if cfg.append_uniprot_suffix:
        user_prompt += uniprot_summary

    if cfg.go_summary_tags_in_prompt:
        system_prompt = system_prompt.rstrip() + _GO_SUMMARY_TAG_INSTRUCTION

    sequence = maybe_get(row, "sequence").strip()
    protein_name = maybe_get(row, "protein_name").strip()
    header_lines: List[str] = []
    if protein_name:
        header_lines.append(f"Protein name: {protein_name}")
    header_lines.append(f"Organism: {organism}")
    header_lines.append(f"Sequence: {sequence}")
    protein_block = "\n".join(header_lines)

    combined = (
        f"{system_prompt.strip()}\n\n{protein_block}\n\n{user_prompt.strip()}"
    )
    return combined.strip()


def left_pad_to_max(seqs: Sequence[torch.Tensor], pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(x.shape[0] for x in seqs)
    input_ids = []
    attention_mask = []

    for seq in seqs:
        pad_len = max_len - seq.shape[0]
        padded = torch.full((max_len,), pad_token_id, dtype=torch.long)
        mask = torch.zeros((max_len,), dtype=torch.long)
        padded[pad_len:] = seq
        mask[pad_len:] = 1
        input_ids.append(padded)
        attention_mask.append(mask)

    return torch.stack(input_ids, dim=0), torch.stack(attention_mask, dim=0)


def sequence_logprob_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    prompt_lengths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    logits: [B, T, V]
    labels: [B, T]
    prompt_lengths: [B]
    Returns:
      seq_logp [B]: sum of per-token log-probs over completion tokens.
      token_logp [B, T-1]: per-token log-prob (no completion mask applied).
      completion_mask [B, T-1]: 1 where the position belongs to the completion.
    """
    # gather + logsumexp avoids materializing the full F.log_softmax output [B, T-1, V]
    # (the original code OOMed at T=768 for Qwen3 vocab ~150k).
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    target_logits = shift_logits.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    lse = torch.logsumexp(shift_logits, dim=-1)
    token_logp = target_logits - lse  # [B, T-1]

    B, Tm1 = token_logp.shape
    positions = torch.arange(Tm1, device=token_logp.device).unsqueeze(0).expand(B, -1)
    # label position j corresponds to original token index j+1
    # completion starts at token index >= prompt_length
    completion_mask = ((positions + 1) >= prompt_lengths.unsqueeze(1)).to(token_logp.dtype)
    seq_logp = (token_logp * completion_mask).sum(dim=1)
    return seq_logp, token_logp, completion_mask


def compute_batch_logprobs(
    model,
    tokenizer,
    prompts: Sequence[str],
    completions: Sequence[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences = []
    prompt_lengths = []

    for prompt, completion in zip(prompts, completions):
        prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
        full_ids = tokenizer(prompt + completion, add_special_tokens=False, return_tensors="pt").input_ids[0]
        sequences.append(full_ids)
        prompt_lengths.append(prompt_ids.shape[0])

    input_ids, attention_mask = left_pad_to_max(sequences, tokenizer.pad_token_id)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    prompt_lengths_t = torch.tensor(prompt_lengths, dtype=torch.long, device=device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return sequence_logprob_from_logits(outputs.logits, input_ids, prompt_lengths_t)


def load_policy_model(
    model_name_or_path: str,
    checkpoint_path: str | None,
    dtype: torch.dtype,
    device_map: str | None = None,
    *,
    peft_trainable: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device_map,
    )

    if checkpoint_path:
        # Assumes checkpoint_path is a PEFT adapter directory.
        model = PeftModel.from_pretrained(
            model, checkpoint_path, is_trainable=peft_trainable
        )

    return model, tokenizer


def load_reference_model(
    model_name_or_path: str,
    checkpoint_path: str | None,
    dtype: torch.dtype,
    device_map: str | None = None,
):
    """
    Frozen copy of the initial policy for KL, without ``deepcopy`` (avoids PEFT / buffer
    fragility and duplicate graph traversal). Loads the same weights from disk again.
    """
    ref, _ = load_policy_model(
        model_name_or_path,
        checkpoint_path,
        dtype,
        device_map,
        peft_trainable=False,
    )
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    return ref


def sample_group_completions(
    model,
    tokenizer,
    prompts: Sequence[str],
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> Tuple[List[str], List[int]]:
    """
    Returns flattened completions and example indices.
    """
    all_completions: List[str] = []
    owner_index: List[int] = []

    model.eval()
    with torch.no_grad():
        for example_idx, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            # Batch the group via num_return_sequences: shares the prefill and
            # runs the per-token decode loop once instead of group_size times.
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=group_size,
                use_cache=True,  # config.use_cache=False (for gc); re-enable here.
            )
            prompt_len = inputs["input_ids"].shape[1]
            for i in range(group_size):
                gen_tokens = out[i][prompt_len:]
                completion = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                all_completions.append(completion)
                owner_index.append(example_idx)

    return all_completions, owner_index


def build_examples(
    dataset_split,
    prompt_cfg: GrpoPromptConfig,
    limit: int | None = None,
) -> List[RolloutExample]:
    examples: List[RolloutExample] = []
    for i, row in enumerate(dataset_split):
        if limit is not None and i >= limit:
            break
        gold = resolve_gold_terms_from_row(row)
        if not gold:
            continue
        examples.append(
            RolloutExample(
                row=row,
                prompt=build_prompt(row, prompt_cfg),
                gold_leaf_terms=gold,
            )
        )
    return examples


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="wanglab/bioreason-pro-sft-reasoning-data")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--limit_examples", type=int, default=None)

    parser.add_argument(
        "--go_gpt_predictions_column",
        type=str,
        default="go_pred",
        help="Column for initial GO hypotheses (matches SFT go_gpt_predictions_column). Empty string disables.",
    )
    parser.add_argument(
        "--interpro_in_prompt",
        type=str2bool,
        default=True,
        help="Include InterPro text in the user prompt when present (interpro_formatted or interpro).",
    )
    parser.add_argument(
        "--ppi_in_prompt",
        type=str2bool,
        default=False,
        help="Include PPI in prompt; uses CAFA5_REASONING_TEMPLATE_WITH_CONTEXT_PPI when PPI mode is on.",
    )
    parser.add_argument(
        "--ask_all_go_aspects",
        type=str2bool,
        default=True,
        help="If true, suffix asks for MF, BP, and CC together (like predict.py).",
    )
    parser.add_argument(
        "--append_uniprot_suffix",
        type=str2bool,
        default=False,
        help='Append "Summarize in UniProt format." to the user prompt (some SFT configs use this).',
    )
    parser.add_argument(
        "--go_summary_tags_in_prompt",
        type=str2bool,
        default=True,
        help=f"Instruct the model to wrap endorsed GO IDs in {GO_SUMMARY_START}…{GO_SUMMARY_END}.",
    )
    parser.add_argument(
        "--reward_extraction",
        type=str,
        default="sft_aligned",
        choices=("sft_aligned", "final_answer", "full"),
        help="How to collect predicted GO IDs from completions (see bioreason2.utils.go_reward).",
    )

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--sft_adapter_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--go_obo_path", type=str, required=True)
    parser.add_argument("--ia_weights_path", type=str, default=None)
    parser.add_argument("--estimate_ia_from_data", action="store_true")

    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)

    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--epsilon_low", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=0.28)
    parser.add_argument("--kl_beta", type=float, default=1e-4)
    parser.add_argument("--adv_epsilon", type=float, default=1e-6)

    parser.add_argument("--save_every_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", type=str2bool, default=False,
                        help="Skip training; load --model_name_or_path (an RL ckpt dir) and run greedy generation on the eval split.")
    parser.add_argument("--eval_split", type=str, choices=["", "id", "ood"], default="",
                        help="With --eval_only: 'id' uses id-test.csv from the dataset dir, 'ood' uses ood-test.csv.")
    parser.add_argument("--eval_max_new_tokens", type=int, default=512)
    parser.add_argument("--eval_temperature", type=float, default=0.0,
                        help="0.0 = greedy decoding for eval.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # Resume: if output_dir already has a complete checkpoint-*, load it as starting weights.
    # Saves model only (no optimizer state), so AdamW momentum is reset on resume — acceptable
    # for finishing a 2-day partial run after timeout.
    def _latest_valid_ckpt(out_dir: Path):
        cks = [p for p in out_dir.glob("checkpoint-*") if p.is_dir()]
        cks.sort(key=lambda p: int(p.name.rsplit("-", 1)[-1]) if p.name.rsplit("-", 1)[-1].isdigit() else -1)
        for p in reversed(cks):
            has_cfg = (p / "config.json").is_file()
            has_w = (
                (p / "model.safetensors").is_file()
                or (p / "pytorch_model.bin").is_file()
                or (p / "model.safetensors.index.json").is_file()
            )
            if has_cfg and has_w:
                return p
        return None

    resume_ckpt = _latest_valid_ckpt(Path(args.output_dir))
    if resume_ckpt is not None:
        print(f"[resume] Found existing checkpoint: {resume_ckpt}; using it as initial weights.")
        args.model_name_or_path = str(resume_ckpt)
        args.sft_adapter_path = ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("Loading ontology...")
    ontology = GeneOntology.from_obo(args.go_obo_path)

    print("Loading dataset...")
    # When dataset_name points at a local folder containing train.csv (e.g. the
    # bioreason protein dir with id-test.csv / ood-test.csv next to train.csv),
    # load_dataset(folder) recursively scans the tree and trips on
    # checkpoints/*/train_metrics.jsonl (matches the "train" pattern as JSON).
    # Bypass the auto-pattern by passing data_files explicitly.
    _eval_split = (getattr(args, "eval_split", "") or "").lower()
    _local_train_csv = None
    _eval_csv = None
    if isinstance(args.dataset_name, str) and os.path.isdir(args.dataset_name):
        _candidate = os.path.join(args.dataset_name, "train.csv")
        if os.path.isfile(_candidate):
            _local_train_csv = _candidate
        if _eval_split == "id":
            _eval_csv = os.path.join(args.dataset_name, "id-test.csv")
        elif _eval_split == "ood":
            _eval_csv = os.path.join(args.dataset_name, "ood-test.csv")
    if args.eval_only and _eval_csv:
        if not os.path.isfile(_eval_csv):
            raise FileNotFoundError(f"--eval_split={_eval_split} but missing CSV: {_eval_csv}")
        print(f"[eval_only] Loading {_eval_split} CSV: {_eval_csv}")
        ds = load_dataset("csv", data_files={"eval": _eval_csv}, split="eval")
    elif _local_train_csv is not None:
        print(f"[grpo] Loading local train CSV: {_local_train_csv}")
        ds = load_dataset("csv", data_files={"train": _local_train_csv}, split="train")
    else:
        ds = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    go_col = args.go_gpt_predictions_column.strip() or None
    prompt_cfg = GrpoPromptConfig(
        go_gpt_predictions_column=go_col,
        interpro_in_prompt=args.interpro_in_prompt,
        ppi_in_prompt=args.ppi_in_prompt,
        ask_all_go_aspects=args.ask_all_go_aspects,
        append_uniprot_suffix=args.append_uniprot_suffix,
        go_summary_tags_in_prompt=args.go_summary_tags_in_prompt,
    )
    examples = build_examples(ds, prompt_cfg, limit=args.limit_examples)
    if not examples:
        raise RuntimeError("No training examples with GO labels were found.")

    print(f"Loaded {len(examples)} rollout examples.")

    if args.ia_weights_path:
        print(f"Loading IA weights from {args.ia_weights_path}")
        ia_weights = load_ia_weights(args.ia_weights_path)
    elif args.estimate_ia_from_data:
        print("Estimating IA weights from dataset annotations...")
        rows = [ex.row for ex in examples]
        ia_weights = estimate_ia_from_annotations(rows, ontology)
    else:
        print("No IA weights provided; using propagated unweighted F1 reward.")
        ia_weights = None

    print("Loading policy model...")
    policy_model, tokenizer = load_policy_model(
        model_name_or_path=args.model_name_or_path,
        checkpoint_path=args.sft_adapter_path,
        dtype=dtype,
        device_map=None,
    )
    policy_model.to(device)
    # When the policy is a PEFT adapter (warm-started from SFT), the base
    # weights are frozen, so input embeddings have no grad path back to a
    # leaf tensor. With gradient_checkpointing_enable() this triggers
    # `RuntimeError: element 0 of tensors does not require grad and does not
    # have a grad_fn` at the first loss.backward(). enable_input_require_grads()
    # registers a forward-hook that re-marks embedding outputs as requires_grad,
    # threading gradients through the checkpoint boundaries to the LoRA params.
    if hasattr(policy_model, "enable_input_require_grads"):
        policy_model.enable_input_require_grads()
    # Activation memory dominates after weights+grads+AdamW state for full-FT 4B on 140 GB.
    # Checkpointing recomputes attention/MLP activations during backward, freeing several GB.
    policy_model.gradient_checkpointing_enable()
    policy_model.config.use_cache = False

    # ---------------------------------------------------------------
    # Eval-only path: generate completions on the chosen split and report
    # weighted GO F1 (uses the existing reward_from_text utility, ia_weights).
    # No ref model, no optimizer, no training loop.
    # ---------------------------------------------------------------
    if args.eval_only:
        eval_max_new = int(getattr(args, "eval_max_new_tokens", 512) or 512)
        eval_temp = float(getattr(args, "eval_temperature", 0.0) or 0.0)
        do_sample = eval_temp > 0.0

        policy_model.eval()
        pr_total = rc_total = f1_total = 0.0
        n = 0
        sample_rows = []

        print(f"[eval_only] split={_eval_split or 'id'} N={len(examples)} max_new_tokens={eval_max_new} temperature={eval_temp}")
        for ex_idx, ex in enumerate(examples):
            inputs = tokenizer(ex.prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                gen_kwargs = {
                    "do_sample": do_sample,
                    "max_new_tokens": eval_max_new,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "use_cache": True,
                }
                if do_sample:
                    gen_kwargs["temperature"] = eval_temp
                out = policy_model.generate(**inputs, **gen_kwargs)
            gen_tokens = out[0][inputs["input_ids"].shape[1]:]
            completion = tokenizer.decode(gen_tokens, skip_special_tokens=True)

            breakdown = reward_from_text(
                generated_text=completion,
                gold_leaf_terms=ex.gold_leaf_terms,
                ontology=ontology,
                ia_weights=ia_weights,
                extraction_mode=args.reward_extraction,
            )
            pr_total += breakdown.weighted_precision
            rc_total += breakdown.weighted_recall
            f1_total += breakdown.weighted_f1
            n += 1
            if len(sample_rows) < 8:
                sample_rows.append({
                    "f1": breakdown.weighted_f1,
                    "n_pred_leaf": len(breakdown.predicted_leaf_terms),
                    "n_gold_leaf": len(breakdown.gold_leaf_terms),
                    "completion": completion[:600],
                })
            if ex_idx == 0 or (ex_idx + 1) % 25 == 0:
                running_f1 = f1_total / max(n, 1)
                print(f"[eval_only] {ex_idx+1}/{len(examples)} running_f1={running_f1:.4f}")

        if n == 0:
            print("[eval_only] No examples processed.")
            return

        split_tag = _eval_split or "id"
        metrics = {
            f"eval_{split_tag}/precision": pr_total / n,
            f"eval_{split_tag}/recall": rc_total / n,
            f"eval_{split_tag}/f1": f1_total / n,
            f"eval_{split_tag}/n_examples": n,
        }
        print(f"[eval_only] FINAL split={split_tag} N={n} "
              f"P={metrics[f'eval_{split_tag}/precision']:.4f} "
              f"R={metrics[f'eval_{split_tag}/recall']:.4f} "
              f"F1={metrics[f'eval_{split_tag}/f1']:.4f}")

        eval_metrics_path = Path(args.output_dir) / f"eval_{split_tag}_metrics.json"
        try:
            eval_metrics_path.write_text(json.dumps({**metrics, "samples": sample_rows}, indent=2))
            print(f"[eval_only] wrote {eval_metrics_path}")
        except Exception as e:
            print(f"[eval_only] failed to write metrics file: {e}")

        try:
            import wandb as _wandb
            if _wandb.run is not None:
                _wandb.log(metrics)
        except Exception:
            pass
        return

    # Reference KL anchor on the policy device. Headroom comes from policy
    # gradient checkpointing above (frees the activation tensors that pushed
    # us past 140 GB); CPU ref forward at ~3-5 min/step would blow the
    # 7-day SLURM walltime.
    print("Loading reference model (frozen KL anchor, separate load) on policy device...")
    ref_device = device
    ref_model = load_reference_model(
        args.model_name_or_path,
        args.sft_adapter_path,
        dtype,
        device_map=None,
    ).to(ref_device)

    optimizer = AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    global_step = 0
    metrics_log_path = Path(args.output_dir) / "train_metrics.jsonl"

    for epoch in range(args.epochs):
        random.shuffle(examples)

        for start in tqdm(range(0, len(examples), args.batch_size), desc=f"epoch {epoch}"):
            batch = examples[start:start + args.batch_size]
            prompts = [ex.prompt for ex in batch]

            # Snapshot "old policy" at rollout time by computing logprobs before any update.
            completions, owner = sample_group_completions(
                model=policy_model,
                tokenizer=tokenizer,
                prompts=prompts,
                group_size=args.group_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )

            flat_prompts = [prompts[i] for i in owner]

            rewards = []
            reward_debug = []
            for completion, example_idx in zip(completions, owner):
                breakdown = reward_from_text(
                    generated_text=completion,
                    gold_leaf_terms=batch[example_idx].gold_leaf_terms,
                    ontology=ontology,
                    ia_weights=ia_weights,
                    extraction_mode=args.reward_extraction,
                )
                rewards.append(breakdown.weighted_f1)
                reward_debug.append(breakdown)

            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

            # Group-centered advantages, normalized by batch std.
            advantages = rewards_t.clone()
            for i in range(len(batch)):
                idx = [j for j, owner_idx in enumerate(owner) if owner_idx == i]
                group_rewards = rewards_t[idx]
                group_mean = group_rewards.mean()
                advantages[idx] = group_rewards - group_mean

            advantages = advantages / (rewards_t.std(unbiased=False) + args.adv_epsilon)

            policy_model.train()
            current_logp, current_token_logp, completion_mask = compute_batch_logprobs(
                model=policy_model,
                tokenizer=tokenizer,
                prompts=flat_prompts,
                completions=completions,
                device=device,
            )

            with torch.no_grad():
                old_logp = current_logp.detach().clone()
                # Reference model lives on CPU; run forward there, ship logp back to policy device.
                ref_seq_logp, ref_token_logp, _ = compute_batch_logprobs(
                    model=ref_model,
                    tokenizer=tokenizer,
                    prompts=flat_prompts,
                    completions=completions,
                    device=ref_device,
                )
                ref_logp = ref_seq_logp.to(device)
                ref_token_logp = ref_token_logp.to(device)

            log_ratio = current_logp - old_logp
            ratio = torch.exp(log_ratio)

            clipped_ratio = torch.clamp(
                ratio,
                min=1.0 - args.epsilon_low,
                max=1.0 + args.epsilon_high,
            )

            surrogate_1 = ratio * advantages
            surrogate_2 = clipped_ratio * advantages
            policy_objective = torch.minimum(surrogate_1, surrogate_2)

            # Per-token k3 unbiased KL estimator (Schulman): always >= 0 per token.
            # The previous sequence-level k1 estimator (current_logp - ref_logp) could go
            # arbitrarily negative; combined with advantage=0 (group-mean rewards collapsing
            # to equal values), it drove the optimizer to maximize the gap rather than
            # penalize it, causing mode collapse to 768-token degenerate outputs.
            log_ratio_ref_tok = ((ref_token_logp - current_token_logp) * completion_mask).clamp(-20.0, 20.0)
            kl_per_token = torch.exp(log_ratio_ref_tok) - log_ratio_ref_tok - 1.0
            kl_term = (kl_per_token * completion_mask).sum(dim=1)
            loss = -(policy_objective - args.kl_beta * kl_term).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()

            batch_mean_reward = float(rewards_t.mean().item())
            # Now reports k3 per-token KL summed over completion (always >= 0).
            batch_mean_kl = float(kl_term.detach().mean().item())
            batch_mean_len = sum(len(tokenizer.encode(c, add_special_tokens=False)) for c in completions) / max(1, len(completions))

            record = {
                "step": global_step,
                "epoch": epoch,
                "loss": float(loss.item()),
                "mean_reward": batch_mean_reward,
                "mean_advantage": float(advantages.mean().item()),
                "mean_kl_seq": batch_mean_kl,
                "mean_completion_tokens": float(batch_mean_len),
                "batch_size": len(batch),
                "group_size": args.group_size,
            }

            with metrics_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            if global_step % 10 == 0:
                print(record)

            if global_step > 0 and global_step % args.save_every_steps == 0:
                ckpt_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)

                if hasattr(policy_model, "save_pretrained"):
                    policy_model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

            global_step += 1

    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(policy_model, "save_pretrained"):
        policy_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"Training complete. Final checkpoint saved to {final_dir}")


if __name__ == "__main__":
    main()