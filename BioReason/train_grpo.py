import os

import pathlib
from typing import List, Optional
from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from datasets import load_dataset

from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModel

from trl import ModelConfig, ScriptArguments, TrlParser

from bioreason.models.dna_llm import DNALLMModel, get_target_modules
from bioreason.dna_modules import NucleotideDNAModule
from bioreason.dataset.utils import truncate_dna
from bioreason.dataset.kegg import format_kegg_for_dna_llm
from bioreason.trainer import DNALLMGRPOTrainer, DNALLMGRPOConfig
from bioreason.models.evo2_tokenizer import register_evo2_tokenizer
register_evo2_tokenizer()

# Custom TrainerCallback to override the saving mechanism
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

def _configure_non_home_cache_dirs(cache_root: Optional[str]) -> str:
    """
    Configure cache-related env vars to avoid writing under HOME by default.
    """
    if cache_root and str(cache_root).strip():
        root = os.path.abspath(os.path.expanduser(str(cache_root)))
    elif os.environ.get("BIOREASON_CACHE_ROOT"):
        root = os.path.abspath(os.path.expanduser(os.environ["BIOREASON_CACHE_ROOT"]))
    elif os.environ.get("SCRATCH"):
        root = os.path.join(os.environ["SCRATCH"], "bioreason_cache")
    else:
        root = os.path.join("/tmp", os.environ.get("USER", "bioreason"), "bioreason_cache")

    uv_cache_dir = os.environ.get("UV_CACHE_DIR", os.path.join(root, "uv"))

    paths = {
        "HF_HOME": os.path.join(root, "hf_home"),
        "TRANSFORMERS_CACHE": os.path.join(root, "transformers"),
        "HF_DATASETS_CACHE": os.path.join(root, "datasets"),
        "XDG_CACHE_HOME": os.path.join(root, "xdg"),
        "TORCH_HOME": os.path.join(root, "torch"),
        "TRITON_CACHE_DIR": os.path.join(root, "triton"),
        "WANDB_DIR": os.path.join(root, "wandb"),
        "WANDB_CACHE_DIR": os.path.join(root, "wandb_cache"),
        "UV_CACHE_DIR": uv_cache_dir,
        "PIP_CACHE_DIR": os.path.join(root, "pip"),
        "MPLCONFIGDIR": os.path.join(root, "mpl"),
        "TMPDIR": os.path.join(root, "tmp"),
    }
    paths["HUGGINGFACE_HUB_CACHE"] = os.path.join(paths["HF_HOME"], "hub")

    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    for key, value in paths.items():
        os.environ.setdefault(key, value)

    return root


class SaveWithPyTorchCallback(TrainerCallback):
    """Custom callback to save models with PyTorch's native save mechanism instead of safetensors"""
    def on_save(self, args, state, control, **kwargs):
        # Get the checkpoint folder
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        os.makedirs(checkpoint_folder, exist_ok=True)
        
        # Save with PyTorch instead of safetensors
        checkpoint_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        model = kwargs.get("model")
        
        # Get model unwrapped from accelerator etc.
        unwrapped_model = model.module if hasattr(model, "module") else model
        
        # Save using PyTorch directly
        torch.save(unwrapped_model.state_dict(), checkpoint_path)
        
        # ProteinLLMModel doesn't have a direct config attribute, so we need to save
        # the configs of its sub-models
        if hasattr(unwrapped_model, "text_model"):
            if hasattr(unwrapped_model.text_model, "config"):
                unwrapped_model.text_model.config.save_pretrained(checkpoint_folder)
            # Handle PEFT models which might have base_model
            elif hasattr(unwrapped_model.text_model, "base_model") and hasattr(unwrapped_model.text_model.base_model, "config"):
                unwrapped_model.text_model.base_model.config.save_pretrained(checkpoint_folder)
        
        # Print info about what's being saved
        print(f"Saved model checkpoint to {checkpoint_folder}")
        lora_params = [k for k in unwrapped_model.state_dict().keys() if "lora" in k]
        print(f"Checkpoint contains {len(lora_params)} LoRA parameters")
        
        # Signal that we've saved
        control.should_save = False
        return control

def get_kegg_questions(truncate_dna_per_side: int = 0, kegg_data_dir_local: str = "") -> Dataset:
    data = None
    if kegg_data_dir_local:
        local_root = os.path.abspath(os.path.expanduser(kegg_data_dir_local))
        train_csv = os.path.join(local_root, "train_network_split.csv")
        id_csv = os.path.join(local_root, "id_test_network_split.csv")
        ood_csv = os.path.join(local_root, "ood_test_network_split.csv")
        if all(os.path.exists(p) for p in (train_csv, id_csv, ood_csv)):
            print(f"Using local KEGG CSV dataset from: {local_root}")
            data = load_dataset(
                "csv",
                data_files={
                    "train": train_csv,
                    "val": id_csv,
                    "test": ood_csv,
                },
            )
        else:
            print(f"Local KEGG CSV files not complete under {local_root}; falling back to wanglab/kegg")

    if data is None:
        data = load_dataset("wanglab/kegg", "default")

    # Apply truncation if specified
    if truncate_dna_per_side > 0:
        data = data.map(truncate_dna, fn_kwargs={"truncate_dna_per_side": truncate_dna_per_side})
    
    data = data.map(format_kegg_for_dna_llm, fn_kwargs={"is_sft": False})

    return data

# Format into conversation
@dataclass
class GRPOModelConfig(ModelConfig):
    text_model_name: str = field(default="Qwen/Qwen3-4B", metadata={"help": "Model checkpoint for weights initialization."})
    dna_model_name: str = field(default="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", metadata={"help": "Model checkpoint for weights initialization."})
    cache_dir: str = field(default=None, metadata={"help": "Path to model cache directory."})
    max_length_text: int = field(default=1024, metadata={"help": "Maximum length of text sequences."})
    max_length_dna: int = field(default=1024, metadata={"help": "Maximum length of DNA sequences, in groups of 6 nucleotides."})
    sft_checkpoint: str = field(default=None, metadata={"help": "Path to the checkpoint for SFT."})
    lora_r: int = field(default=16, metadata={"help": "LoRA R value."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0, metadata={"help": "LoRA dropout."})
    lora_modules_to_save: Optional[List[str]] = field(
        default="embed_tokens",
        metadata={"help": "Model layers to unfreeze & train."},
    )
    dna_model_finetune: bool = False
    dna_projection_finetune: bool = True
    peft_ckpt: bool = False
    dna_is_evo2: bool = field(default=False, metadata={"help": "Whether the DNA model is Evo2."})
    dna_embedding_layer: str = field(default=None, metadata={"help": "Evo2 layer name to extract embeddings from (required when dna_is_evo2=True)."})
    truncate_dna_per_side: int = field(default=0, metadata={"help": "Number of base pairs to truncate from each end of the DNA sequence. If 0, no truncation is applied."})
    kegg_data_dir_local: str = field(default="", metadata={"help": "Optional local KEGG/Genomics CSV directory."})
    train_data_fraction: float = field(default=1.0, metadata={"help": "Fraction of train split to keep, in (0,1]. Selected deterministically from the head."})
    eval_only: bool = field(default=False, metadata={"help": "If True, skip trainer.train and only run trainer.evaluate on the requested split."})
    eval_split: str = field(default="", metadata={"help": "When --eval_only: 'id' uses dataset['val'] (id_test_network_split), 'ood' uses dataset['test'] (ood_test_network_split)."})
    eval_ckpt: str = field(default="", metadata={"help": "RL checkpoint dir to load before eval (e.g. .../checkpoint-NNNN). Falls back to training_args.resume_from_checkpoint."})

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    dataset_name: str = field(default="wanglab/kegg", metadata={"help": "Dataset name with default."})
    full_ckpt: str = field(default=None, metadata={"help": "Path to full checkpoint to load"})
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    reward_funcs: List[str] = field(
        #default_factory=lambda: ["accuracy", "format"],
        default_factory=lambda: ["xmlcount", "soft_format", "strict_format", "concise", "correctness"],
        #metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'xmlcount', 'soft_format', 'strict_format', 'concise', 'correctness', 'depth'"},
    )

reward_funcs_registry = {
    "xmlcount": NucleotideDNAModule.xmlcount_reward_func,
    "soft_format": NucleotideDNAModule.soft_format_reward_func,
    "strict_format": NucleotideDNAModule.strict_format_reward_func,
    "concise": NucleotideDNAModule.concise_reward_func,
    "correctness": NucleotideDNAModule.correctness_reward_func,
}

def get_vlm_module(text_model_name):
    if any(mini_name in text_model_name.lower() for mini_name in ["qwen", "smol"]):
        return NucleotideDNAModule
    else:
        raise ValueError(f"Unsupported model: {text_model_name}")

def _prep_for_training(
    model: DNALLMModel,
    training_args,
    dna_model_finetune: bool = False,
    dna_projection_finetune: bool = True
    ) -> Optional[LoraConfig]:
    """
    Load and configure the ProteinLLMModel for training.
    Since ProteinLLMModel starts everything in .eval() mode with frozen parameters,
    we need to systematically enable training for each component based on parameters.
    """    
    # DNA encoder
    # Handle Evo2 wrapper vs HuggingFace models
    if hasattr(model, 'dna_is_evo2') and model.dna_is_evo2:
        # For Evo2, access the internal model
        dna_model = model.dna_model.model
    else:
        # For HF models, access directly
        dna_model = model.dna_model
    
    if dna_model_finetune:
        dna_model.train()
        print("DNA model is training")
        for param in dna_model.parameters():
            param.requires_grad = True
    else:
        dna_model.eval()
        print("DNA model is eval")
        for param in dna_model.parameters():
            param.requires_grad = False
    
    # DNA projection
    if dna_projection_finetune:
        model.dna_projection.train()
        print("DNA projection is training")
        for param in model.dna_projection.parameters():
            param.requires_grad = True
    else:
        model.dna_projection.eval()
        print("DNA projection is eval")
        for param in model.dna_projection.parameters():
            param.requires_grad = False


    # Text model: setup LoRA and set to train mode
    if training_args.lora_r == 0:
        # Text model: full finetune
        model.text_model.train()
        print("Text model is training")
        for param in model.text_model.parameters():
            param.requires_grad = True
        return None
    else:
        # Text model: setup LoRA and set to train mode
        target_modules = get_target_modules(model)

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            target_modules=target_modules,
            init_lora_weights="gaussian",
            bias="none",
            task_type="CAUSAL_LM",
        )

        model.text_model = prepare_model_for_kbit_training(model.text_model)
        model.text_model = get_peft_model(model.text_model, lora_config)
        model.text_model.train()

        return lora_config


def main(script_args, training_args, model_args):
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    # Load model
    model = DNALLMModel(
        text_model_name=model_args.text_model_name,
        dna_model_name=model_args.dna_model_name,
        cache_dir=model_args.cache_dir,
        max_length_text=model_args.max_length_text,
        max_length_dna=model_args.max_length_dna,
        text_model_finetune=True,
        dna_model_finetune=model_args.dna_model_finetune,
        dna_is_evo2=model_args.dna_is_evo2,
        dna_embedding_layer=model_args.dna_embedding_layer,
        device="cuda",
    ).to("cuda")

    model.text_model.config.use_cache = False

    # Load checkpoint
    if model_args.sft_checkpoint is not None:
        training_args.vllm_ckpt = model_args.sft_checkpoint
        print(f"Loading SFT checkpoint from {model_args.sft_checkpoint}")
        
        # Resolve DeepSpeed Lightning checkpoint directories to an actual state-dict file.
        # Example path:
        #   .../last.ckpt/checkpoint/mp_rank_00_model_states.pt
        sft_checkpoint_path = model_args.sft_checkpoint
        if os.path.isdir(sft_checkpoint_path):
            ds_state_dict_path = os.path.join(
                sft_checkpoint_path, "checkpoint", "mp_rank_00_model_states.pt"
            )
            if os.path.isfile(ds_state_dict_path):
                print(
                    "Detected DeepSpeed checkpoint directory; "
                    f"loading state dict from {ds_state_dict_path}"
                )
                sft_checkpoint_path = ds_state_dict_path

        # Determine if checkpoint should be treated as a directory (HF/PEFT)
        # or as a PyTorch state dict file.
        is_directory = os.path.isdir(sft_checkpoint_path)
        print(f"model_args.peft_ckpt: {model_args.peft_ckpt}")
        
        if is_directory and model_args.peft_ckpt:
            # Load tokenizer from checkpoint
            print(f"Loading tokenizer from PEFT checkpoint: {model_args.sft_checkpoint}")
            model.text_tokenizer = AutoTokenizer.from_pretrained(
                model_args.sft_checkpoint, trust_remote_code=True
            )
            
            # Load text model from checkpoint
            model.text_model = AutoModelForCausalLM.from_pretrained(
                model_args.sft_checkpoint, trust_remote_code=True
            )
            
            # First initialize the text model with PEFT
            print("Loading as PEFT checkpoint directory")
            model.text_model = PeftModel.from_pretrained(
                model.text_model,
                model_args.sft_checkpoint,
                is_trainable=True
            )
            
            # Verify loaded adapters
            print("Loaded LoRA adapters:", model.text_model.active_adapter)
            
            # Optional: Merge weights into base model
            print("Merging SFT LoRA weights into base model...")
            model.text_model = model.text_model.merge_and_unload()
            print("Successfully merged SFT knowledge into base model")

        elif is_directory and not model_args.peft_ckpt:
            # Load tokenizer from checkpoint
            print(f"Loading tokenizer from checkpoint: {model_args.sft_checkpoint}")
            model.text_tokenizer = AutoTokenizer.from_pretrained(
                model_args.sft_checkpoint, trust_remote_code=True
            )
            
            # Use hf from_pretrained
            model.text_model = AutoModelForCausalLM.from_pretrained(
                model_args.sft_checkpoint, trust_remote_code=True
            )
            print(f"CALLING FIRST PREP_FOR_TRAINING with dna_model_finetune: {getattr(model_args, 'dna_model_finetune', False)}, dna_projection_finetune: {getattr(model_args, 'dna_projection_finetune', False)}")
            _ = _prep_for_training(
                model,
                model_args,  # must contain lora_r, lora_alpha, lora_dropout
                dna_model_finetune = getattr(model_args, "dna_model_finetune", False),
                dna_projection_finetune = getattr(model_args, "dna_projection_finetune", False)
            )
            print("model.text_model after loading", model.text_model)
            #check the tokenizer:

            print("Successfully loaded SFT checkpoint")

        else:
            # It's a PyTorch state dict file
            print("Loading as PyTorch state dict file")
            checkpoint = torch.load(sft_checkpoint_path, map_location="cpu")
            
            # replace model.text_model with text_model for all in state dict
            def new_key(k):
                if k.startswith("=model."): return k[6:]
                elif k.startswith("_forward_module."): return k[len("_forward_module."):]
                else: return k
            
            if "state_dict" in checkpoint:
                magic = {new_key(k): v for k, v in checkpoint["state_dict"].items()}
            elif "module" in checkpoint:
                magic = {new_key(k): v for k, v in checkpoint["module"].items()}
            elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
                # Direct state dict - the checkpoint itself is the state dict
                print("Detected direct state dict format")
                magic = {new_key(k): v for k, v in checkpoint.items()}
            else:
                raise ValueError(f"Unsupported checkpoint format: {sft_checkpoint_path}")
            
            # Handle prefix mapping for different model architectures
            lora_prefix = False
            for key in magic.keys():
                if "lora" in key:
                    lora_prefix = True
                    break
            
            if lora_prefix:
                print("Detected LoRA weights in state dict")
                # First prepare model for LoRA training
                dna_model_finetune = getattr(model_args, "dna_model_finetune", False)
                dna_projection_finetune = getattr(model_args, "dna_projection_finetune", True)

                # Detect SFT LoRA rank from any lora_A.*.weight (shape [r, in_dim]).
                # If it differs from the requested RL rank, load SFT at its native
                # rank, merge into the base, then re-attach a fresh RL-rank adapter.
                sft_lora_r = None
                for _k, _v in magic.items():
                    if "lora_A" in _k and hasattr(_v, "shape") and len(_v.shape) == 2:
                        sft_lora_r = int(_v.shape[0])
                        break
                rl_lora_r = int(getattr(model_args, "lora_r", 0))
                rl_lora_alpha = int(getattr(model_args, "lora_alpha", 0))
                rank_mismatch = (
                    sft_lora_r is not None and rl_lora_r > 0 and sft_lora_r != rl_lora_r
                )
                if rank_mismatch:
                    print(
                        f"[rl] SFT LoRA rank={sft_lora_r}, requested RL rank={rl_lora_r}; "
                        "loading SFT at native rank, merging into base, then attaching RL adapter."
                    )
                    # Phase 1: build adapter at SFT rank so load_state_dict shapes match.
                    # Convention used by both submit_dna_lora_sweeps.sh stages: alpha = 2*rank.
                    model_args.lora_r = sft_lora_r
                    model_args.lora_alpha = sft_lora_r * 2
                print(
                    f"CALLING SECOND PREP_FOR_TRAINING with dna_model_finetune: {dna_model_finetune}, "
                    f"dna_projection_finetune: {dna_projection_finetune}"
                )
                _prep_for_training(
                    model,
                    model_args,
                    dna_model_finetune=dna_model_finetune,
                    dna_projection_finetune=dna_projection_finetune,
                )
                
                # Print some diagnostic info about the keys
                model_keys = set(model.state_dict().keys())
                checkpoint_keys = set(magic.keys())
                print(f"Model has {len(model_keys)} keys")
                print(f"Checkpoint has {len(checkpoint_keys)} keys")
                
                # Try to map LoRA keys more intelligently
                new_magic = {}
                for k, v in magic.items():
                    # Try different prefix mappings based on common patterns
                    if "base_model.model" in k and k not in model_keys:
                        new_k = k.replace("text_model.base_model.model", "text_model")
                        if new_k in model_keys:
                            new_magic[new_k] = v
                            continue
                    
                    # Try removing common prefixes
                    if k.startswith("text_model.") and k not in model_keys:
                        new_k = "text_model.base_model.model." + k[len("text_model."):]
                        if new_k in model_keys:
                            new_magic[new_k] = v
                            continue
                    
                    # Keep original key if no mapping found
                    new_magic[k] = v
                
                # Include missing target modules in diagnostic info
                magic = new_magic
                print(f"After key mapping: {len(magic)} keys")
                
                # Then load weights, allowing missing/extra keys
                result = model.load_state_dict(magic, strict=False)
                
                if len(result.unexpected_keys) > 0:
                    print(f"Sample unexpected keys: {result.unexpected_keys[:5]}")
                if len(result.missing_keys) > 0:
                    print(f"Sample missing keys: {result.missing_keys[:5]}")
                    
                print(f"Loaded checkpoint with {len(result.missing_keys)} missing keys and {len(result.unexpected_keys)} unexpected keys")

                if rank_mismatch:
                    print(
                        f"[rl] Merging SFT LoRA (r={sft_lora_r}) into base and "
                        f"reattaching RL adapter at r={rl_lora_r}, alpha={rl_lora_alpha}."
                    )
                    model.text_model = model.text_model.merge_and_unload()
                    model_args.lora_r = rl_lora_r
                    model_args.lora_alpha = rl_lora_alpha
                    _prep_for_training(
                        model,
                        model_args,
                        dna_model_finetune=dna_model_finetune,
                        dna_projection_finetune=dna_projection_finetune,
                    )
            else:
                print("Standard weights detected - remapping keys")
                # Map keys to model structure
                magic = {k.replace("text_model", "text_model.base_model.model"): v for k, v in magic.items()}
                magic = {k.replace("protein_model", "protein_model"): v for k, v in magic.items()}
                
                # Fix the shared memory tensors issue by making a copy of weights
                for key in list(magic.keys()):
                    if 'lm_head.weight' in key:
                        magic[key] = magic[key].clone()
                
                # Load weights before setting up LoRA
                result = model.load_state_dict(magic, strict=False)
                print(f"Loaded checkpoint with {len(result.missing_keys)} missing keys and {len(result.unexpected_keys)} unexpected keys")
                
                # Now prepare for LoRA training
                dna_model_finetune = getattr(model_args, "dna_model_finetune", False)
                dna_projection_finetune = getattr(model_args, "dna_projection_finetune", True)
                print(
                    f"CALLING THIRD PREP_FOR_TRAINING with dna_model_finetune: {dna_model_finetune}, "
                    f"dna_projection_finetune: {dna_projection_finetune}"
                )
                _ = _prep_for_training(
                    model,
                    model_args,
                    dna_model_finetune=dna_model_finetune,
                    dna_projection_finetune=dna_projection_finetune,
                )
    
    else:
        # No checkpoint, just prepare for training
        dna_model_finetune = getattr(model_args, "dna_model_finetune", False)
        dna_projection_finetune = getattr(model_args, "dna_projection_finetune", True)
        print(
            f"CALLING FOURTH PREP_FOR_TRAINING with dna_model_finetune: {dna_model_finetune}, "
            f"dna_projection_finetune: {dna_projection_finetune}"
        )
        _ = _prep_for_training(
            model,
            model_args,
            dna_model_finetune=dna_model_finetune,
            dna_projection_finetune=dna_projection_finetune,
        )
    if script_args.full_ckpt is not None:
        print(f"Loading full checkpoint from {script_args.full_ckpt}")
        checkpoint_path = os.path.join(script_args.full_ckpt, "pytorch_model.bin")
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(checkpoint, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
            
            print(f"Loaded checkpoint with {len(missing)} missing keys and {len(unexpected)} unexpected keys")
        else:
            print(f"Checkpoint file not found at {checkpoint_path}")


    # Move the model to GPU
    model = model.to(training_args.device)

    vlm_module_cls = get_vlm_module(model_args.text_model_name)

    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    
    print("reward_funcs:", reward_funcs)
    print(f"Tokenizer loaded successfully. Vocab size: {len(model.text_tokenizer)}")
    print(f"ID for '<|dna_pad|>' is: {model.text_tokenizer.convert_tokens_to_ids('<|dna_pad|>')}")

    
    dataset = get_kegg_questions(
        truncate_dna_per_side=model_args.truncate_dna_per_side,
        kegg_data_dir_local=model_args.kegg_data_dir_local,
    )

    frac = float(getattr(model_args, "train_data_fraction", 1.0) or 1.0)
    if 0.0 < frac < 1.0:
        n_full = len(dataset["train"])
        n_keep = max(1, int(n_full * frac))
        dataset["train"] = dataset["train"].select(range(n_keep))
        print(f"[train_data_fraction] keeping {n_keep}/{n_full} GRPO train examples (fraction={frac})")

    # Custom callback to handle saving with PyTorch's native mechanism
    custom_save_callback = SaveWithPyTorchCallback()
    print("model.text_model:", model.text_model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {total_trainable:,} / {total_params:,} ({100 * total_trainable / total_params:.2f}%)")

    # Initialize the GRPO trainer with custom callback
    print("training_args:", training_args)
    trainer = DNALLMGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        dna_module=vlm_module_cls(),
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'] if training_args.eval_strategy != "no" else None,
        peft_config=None,
        callbacks=[custom_save_callback],
        processing_class=model.processor,  # Add our custom callback
    )

    # Set the trainer to save in PyTorch format instead of safetensors
    training_args.save_safetensors = False

    # Train and push the model to the Hub
    print("="*50)
    print("Verifying Trainable Parameters...")
    print("="*50)
    total_trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  [TRAINABLE]: {name} | Size: {param.shape} | Device: {param.device}")
            total_trainable_params += param.numel()
    print(f"\nTotal number of trainable parameters: {total_trainable_params:,}")
    print("="*50)

    # Handle resume from checkpoint
    # Note: resume_from_checkpoint is already in training_args from HuggingFace TrainingArguments
    resume_from_checkpoint = training_args.resume_from_checkpoint
    if isinstance(resume_from_checkpoint, str):
        lowered = resume_from_checkpoint.strip().lower()
        if lowered in {"false", "none", ""}:
            resume_from_checkpoint = None
        elif lowered == "true":
            resume_from_checkpoint = "true"

    if resume_from_checkpoint == "true":
        # Auto-detect latest checkpoint
        checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        if checkpoints:
            resume_from_checkpoint = str(max(checkpoints, key=os.path.getmtime))
            print(f"Auto-resuming from latest checkpoint: {resume_from_checkpoint}")
        else:
            print("No checkpoints found to resume from. Starting fresh training.")
            resume_from_checkpoint = None
    elif resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
    
    # ---------------------------------------------------------------
    # Eval-only path: load RL ckpt, run trainer.evaluate on chosen split.
    # 'id'  -> dataset['val']   (id_test_network_split.csv)
    # 'ood' -> dataset['test']  (ood_test_network_split.csv)
    # ---------------------------------------------------------------
    if bool(getattr(model_args, "eval_only", False)):
        eval_split = (getattr(model_args, "eval_split", "") or "id").lower()
        split_key = "test" if eval_split == "ood" else "val"
        chosen = dataset[split_key]
        print(f"[eval_only] split={eval_split} -> dataset['{split_key}'] N={len(chosen)}")

        ckpt_for_eval = (getattr(model_args, "eval_ckpt", "") or "").strip() or resume_from_checkpoint
        if ckpt_for_eval and ckpt_for_eval != "true":
            print(f"[eval_only] loading RL ckpt: {ckpt_for_eval}")
            try:
                trainer._load_from_checkpoint(ckpt_for_eval)
            except Exception as e:
                print(f"[eval_only] _load_from_checkpoint failed ({e}); falling back to trainer.train(max_steps=0).")
                training_args.max_steps = 0
                trainer.train(resume_from_checkpoint=ckpt_for_eval)

        metrics = trainer.evaluate(eval_dataset=chosen, metric_key_prefix=f"eval_{eval_split}")
        # GRPO accumulates rewards/correctness/mean, completions/*, kl, etc. into
        # trainer._metrics["eval"] inside compute_loss, but DNALLMGRPOTrainer does
        # not override log() so these are never flushed into the metrics dict
        # returned by trainer.evaluate(). Pull them in for the report and wandb.
        _all_metrics = getattr(trainer, "_metrics", {})
        print(f"[eval_only] trainer._metrics keys: train={list(_all_metrics.get('train', {}).keys())[:6]} eval={list(_all_metrics.get('eval', {}).keys())[:6]}")
        # GRPO compute_loss may write to either "train" or "eval" depending on
        # self.model.training; HF evaluate sets training=False but trl uses the
        # unwrapped model. Take whichever has data.
        _grpo_acc = _all_metrics.get("eval", {}) or _all_metrics.get("train", {})
        for _k, _v in _grpo_acc.items():
            if isinstance(_v, list) and _v:
                metrics[f"eval_{eval_split}_{_k}"] = sum(_v) / len(_v)
        print(f"[eval_only] metrics: {metrics}")
        try:
            import wandb as _wandb
            if _wandb.run is not None:
                _wandb.log(metrics)
        except Exception:
            pass
        return

    # Train and push the model to the Hub
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    # Avoid DeepSpeed MPI discovery dependency on libmpi in single-node jobs.
    os.environ.setdefault("DEEPSPEED_NO_MPI", "1")
    # Provide single-process distributed defaults so DeepSpeed won't try MPI discovery.
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    # Avoid HF datasets multiprocessing issues under multi-rank runs
    os.environ.setdefault("HF_DATASETS_DISABLE_MULTIPROCESSING", "1")
    # Set wandb project
    os.environ.setdefault("WANDB_PROJECT", "dna-grpo")
    parser = TrlParser((GRPOScriptArguments, DNALLMGRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    model_args.cache_dir = _configure_non_home_cache_dirs(model_args.cache_dir)
    if script_args.arrow_cache_dir is None:
        script_args.arrow_cache_dir = os.path.join(model_args.cache_dir, "arrow")
        os.makedirs(script_args.arrow_cache_dir, exist_ok=True)
    
    # Ensure we use PyTorch's save mechanism instead of safetensors
    training_args.save_safetensors = False
    # Prefer launcher-provided endpoint if available (set by run_grpo_multinode.sh)
    training_args.vllm_server_base_url = os.environ.get("VLLM_BASE_URL")

    main(script_args, training_args, model_args)
