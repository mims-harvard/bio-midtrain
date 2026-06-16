from __future__ import annotations

import os
import pkgutil
from pathlib import Path

import yaml
from huggingface_hub import constants, hf_hub_download, snapshot_download

from bioreason.utils.force_vortex_pytorch_linear import is_enabled as is_vortex_pytorch_linear_enabled


def load_evo2(model_name: str, local_path: str | None = None):
    from evo2 import Evo2

    if local_path is not None or not is_vortex_pytorch_linear_enabled():
        return Evo2(model_name, local_path=local_path)

    from evo2.utils import CONFIG_MAP

    config_bytes = pkgutil.get_data('evo2', CONFIG_MAP[model_name])
    if config_bytes is None:
        raise FileNotFoundError(f'Unable to load Evo2 config for {model_name}.')

    config = yaml.safe_load(config_bytes)
    if not config.get('use_fp8_input_projections', False):
        return Evo2(model_name, local_path=local_path)

    from evo2.utils import HF_MODEL_NAME_MAP
    from vortex.model.model import StripedHyena
    from vortex.model.tokenizer import CharLevelTokenizer
    from vortex.model.utils import dotdict, load_checkpoint

    filename = f'{model_name}.pt'
    hf_model_name = HF_MODEL_NAME_MAP[model_name]
    final_weights_path = Path(constants.HF_HUB_CACHE).parent / filename

    if final_weights_path.exists():
        weights_path = final_weights_path
        hf_hub_download(repo_id=hf_model_name, filename='config.json')
    else:
        repo_dir = Path(snapshot_download(repo_id=hf_model_name))
        repo_weights_path = repo_dir / filename
        if repo_weights_path.exists():
            print(f'Found complete file in repo: {filename}')
            weights_path = repo_weights_path
        else:
            parts = []
            part_num = 0
            while True:
                part_path = repo_dir / f'{filename}.part{part_num}'
                if not part_path.exists():
                    break
                parts.append(part_path)
                part_num += 1

            if not parts:
                raise FileNotFoundError(
                    f'Could not find {filename} or any of its shards in {repo_dir}'
                )

            print(f'Found {len(parts)} shards, merging them...')
            with final_weights_path.open('wb') as outfile:
                for part in parts:
                    print(f'Merging shard: {part.name}')
                    with part.open('rb') as infile:
                        while True:
                            chunk = infile.read(8192 * 1024)
                            if not chunk:
                                break
                            outfile.write(chunk)

            print(f'Successfully merged all shards into {final_weights_path}')
            weights_path = final_weights_path
            for part in parts:
                real_path = part.resolve()
                if real_path.exists():
                    real_path.unlink()
                if part.exists():
                    part.unlink()

    config['use_fp8_input_projections'] = False
    global_config = dotdict(config, Loader=yaml.FullLoader)
    print(
        f'Disabling Evo2 FP8 input projections for {model_name} because '
        'BIOREASON_USE_VORTEX_PYTORCH_LINEAR=1.'
    )
    model = StripedHyena(global_config)
    load_checkpoint(model, str(weights_path))

    evo2_model = Evo2.__new__(Evo2)
    evo2_model.model = model
    evo2_model.tokenizer = CharLevelTokenizer(512)
    return evo2_model
