"""ZipEnhancer backbone wrapper for the vendored community implementation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..vendor.zipenhancer_community.models.zipenhancer import (
    AttrDict,
    ZipEnhancer,
    mag_pha_istft,
    mag_pha_stft,
)

_DEFAULT_CFG = Path(__file__).resolve().parents[1] / "vendor" / "zipenhancer_community" / "configuration.json"


def build_backbone(config: str | Path | dict[str, Any] | None = None) -> ZipEnhancer:
    """Build an offline ZipEnhancer generator from an official-style config.

    `config` may be a ModelScope `configuration.json` path, a parsed
    configuration dict, or `None` to use the packaged community config.
    """
    if config is None:
        config = _DEFAULT_CFG
    if isinstance(config, (str, Path)):
        with open(config, encoding="utf-8") as f:
            model_cfg = json.load(f)["model"]
    elif isinstance(config, dict):
        model_cfg = config.get("model", config)
    else:
        raise TypeError(type(config))

    former = dict(model_cfg["former_conf"])
    former["causal"] = False
    h = AttrDict(
        dict(
            num_tsconformers=model_cfg["num_tsconformers"],
            dense_channel=model_cfg["dense_channel"],
            former_conf=former,
            batch_first=model_cfg["batch_first"],
            model_num_spks=model_cfg["model_num_spks"],
        )
    )
    return ZipEnhancer(h)


__all__ = ["AttrDict", "ZipEnhancer", "build_backbone", "mag_pha_stft", "mag_pha_istft"]
