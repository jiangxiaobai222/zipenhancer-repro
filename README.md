# ZipEnhancer Repro

Language: English | [中文](README.zh-CN.md)

An unofficial PyTorch reproduction and optimized inference package for
[ZipEnhancer: Dual-Path Down-Up Sampling-based Zipformer for Monaural Speech
Enhancement](https://arxiv.org/abs/2501.05183).

This repository can train ZipEnhancer-S on VoiceBank+DEMAND, strict-load the
official ModelScope weights, and run ready-to-use speech enhancement with two
packaged lightweight checkpoints.

![ZipEnhancer architecture](docs/assets/zipenhancer_architecture.png)

*Architecture overview from ZipEnhancer paper Fig. 1.*

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[download]"
```

Run optimized inference on the packaged sample wav with the
official-compatible checkpoint:

```bash
bash examples/quick_infer_official.sh
```

Run optimized inference on the same sample with the VoiceBank reproduction
checkpoint:

```bash
bash examples/quick_infer_voicebank.sh
```

The repository does not redistribute datasets. It includes two lightweight
ZipEnhancer-S checkpoint files for quick inference: an official-compatible
ModelScope weight mirror and a VoiceBank+DEMAND reproduction checkpoint.

## Model Card Summary

- Task: single-channel speech enhancement / acoustic noise suppression.
- Input: 16 kHz mono noisy waveform.
- Output: 16 kHz mono enhanced waveform with the same sample domain.
- Backbone: ZipEnhancer-S, a TF-domain dual-path model with magnitude and phase
  decoding, FT-Zipformer blocks, and paired downsample/upsample stacks.
- Size: about 2.04M parameters in the packaged ZipEnhancer-S configuration.
- Official model: ModelScope
  `iic/speech_zipenhancer_ans_multiloss_16k_base`, released for acoustic noise
  suppression. The official model card reports PESQ 3.69 on DNS2020 and PESQ
  3.63 on VoiceBank+DEMAND for ZipEnhancer-S.
- License boundary: this repository code is MIT-licensed. Packaged checkpoints
  keep their original usage boundaries; follow the official ModelScope model
  card and license terms when using the official-compatible weight.

## What This Repository Adds

This is not a clean-room rewrite of every ZipEnhancer backbone layer. The
backbone is vendored from a community PyTorch extraction so this package can
strict-load the official ModelScope weights. The contributions here are the
reproducible training system, compatibility validation, optimized offline
inference, and packaging around that backbone.

- Official-compatible backbone packaging:
  vendor the minimum community ZipEnhancer-S PyTorch implementation, wrap it as
  `zipenhancer_repro.models.backbone`, and verify strict-load compatibility with
  the official ModelScope `pytorch_model.bin`.
- VoiceBank+DEMAND training reproduction:
  dataset pipeline, MP-SENet-style multi-loss, PESQ-GAN discriminator,
  ScaledAdam/Eden, checkpoint save/resume, TensorBoard audio/spectrogram logs,
  and full-set chunked evaluation.
- Offline inference optimization:
  global normalization, Hann overlap-add chunking, relative-position no-repeat
  memory patch, fp16 Swoosh compatibility patch, and numerical equivalence
  checks.
- Ready-to-run inference release:
  two packaged checkpoints under `weights/`, optimized inference entry points,
  and quick scripts for file or directory enhancement without downloading extra
  model files.
- Independent open-source package structure:
  `pyproject.toml`, console entry points, configs, scripts, examples, tests,
  MIT license, and third-party notices.

## Upstream And Acknowledgements

This project is an unofficial reproduction and optimization package. Please cite
and credit the original work when using it:

- ZipEnhancer paper: [arXiv:2501.05183](https://arxiv.org/abs/2501.05183)
- Official ModelScope model:
  [iic/speech_zipenhancer_ans_multiloss_16k_base](https://www.modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base/summary)
- Community PyTorch extraction used as the vendor basis:
  [boreas-l/zipEnhancer](https://github.com/boreas-l/zipEnhancer)
- Optimizer components are adapted from
  [k2-fsa/icefall](https://github.com/k2-fsa/icefall) ScaledAdam/Eden.
- Loss/discriminator logic follows
  [yxlu-0102/MP-SENet](https://github.com/yxlu-0102/MP-SENet)-style speech
  enhancement training.

See `NOTICE` for third-party source notes. Packaged checkpoints are provided
for quick reproduction and inference only; datasets are still downloaded by the
user.

## Project Structure

```text
configs/                         Reproduction and inference YAML configs
docs/assets/                     README figures and public documentation assets
examples/                        Minimal command-line examples and sample wavs
examples/test_datas/             Two noisy wav files for quick inference checks
scripts/                         Download and data preparation helpers
weights/                         Packaged official-compatible and reproduced checkpoints
src/zipenhancer_repro/
  data/                          VoiceBank+DEMAND dataset loader
  losses/                        MP-SENet-style losses and PESQ-GAN discriminator
  models/                        Public backbone wrapper
  optim/                         ScaledAdam/Eden optimizer wrapper
  infer_opt/                     Optimized offline inference and validation
  vendor/zipenhancer_community/  Official-compatible ZipEnhancer-S backbone
  train.py                       VoiceBank training entry point
  infer.py                       Chunked overlap-add inference entry point
  evaluate.py                    VoiceBank evaluation entry point
  verify_official.py             Official-weight strict-load check
tests/                           Lightweight package import tests
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[download]"
```

For development and local verification:

```bash
pip install -e ".[dev,download]"
python -m pytest -q
```

## Data And Official Weights

Prepare VoiceBank+DEMAND under:

```text
data/VoiceBank/clean_trainset_28spk_wav
data/VoiceBank/noisy_trainset_28spk_wav
data/VoiceBank/clean_testset_wav
data/VoiceBank/noisy_testset_wav
```

Two checkpoints are included for quick inference:

```text
weights/pytorch_model.bin        official-compatible ModelScope ZipEnhancer-S weight
weights/best_00082500.bin        reproduced VoiceBank+DEMAND checkpoint, step 82.5k
```

You can also download or refresh official weights from ModelScope:

```bash
bash scripts/download_official.sh checkpoints/official
python -m zipenhancer_repro.verify_official \
  --weights checkpoints/official/pytorch_model.bin
```

The package can train from random initialization without official weights:

```bash
python -m zipenhancer_repro.train --config configs/zipenhancer_s.yaml
```

`--init-weights` is optional and only initializes from official weights or a
previous checkpoint.

## Training

```bash
python -m zipenhancer_repro.train --config configs/zipenhancer_s.yaml --smoke
python -m zipenhancer_repro.train --config configs/zipenhancer_s.yaml
python -m zipenhancer_repro.train --config configs/zipenhancer_s.yaml --resume auto
```

TensorBoard logs are written under the configured output directory:

```bash
tensorboard --logdir outputs/zipenhancer_s/tb
```

### Current Reproduction Status

Experiments were run on both NVIDIA A10 and H20 GPUs. The training code has been
validated on VoiceBank+DEMAND from scratch. Representative checkpoints:

| Hardware | Dataset          |  Step | Evaluation          | WB-PESQ |   STOI | SI-SDR | Notes                                                         |
| -------- | ---------------- | ----: | ------------------- | ------: | -----: | -----: | ------------------------------------------------------------- |
| A10      | VoiceBank+DEMAND |   90k | full 824 utterances |  3.5267 | 0.9588 | 18.283 | best observed in the long A10 reproduction run                |
| A10      | VoiceBank+DEMAND |  285k | full 824 utterances |  3.4663 | 0.9613 | 18.941 | latest evaluated long-run checkpoint, not the best checkpoint |
| H20      | VoiceBank+DEMAND | 82.5k | full 824 utterances |  3.5265 | 0.9597 | 19.689 | best observed in the H20 large-batch reproduction run         |

These numbers are reproduction progress, not official claims. The official
paper reports PESQ 3.63 on VoiceBank+DEMAND and 3.69 on DNS2020 for ZipEnhancer-S.

Training curves from the VoiceBank reproduction runs:

![VoiceBank A10 reproduction TensorBoard curves](docs/assets/eval_tb.png)

![VoiceBank H20 reproduction TensorBoard curves](docs/assets/eval_tb_h20.png)

## Inference

The recommended path is optimized full-utterance inference through
`infer_opt.infer_lite`. The quick scripts use it first and fall back to chunked
overlap-add if the input is too long for the available GPU memory.

Fast path with the packaged official-compatible weight:

```bash
bash examples/quick_infer_official.sh \
  examples/test_datas/speech_with_noise.wav \
  outputs/enhanced_official.wav
```

Fast path with the packaged VoiceBank reproduction checkpoint:

```bash
bash examples/quick_infer_voicebank.sh \
  examples/test_datas/speech_with_noise.wav \
  outputs/enhanced_voicebank.wav
```

Use the raw optimized entry point when you want direct control over the
full-utterance path:

```bash
python -m zipenhancer_repro.infer_opt.infer_lite \
  --ckpt weights/pytorch_model.bin \
  --input examples/test_datas/speech_with_noise.wav \
  --output outputs/enhanced_lite.wav
```

Compatibility-oriented chunked overlap-add inference remains available for
official-weight-aligned checks, longer utterances, and directory processing:

```bash
python -m zipenhancer_repro.infer \
  --ckpt weights/pytorch_model.bin \
  --input examples/test_datas \
  --output outputs/enhanced_official_chunked
```

Optimized full-utterance inference:

```bash
python -m zipenhancer_repro.infer_opt.verify_numeric \
  --ckpt weights/pytorch_model.bin

python -m zipenhancer_repro.infer_opt.infer_lite \
  --ckpt weights/pytorch_model.bin \
  --input examples/test_datas/speech_with_noise.wav \
  --output enhanced.wav
```

## Inference Results On VoiceBank

All rows below use the VoiceBank test set with 824 utterances. Peak memory was
measured on an NVIDIA A10. `infer_opt` uses global normalization, Hann overlap-add,
and the relative-position no-repeat patch.

### Official DNS2020 Weight, Different Inference Algorithms

| Cell                    | Dtype | Window | Splice      | Norm      | Trigger | WB-PESQ |   STOI | SI-SDR | Peak MB |    RTF |
| ----------------------- | ----- | -----: | ----------- | --------- | ------- | ------: | -----: | -----: | ------: | -----: |
| official 2s             | fp32  |     2s | hard splice | per chunk | >6s     |  3.2836 | 0.9537 | 19.528 |  5603.2 | 0.0623 |
| infer_opt 2s            | fp32  |     2s | Hann-OLA    | global    | always  |  3.4124 | 0.9598 | 19.937 |   746.5 | 0.1006 |
| infer_opt 2s            | fp16  |     2s | Hann-OLA    | global    | always  |  3.4121 | 0.9598 | 19.938 |   378.6 | 0.0736 |
| infer_opt 4s            | fp16  |     4s | Hann-OLA    | global    | always  |  3.4278 | 0.9613 | 20.011 |  1377.4 | 0.0458 |
| 4s hard-splice baseline | fp16  |     4s | hard splice | per chunk | >8s     |  2.6514 | 0.8939 | 12.741 |  4270.3 | 0.0593 |

The recommended production default for official weights is `infer_opt 2s fp16`:
it keeps PESQ essentially identical to fp32 while reducing peak memory to about
379 MB.

### Official Weight Versus Reproduced VoiceBank Checkpoint

| Checkpoint           | Algorithm             |       N | OOM | WB-PESQ |   STOI | SI-SDR | Peak MB |    RTF |
| -------------------- | --------------------- | ------: | --: | ------: | -----: | -----: | ------: | -----: |
| official DNS2020     | infer_opt fp16        | 824/824 |   0 |  3.4121 | 0.9598 | 19.938 |   378.6 | 0.0751 |
| official DNS2020     | ModelScope-style fp32 | 824/824 |   0 |  3.2836 | 0.9537 | 19.528 |  5603.2 | 0.0617 |
| reproduced VoiceBank | infer_opt fp16        | 824/824 |   0 |  3.5128 | 0.9602 | 19.784 |   378.6 | 0.0727 |
| reproduced VoiceBank | ModelScope-style fp32 | 824/824 |   0 |  3.5284 | 0.9601 | 19.876 |  5603.2 | 0.0591 |

Example noisy / clean / enhanced spectrogram from the reproduction evaluation:

![Speech enhancement spectrogram example](docs/assets/se_example.png)

## Local Release Checklist

```bash
python -m pytest -q
python -m zipenhancer_repro.verify_official --weights weights/pytorch_model.bin
CUDA_VISIBLE_DEVICES=0 python -m zipenhancer_repro.train --config configs/zipenhancer_s.yaml --smoke
python -m zipenhancer_repro.infer_opt.verify_numeric --ckpt weights/pytorch_model.bin --seconds 1.0
```

## Scope

This first public package intentionally excludes internal experiment records
and ongoing extension research. Completed and validated extensions will be
released in future updates. Vendored code is kept minimal and documented in
`NOTICE`.
