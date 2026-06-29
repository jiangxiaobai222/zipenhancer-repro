#!/usr/bin/env bash
# =====================================================================
# Download VoiceBank+DEMAND (28spk), the standard ZipEnhancer/MP-SENet dataset.
# 来源: Edinburgh DataShare DS_10283_2791
# 原始采样率 48kHz, 训练时重采样到 16kHz
# Usage:
#   bash scripts/download_voicebank.sh
# =====================================================================
set -e

DEST=data/VoiceBank
mkdir -p "$DEST"
cd "$DEST"

BASE="https://datashare.ed.ac.uk/bitstream/handle/10283/2791"

FILES=(
    "clean_trainset_28spk_wav.zip"
    "noisy_trainset_28spk_wav.zip"
    "clean_testset_wav.zip"
    "noisy_testset_wav.zip"
)

for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "[skip] $f already exists"
    else
        echo "[download] $f ..."
        wget -c --no-check-certificate "$BASE/$f" -O "$f"
    fi
done

echo "[unzip] extracting ..."
for f in "${FILES[@]}"; do
    d="${f%.zip}"
    if [ ! -d "$d" ]; then
        unzip -q "$f"
    fi
done

echo "[done] VoiceBank at $(pwd)"
echo "  train clean: clean_trainset_28spk_wav/  ($(ls clean_trainset_28spk_wav 2>/dev/null | wc -l) wavs)"
echo "  train noisy: noisy_trainset_28spk_wav/  ($(ls noisy_trainset_28spk_wav 2>/dev/null | wc -l) wavs)"
echo "  test  clean: clean_testset_wav/         ($(ls clean_testset_wav 2>/dev/null | wc -l) wavs)"
echo "  test  noisy: noisy_testset_wav/         ($(ls noisy_testset_wav 2>/dev/null | wc -l) wavs)"
