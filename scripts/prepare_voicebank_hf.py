"""Download VoiceBank+DEMAND (16k) from HuggingFace (via hf-mirror) and extract
to the standard wav-folder layout used by MP-SENet / our PairedSEDataset.

Output:
    data/VoiceBank/clean_trainset_28spk_wav/{id}.wav
    data/VoiceBank/noisy_trainset_28spk_wav/{id}.wav
    data/VoiceBank/clean_testset_wav/{id}.wav
    data/VoiceBank/noisy_testset_wav/{id}.wav

Run (zipses env):
    HF_ENDPOINT=https://hf-mirror.com python scripts/prepare_voicebank_hf.py
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "JacobLinCool/VoiceBank-DEMAND-16k"
DEST = os.path.join("data", "VoiceBank")

SPLITS = {
    "train": {
        "files": [f"data/train-0000{i}-of-00005.parquet" for i in range(5)],
        "clean_dir": "clean_trainset_28spk_wav",
        "noisy_dir": "noisy_trainset_28spk_wav",
    },
    "test": {
        "files": ["data/test-00000-of-00001.parquet"],
        "clean_dir": "clean_testset_wav",
        "noisy_dir": "noisy_testset_wav",
    },
}


def extract_split(name, spec):
    clean_dir = os.path.join(DEST, spec["clean_dir"])
    noisy_dir = os.path.join(DEST, spec["noisy_dir"])
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    n = 0
    for fname in spec["files"]:
        print(f"[{name}] downloading {fname} ...", flush=True)
        path = hf_hub_download(repo_id=REPO, filename=fname, repo_type="dataset")
        table = pq.read_table(path)
        ids = table["id"].to_pylist()
        cleans = table["clean"].to_pylist()
        noisies = table["noisy"].to_pylist()
        for i, cid in enumerate(ids):
            with open(os.path.join(clean_dir, f"{cid}.wav"), "wb") as f:
                f.write(cleans[i]["bytes"])
            with open(os.path.join(noisy_dir, f"{cid}.wav"), "wb") as f:
                f.write(noisies[i]["bytes"])
            n += 1
        print(f"[{name}] {fname}: extracted {len(ids)} pairs (total {n})", flush=True)
    print(f"[{name}] DONE -> {clean_dir} / {noisy_dir} ({n} pairs)\n", flush=True)


def main():
    for name, spec in SPLITS.items():
        extract_split(name, spec)
    print("[all done] VoiceBank+DEMAND 16k ready under data/VoiceBank/")


if __name__ == "__main__":
    main()
