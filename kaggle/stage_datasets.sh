#!/usr/bin/env bash
# Stage the two Kaggle datasets for the Orkun self-play run.
#   orkun-selfplay-code : Orkun package at root + minimal Orkish substrate under _orkish/
#   orkun-ckpt          : ema.safetensors (pretrained start)
# Orkish is READ ONLY here — we copy from it, never write to it.
set -euo pipefail

ORKUN="."
ORKISH="../Orkish"
STAGE="$ORKUN/kaggle/_stage"
CODE="$STAGE/code"
CKPT="$STAGE/ckpt"

rm -rf "$STAGE"
mkdir -p "$CODE" "$CKPT"

# --- orkun-selfplay-code : Orkun package (no caches/build/tests/runs) ---
rsync -a --prune-empty-dirs \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' \
  --exclude 'egg-info' --exclude '*.egg-info' \
  "$ORKUN/orkun" "$CODE/"
rsync -a "$ORKUN/configs" "$CODE/"
cp "$ORKUN/pyproject.toml" "$CODE/pyproject.toml"

# --- _orkish substrate: exactly what the Orkun code imports at runtime ---
OS="$CODE/_orkish"
mkdir -p "$OS/scripts" "$OS/data/tokenizer" "$OS/configs"
for pkg in infer model torch_impl train; do
  rsync -a --exclude '__pycache__' --exclude '*.pyc' "$ORKISH/$pkg" "$OS/"
done
# `train` (top-level) is needed by torch_impl.train.sft -> `from train.schedules import WSDSchedule`
# for the SFT warmup pass that crosses the cold-start tool-call format wall before self-play.
# scripts/ is a namespace package upstream (no __init__.py); copy only the two modules used.
cp "$ORKISH/scripts/verifier.py"  "$OS/scripts/verifier.py"
cp "$ORKISH/scripts/self_play.py" "$OS/scripts/self_play.py"

# Neutralise the model package __init__ in the COPY (Orkish untouched): upstream it does
# `from model.waaagh import ...` which pulls the MLX (macOS-only) implementation. Kaggle has
# no mlx and Orkun only ever uses `model.tokenizer` + `torch_impl.model.waaagh`, so an empty
# package init keeps `from model.tokenizer import OrkishTokenizer` working without mlx.
cat > "$OS/model/__init__.py" <<'PY'
# Staged for Kaggle (torch-only): the upstream __init__ imports the MLX model, unavailable
# on Kaggle. Orkun imports `model.tokenizer` and `torch_impl.model.waaagh` only.
PY
# runtime data the loop needs
cp "$ORKISH/data/tokenizer/orkish-bpe-8k.json" "$OS/data/tokenizer/"
cp "$ORKISH/data/tool_registry.json"           "$OS/data/"
cp "$ORKISH/configs/waaagh_grot_80m.yaml"       "$OS/configs/"
cp "$ORKISH/configs/waaagh_grot_80m_v11.yaml"   "$OS/configs/"

cat > "$CODE/dataset-metadata.json" <<'JSON'
{
  "title": "orkun-selfplay-code",
  "id": "maximemoya/orkun-selfplay-code",
  "licenses": [{"name": "CC0-1.0"}]
}
JSON

# --- orkun-ckpt : fallback init + packed SFT corpus for the warmup ---
# Init NOMINAL du kernel SFT = dernier step_*.safetensors du dataset orkish-ckpt
# (backbone WAAAGH 1.1, monté en 3e source — cf kernel_sft/kernel-metadata.json).
# On stage ici les poids RAW 1.0 (best, step 1100, val 4.21) comme FALLBACK si
# orkish-ckpt n'est pas attaché. Plus l'ema (val 4.50) — raw > ema, cf handoff.
cp "$ORKISH/out_final/run/ckpts/best.safetensors" "$CKPT/raw_best_10.safetensors"
# Packed PROCEDURAL SFT V3 corpus (~16k traces, seq 1408, oracles copy-only, every record
# verified via run_fresh in the real sandbox — same distribution as self-play; npz repackés
# avec le fix loss-mask 06-10). The orkun_sft Kaggle kernel trains the backbone on these
# (1800 steps ≈ 3.6 epochs) to cross the tool-call format wall before self-play.
cp "$ORKUN/orkun/data/sft_proc/sft_proc_train.npz" "$CKPT/sft_proc_train.npz"
cp "$ORKUN/orkun/data/sft_proc/sft_proc_val.npz"   "$CKPT/sft_proc_val.npz"
# Packed v3 CAVEMAN blend (8550/450) — run différé. ATTENTION: packés AVANT le fix
# loss-mask 06-10 → REPACK obligatoire avant tout run caveman.
cp "$ORKUN/orkun/data/sft_proc/sft_caveman_train.npz" "$CKPT/sft_caveman_train.npz"
cp "$ORKUN/orkun/data/sft_proc/sft_caveman_val.npz"   "$CKPT/sft_caveman_val.npz"
cat > "$CKPT/dataset-metadata.json" <<'JSON'
{
  "title": "orkun-ckpt-ema",
  "id": "maximemoya/orkun-ckpt",
  "licenses": [{"name": "CC0-1.0"}]
}
JSON

echo "=== staged code dataset ($CODE) ==="
find "$CODE" -type f -not -path '*/__pycache__/*' | sed "s#$CODE/##" | sort
echo "=== staged ckpt dataset ($CKPT) ==="
ls -la "$CKPT"
echo "=== sizes ==="
du -sh "$CODE" "$CKPT"
