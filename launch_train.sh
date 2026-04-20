#!/usr/bin/env bash
# ============================================================
# Cluster Data Selection — 8-GPU DDP launch script.
# All hyperparameters live in configs/default.yaml; this script
# only handles launcher args (GPUs, port), env vars, and output dir.
#
# Usage:
#   bash launch_train.sh                    # default: 8 GPU, auto timestamped OUT_DIR
#   bash launch_train.sh 4                  # use 4 GPU instead
#   OUT_DIR=outputs/my_run bash launch_train.sh   # custom output dir
#   MASTER_PORT=29501 bash launch_train.sh  # change port if 29500 is taken
# ============================================================
set -euo pipefail

NGPU="${1:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
OUT_DIR="${OUT_DIR:-outputs/run_$(date +%Y%m%d_%H%M%S)}"
CONFIG="${CONFIG:-configs/default.yaml}"

mkdir -p "$OUT_DIR"

# Reduce CUDA memory fragmentation (helps near-full GPUs on long runs).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=============================================="
echo "  NGPU        : $NGPU"
echo "  MASTER_PORT : $MASTER_PORT"
echo "  CONFIG      : $CONFIG"
echo "  OUT_DIR     : $OUT_DIR"
echo "  LOG         : $OUT_DIR/train.log"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "=============================================="

# Run in foreground; redirect all stdout/stderr to the per-run log.
# To run in background: append ' > $OUT_DIR/train.log 2>&1 &' or use `nohup`.
torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
    train.py --config "$CONFIG" \
    training.save_dir="$OUT_DIR" \
    2>&1 | tee "$OUT_DIR/train.log"
