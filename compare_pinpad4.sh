#!/usr/bin/env bash
set -euo pipefail

BASE_LOGDIR=${1:-$HOME/logdir/compare_pinpad4_$(date +%Y%m%d-%H%M%S)}
SEED=${SEED:-0}
STEPS=${STEPS:-5000000}
CONTEXT_DIM=${CONTEXT_DIM:-16}
CONTEXT_GATE_SCALE=${CONTEXT_GATE_SCALE:-1e-3}

mkdir -p "$BASE_LOGDIR"

echo "Logdir: $BASE_LOGDIR"
echo "Seed: $SEED"
echo "Steps: $STEPS"
echo "Context dim: $CONTEXT_DIM"
echo "Context gate scale: $CONTEXT_GATE_SCALE"

python embodied/agents/director/train.py \
  --logdir "$BASE_LOGDIR/baseline" \
  --configs pinpad \
  --task pinpad_four \
  --seed "$SEED" \
  --train.steps "$STEPS" \
  --rssm.context 0 \
  --loss_scales.context_gate "$CONTEXT_GATE_SCALE"

python embodied/agents/director/train.py \
  --logdir "$BASE_LOGDIR/context" \
  --configs pinpad \
  --task pinpad_four \
  --seed "$SEED" \
  --train.steps "$STEPS" \
  --rssm.context "$CONTEXT_DIM" \
  --loss_scales.context_gate "$CONTEXT_GATE_SCALE"

echo "Done. Logs in $BASE_LOGDIR"
