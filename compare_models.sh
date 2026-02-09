#!/bin/bash
# Configuration
STEPS=400000
TASK="pinpad_four"
LOGDIR_BASE="/tmp/comparison_$(date +%Y%m%d_%H%M%S)"

# Environment Setup
export PYTHONPATH=$PYTHONPATH:.
CONDA_ENV="director"

echo "Starting comparison on GPU..."
echo "Total steps per model: $STEPS"
echo "Log directory base: $LOGDIR_BASE"

# 1. Run Standard RSSM
echo "--- Running Standard RSSM ---"
conda run --no-capture-output -n $CONDA_ENV python embodied/agents/director/train.py \
  --configs pinpad \
  --task $TASK \
  --train.steps $STEPS \
  --rssm_type rssm \
  --tf.jit False \
  --train.train_fill 2000 \
  --train.log_keys_video none \
  --logdir "$LOGDIR_BASE/rssm"

# 2. Run Context RSSM
echo "--- Running Context RSSM ---"
conda run --no-capture-output -n $CONDA_ENV python embodied/agents/director/train.py \
  --configs pinpad \
  --task $TASK \
  --train.steps $STEPS \
  --rssm_type context \
  --tf.jit False \
  --train.train_fill 2000 \
  --train.log_keys_video none \
  --logdir "$LOGDIR_BASE/context"

echo "Comparison finished."
echo "Results located in: $LOGDIR_BASE"
