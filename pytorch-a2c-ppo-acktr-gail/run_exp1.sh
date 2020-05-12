#!/bin/bash

PRUNE_PCNT=0.6
PREFIX=exp1_cl_$PRUNE_PCNT_pruning
MODEL_PATH="trained_models/a2c/$PREFIX.pt"

# space invaders
python main.py \
    --num-processes=16 \
    --model-path=$MODEL_PATH \
    --cl-step=1 \
    --log-dir=logs/$PREFIX_spaceinvaders/ \
    --env-name=SpaceInvadersNoFrameskip-v4 \
    --max-prune-percent=$PRUNE_PCNT \
    --prune-interval=500 \
    --prune-percent=0.01 \
    --prune-start=50000

cp $MODEL_PATH "$MODEL_PATH.spaceinvaders"

# breakout
python main.py \
    --num-processes=16 \
    --model-path=$MODEL_PATH \
    --cl-step=2 \
    --log-dir=logs/$PREFIX_breakout/ \
    --env-name=SixActionBreakoutNoFrameskip-v4 \
    --max-prune-percent=$PRUNE_PCNT \
    --prune-interval=500 \
    --prune-percent=0.01 \
    --prune-start=50000

cp $MODEL_PATH "$MODEL_PATH.breakout"


# pong. no pruning for pong
python main.py \
    --num-processes=16 \
    --model-path=$MODEL_PATH \
    --cl-step=3 \
    --log-dir=logs/$PREFIX_pong/ \
    --env-name=PongNoFrameskip-v4 \
    --max-prune-percent=0.0

cp $MODEL_PATH "$MODEL_PATH.pong"
