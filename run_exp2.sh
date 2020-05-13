#!/bin/bash

PRUNE_PCNT=0.6
PREFIX=exp2_cl_"$PRUNE_PCNT"_pruning
MODEL_PATH="trained_models/a2c/$PREFIX.pt"

# space invaders
python main.py \
    --num-processes=16 \
    --model-path=$MODEL_PATH \
    --log-dir=logs/"$PREFIX"_spaceinvaders/ \
    --env-name=SpaceInvadersNoFrameskip-v4 \
    --eval-interval=1000 \
    --max-prune-percent=$PRUNE_PCNT \
    --prune-interval=500 \
    --prune-percent=0.01 \
    --prune-start=500 \
    --cl-step=1

cp $MODEL_PATH "$MODEL_PATH.spaceinvaders"

# breakout
python main.py \
    --num-processes=16 \
    --model-path=$MODEL_PATH \
    --log-dir=logs/"$PREFIX"_breakout/ \
    --env-name=SixActionBreakoutNoFrameskip-v4 \
    --eval-interval=1000 \
    --max-prune-percent=$PRUNE_PCNT \
    --prune-interval=500 \
    --prune-percent=0.01 \
    --prune-start=500 \
    --cl-step=2

cp $MODEL_PATH "$MODEL_PATH.breakout"


# pong. no pruning for pong
python main.py \
    --num-processes=16 \
    --model-path=$MODEL_PATH \
    --log-dir=logs/"$PREFIX"_pong/ \
    --env-name=PongNoFrameskip-v4 \
    --eval-interval=1000 \
    --cl-step=3

cp $MODEL_PATH "$MODEL_PATH.pong"
