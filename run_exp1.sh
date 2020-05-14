#!/bin/bash

PREFIX=exp1_no_pruning
MODEL_PATH="trained_models/a2c/$PREFIX.pt"

# space invaders
python main.py \
   --num-processes=16 \
   --model-path=$MODEL_PATH \
   --log-dir=logs/"$PREFIX"_spaceinvaders/ \
   --env-name=SpaceInvadersNoFrameskip-v4 \
   --eval-interval=5000 \
   --device="cuda:0"

cp $MODEL_PATH "$MODEL_PATH.spaceinvaders"

# breakout
python main.py \
   --num-processes=16 \
   --model-path=$MODEL_PATH \
   --log-dir=logs/"$PREFIX"_breakout/ \
   --env-name=SixActionBreakoutNoFrameskip-v4 \
   --eval-interval=5000 \
   --device="cuda:0"

cp $MODEL_PATH "$MODEL_PATH.breakout"


# pong. no pruning for pong
python main.py \
    --num-processes=16 \
    --model-path=$MODEL_PATH \
    --log-dir=logs/"$PREFIX"_pong/ \
    --env-name=PongNoFrameskip-v4 \
    --eval-interval=5000 \
    --device="cuda:0"

cp $MODEL_PATH "$MODEL_PATH.pong"
