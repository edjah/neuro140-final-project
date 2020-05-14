#!/bin/bash

PREFIX=exp0_baseline

# space invaders
python main.py \
   --num-processes=16 \
   --model-path=trained_models/a2c/"$PREFIX"_spaceinvaders.pt \
   --log-dir=logs/"$PREFIX"_spaceinvaders/ \
   --env-name=SpaceInvadersNoFrameskip-v4 \
   --eval-interval=5000 \
   --device="cuda:0"


# breakout
python main.py \
   --num-processes=16 \
   --model-path=trained_models/a2c/"$PREFIX"_breakout.pt \
   --log-dir=logs/"$PREFIX"_breakout/ \
   --env-name=SixActionBreakoutNoFrameskip-v4 \
   --eval-interval=5000 \
   --device="cuda:0"


# pong
python main.py \
    --num-processes=16 \
    --model-path=trained_models/a2c/"$PREFIX"_pong.pt \
    --log-dir=logs/"$PREFIX"_pong/ \
    --env-name=PongNoFrameskip-v4 \
    --eval-interval=5000 \
    --device="cuda:0"
