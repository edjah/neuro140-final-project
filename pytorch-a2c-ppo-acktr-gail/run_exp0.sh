#!/bin/bash

PREFIX=exp0_baseline

# space invaders
python main.py \
    --num-processes=16 \
    --model-path=trained_models/a2c/$PREFIX_spaceinvaders.pt \
    --cl-step=1 \
    --log-dir=logs/$PREFIX_spaceinvaders/ \
    --env-name=SpaceInvadersNoFrameskip-v4


# breakout
python main.py \
    --num-processes=16 \
    --model-path=trained_models/a2c/$PREFIX_breakout.pt \
    --cl-step=1 \
    --log-dir=logs/$PREFIX_breakout/ \
    --env-name=SixActionBreakoutNoFrameskip-v4


# pong
python main.py \
    --num-processes=16 \
    --model-path=trained_models/a2c/$PREFIX_pong.pt \
    --cl-step=1 \
    --log-dir=logs/$PREFIX_pong/ \
    --env-name=PongNoFrameskip-v4
