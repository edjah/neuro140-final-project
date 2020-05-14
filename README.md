# Continual Learning in Video Games

Note: This repository is based on an PyTorch implementation of the Advantage Actor Critic algorithm here: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail.

This was a final project for [Neuro 140](http://klab.tch.harvard.edu/academia/classes/BAI/bai.html) at Harvard.


## Background
Continual learning is the problem of training a neural network on different tasks in a sequential fashion without ever retraining on previous tasks. When the task switches, standard neural networks will adjust their weights to be better suited for the current task. This leads them to generally fail to retain their performance on previous tasks -- a phenomenon known as **catastrophic forgetting**.

In this repository, we solve the continual learning problem in a video game reinforcement learning context by utilizing a technique known as [weight pruning](https://arxiv.org/abs/1510.00149) to preserve only the weights that are important to each task. This allows the network to fully retain its performance on previous tasks while opening up capacity to learn new ones.

See the associated [paper](paper.pdf) for more details.


## Requirements
The requirements for this library are the same as the ones for the [repo](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) it's based on. In summary:

- Python 3
- PyTorch
- OpenAI baselines (also requires Tensorflow)

To install the requirements, run:
```
# PyTorch and Tensorflow
pip install pytorch tensorflow

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
pip install -e baselines/

# Other requirements
pip install -r requirements.txt
```

## How to Use

Our reinforcement learning environments come from the OpenAI gym. The environments that you use must have in common:
- The number of actions
- The shape of the observations

If either of these requirements is not met, custom environments which normalize these properties can be constructed. See [here](custom_envs/custom_envs.py) for an example.

#### Training
To train a model on tasks in a continual manner, you would do something like this:
```
# Train a model with 80% sparsity to play space invaders. The default is 10m frames
python main.py --cl-step=1 --env-name=SpaceInvadersNoFrameskip-v4 --model-path=continual.pt --max-prune-percent=0.8

# Update the model to play breakout. 80% of the weights are available for retraining.
python main.py --cl-step=2 --env-name=SixActionBreakoutNoFrameskip-v4 --model-path=continual.pt --max-prune-percent=0.8

# Update the model to play pong. 64% of the weights are available for retraining
python main.py --cl-step=3 --env-name=PongNoFrameskip-v4 --model-path=continual.pt
```

#### Evaluation
To evaluate the model (i.e. render the game), you can you run:
```
# play space invaders like a pro
python main.py --cl-step=1 --env-name=SpaceInvadersNoFrameskip-v4 --model-path=continual-pretrained.pt --no-update --render

# play breakout like a pro
python main.py --cl-step=2 --env-name=SixActionBreakoutNoFrameskip-v4 --model-path=continual-pretrained.pt --no-update --render

# play pong like a pro
python main.py --cl-step=3 --env-name=PongNoFrameskip-v4 --model-path=continual-pretrained.pt --no-update --render
```
