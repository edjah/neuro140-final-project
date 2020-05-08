import time
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import gym

from PIL import Image

torch.set_default_tensor_type(torch.cuda.FloatTensor)


class Model(nn.Module):
    def __init__(self, hidden_size, num_actions):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ELU(),
        )

        self.rnn = nn.GRUCell(32 * 5 * 5, hidden_size)
        self.out = nn.Linear(hidden_size, num_actions)

    def forward(self, x, hx=None):
        x = self.cnn(x).view(1, -1)
        hx = self.rnn(x, hx)
        return self.out(hx), hx


def preprocess_state(img):
    # make an 80 x 80 grayscale image
    img = img[35:195].mean(2)
    img = np.array(Image.fromarray(img).resize((80, 80)))
    return img.astype(np.float32).reshape(1, 1, 80, 80) / 255.0


def run_it(model, env_name, seed, render=False):
    # TODO: make this parallelizeable
    model.eval()

    env = gym.make(env_name)
    env.seed(seed)

    state = torch.tensor(preprocess_state(env.reset()))
    hx = None
    done = False

    total_reward = 0
    num_frames = 0

    while not done:
        with torch.no_grad():
            values, hx = model(state, hx)
            action = values.argmax().item()

        state, reward, done, _ = env.step(action)
        if render:
            env.render()

        state = torch.tensor(preprocess_state(state))
        num_frames += 1
        total_reward += reward

    return total_reward, num_frames


def main():
    action_space_size = gym.make('Pong-v0').action_space.n
    model = Model(hidden_size=128, num_actions=action_space_size)
    run_it_model = Model(hidden_size=128, num_actions=action_space_size)

    num_params = 0
    for k, p in model.named_parameters():
        num_params += p.numel()
    print(f'total num params = {num_params / 1e6:.3f}m')

    pop_size = 50
    sigma = 0.1
    alpha = 0.001
    total_frame_count = 0

    for epoch in range(1000):
        epoch_start = time.time()
        avg_reward = 0

        base_params = model.state_dict()
        neighbor_params = []
        rewards = []

        for i in range(pop_size):
            p = {
                k: v + torch.randn(v.shape) * sigma
                for k, v in base_params.items()
            }
            run_it_model.load_state_dict(p)
            neighbor_params.append(p)

            reward, num_frames = run_it(run_it_model, 'Pong-v0', 0, False)

            rewards.append(reward)
            total_frame_count += num_frames
            avg_reward += reward / pop_size

        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / rewards.std()
        avg_params = {k: torch.zeros(v.shape) for k, v in base_params.items()}
        for p, r in zip(neighbor_params, rewards.unbind()):
            for k, v in p.items():
                avg_params[k] += alpha * r / (pop_size * sigma) * v

        model.load_state_dict(avg_params)

        epoch_end = time.time()
        print(f'Epoch: {epoch} | Frames: {total_frame_count} | Avg reward: {avg_reward} | {epoch_end - epoch_start:.1f} sec')


if __name__ == '__main__':
    main()
