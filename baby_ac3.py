import os
import time
import argparse
import sys
import glob

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from scipy.signal import lfilter
from PIL import Image

os.environ["OMP_NUM_THREADS"] = "1"

# torch.set_default_tensor_type(torch.cuda.FloatTensor)


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--env", default="Breakout-v4", type=str, help="gym environment")
    parser.add_argument("--processes", default=12, type=int, help="number of processes to train with")
    parser.add_argument("--render", action="store_true", help="renders the atari environment")
    parser.add_argument("--test", action="store_true", help="sets lr=0, chooses most likely actions")
    parser.add_argument("--rnn_steps", default=20, type=int, help="steps to train LSTM over")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--seed", default=1, type=int, help="seed random # generators (for reproducibility)")
    parser.add_argument("--gamma", default=0.99, type=float, help="rewards discount factor")
    parser.add_argument("--tau", default=1.0, type=float, help="generalized advantage estimation discount")
    parser.add_argument("--horizon", default=0.99, type=float, help="horizon for running averages")
    parser.add_argument("--hidden", default=256, type=int, help="hidden size of GRU")
    return parser.parse_args()


def discount(x, gamma):
    # discounted rewards one liner
    return lfilter([1], [1, -gamma], x[::-1])[::-1]


def preprocess_state(img):
    # using Pillow's resize preserves single-pixel info unlike img[::2,::2]
    img = img[35:195].mean(2)
    img = np.array(Image.fromarray(img).resize((80, 80)))
    return img.astype(np.float32).reshape(1, 80, 80) / 255.0


def printlog(args, s, end="\n", mode="a"):
    print(s, end=end)
    with open(args.save_dir + "log.txt", mode) as fp:
        fp.write(s + "\n")


# an actor-critic neural network
class NNPolicy(nn.Module):
    def __init__(self, channels, memsize, num_actions):
        super(NNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.gru = nn.GRUCell(32 * 5 * 5, memsize)
        self.critic_linear = nn.Linear(memsize, 1)
        self.actor_linear = nn.Linear(memsize, num_actions)

    def forward(self, inputs, train=True, hard=False):
        inputs, hx = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        hx = self.gru(x.view(-1, 32 * 5 * 5), hx)

        critic = self.critic_linear(hx)
        actor = self.actor_linear(hx)
        return critic, actor, hx

    def try_load(self, save_dir):
        paths = glob.glob(save_dir + "*.tar")
        step = 0

        if len(paths) > 0:
            ckpts = [int(s.split(".")[-2]) for s in paths]
            ix = np.argmax(ckpts)
            step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
            print("\tloaded model: {}".format(paths[ix]))
        else:
            print("\tno saved models")

        return step


# extend a pytorch optimizer so it shares grads across processes
class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["shared_steps"] = torch.zeros(1).share_memory_()
                state["step"] = 0
                state["exp_avg"] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state["exp_avg_sq"] = p.data.new().resize_as_(p.data).zero_().share_memory_()

        def step(self, closure=None):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self.state[p]["shared_steps"] += 1
                    self.state[p]["step"] = self.state[p]["shared_steps"][0] - 1
                    # a "step += 1" comes later

            super.step(closure)


def cost_func(args, values, logps, actions, rewards):
    np_values = values.view(-1).data.cpu().numpy()

    # generalized advantage estimation using \delta_t residuals (a policy gradient method)
    delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1]
    logpys = logps.gather(1, actions.view(-1, 1))
    gen_adv_est = discount(delta_t, args.gamma * args.tau)
    policy_loss = -(logpys.view(-1) * torch.Tensor(gen_adv_est.copy())).sum()

    # l2 loss over value estimator
    rewards[-1] += args.gamma * np_values[-1]
    discounted_r = discount(np.asarray(rewards), args.gamma)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = 0.5 * (discounted_r - values[:-1, 0]).pow(2).sum()

    # entropy definition, for entropy regularization
    entropy_loss = (-logps * torch.exp(logps)).sum()
    return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss


def train(shared_model, shared_optimizer, rank, args, info):
    # make a local (unshared) environment
    env = gym.make(args.env)
    env.seed(args.seed + rank)

    # seed everything
    torch.manual_seed(args.seed + rank)

    # a local/unshared model
    model = NNPolicy(channels=1, memsize=args.hidden, num_actions=args.num_actions)

    # get first state and initialize hidden state
    state = torch.tensor(preprocess_state(env.reset()))
    hx = torch.zeros(1, 256)

    # bookkeeping
    start_time = last_disp_time = time.time()
    episode_length, epr, eploss, done = 0, 0, 0, True

    # openai baselines uses 40M frames...we'll use 80M
    while args.test or info["frames"][0] <= 8e7:
        if shared_model is not None:
            # sync with shared model
            model.load_state_dict(shared_model.state_dict())

        # rnn activation vector
        hx = torch.zeros(1, 256) if done else hx.detach()

        # save values for computing gradientss
        values, logps, actions, rewards = [], [], [], []

        for step in range(args.rnn_steps):
            episode_length += 1
            value, logit, hx = model((state.view(1, 1, 80, 80), hx))
            logp = F.log_softmax(logit, dim=-1)

            if args.test:
                action = logp.argmax(dim=1)
                # print('action.shape =', action.shape)
            else:
                action = torch.exp(logp).multinomial(num_samples=1)
                # print('action.shape =', action.shape)
                # action = action.item()

            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()

            state = torch.tensor(preprocess_state(state))
            epr += reward
            reward = np.clip(reward, -1, 1)  # reward
            done = done or episode_length >= 1e4  # don't playing one ep for too long

            info["frames"].add_(1)
            num_frames = int(info["frames"].item())
            if num_frames % 1000000 == 0:  # save every 1M frames
                printlog(
                    args, "\n\t{:.0f}M frames: saved model\n".format(num_frames / 1e6)
                )

                save_path = args.save_dir + "model.{:.0f}.tar".format(num_frames / 1e6)
                if shared_model is not None:
                    torch.save(shared_model.state_dict(), save_path)
                else:
                    torch.save(model.state_dict(), save_path)

            if done:  # update shared data
                info["episodes"] += 1
                interp = 1 if info["episodes"][0] == 1 else 1 - args.horizon
                info["run_epr"].mul_(1 - interp).add_(interp * epr)
                info["run_loss"].mul_(1 - interp).add_(interp * eploss)

            # print info ~ every minute
            if rank == 0 and time.time() - last_disp_time > 60:
                dur = time.gmtime(time.time() - start_time)
                elapsed = time.strftime("%Hh %Mm %Ss", dur)
                printlog(
                    args,
                    "time {}, episodes {:.0f}, frames {:.2f}M, mean epr {:.2f}, run loss {:.2f}".format(
                        elapsed,
                        info["episodes"].item(),
                        num_frames / 1e6,
                        info["run_epr"].item(),
                        info["run_loss"].item(),
                    ),
                )
                last_disp_time = time.time()

            if done:
                episode_length, epr, eploss = 0, 0, 0
                state = torch.tensor(preprocess_state(env.reset()))

            values.append(value)
            logps.append(logp)
            actions.append(action)
            rewards.append(reward)

        if done:
            next_value = torch.zeros(1, 1)
        else:
            next_value = model((state.unsqueeze(0), hx))[0]
        values.append(next_value.detach())

        loss = cost_func(
            args,
            torch.cat(values),
            torch.cat(logps),
            torch.cat(actions),
            np.asarray(rewards),
        )
        eploss += loss.detach()
        shared_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

        # sync gradients with shared model
        if shared_model:
            for param, shared_param in zip(model.parameters(), shared_model.parameters()):
                if shared_param.grad is None:
                    shared_param._grad = param.grad
        shared_optimizer.step()


if __name__ == "__main__":
    if sys.version_info[0] > 2:
        mp.set_start_method("spawn")  # this must not be in global scope

    args = get_args()
    args.save_dir = "{}/".format(args.env.lower())

    torch.manual_seed(args.seed)

    if args.render:
        args.processes = 1
        args.test = True  # render mode -> test mode w one process

    if args.test:
        args.lr = 0  # don't train in render mode

    args.num_actions = gym.make(args.env).action_space.n
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    shared_model = NNPolicy(
        channels=1,
        memsize=args.hidden,
        num_actions=args.num_actions
    ).share_memory()
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)

    info = {
        k: torch.DoubleTensor([0]).share_memory_()
        for k in ["run_epr", "run_loss", "episodes", "frames"]
    }
    info["frames"] += shared_model.try_load(args.save_dir) * 1e6

    # clear log file
    if int(info["frames"].item()) == 0:
        printlog(args, "", end="", mode="w")

    processes = []
    for rank in range(args.processes):
        p = mp.Process(
            target=train, args=(shared_model, shared_optimizer, rank, args, info)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # rank = 0
    # train(shared_model, shared_optimizer, rank, args, info)
