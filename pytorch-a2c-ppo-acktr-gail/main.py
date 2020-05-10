from custom_envs.custom_envs import register_custom_envs
register_custom_envs()

import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate


def main():
    args = get_args()

    if args.cl_step == 1:
        args.env_name = 'PongNoFrameskip-v4'
        args.log_dir = 'logs/pong/'
    elif args.cl_step == 2:
        args.env_name = 'SixActionBreakoutNoFrameskip-v4'
        args.log_dir = 'logs/breakout/'
    elif args.cl_step == 3:
        args.env_name = 'SpaceInvadersNoFrameskip-v4'
        args.log_dir = 'logs/spaceinvaders/'

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    if args.model_path and os.path.exists(args.model_path):
        actor_critic, _ = torch.load(args.model_path)
        print('Loaded saved model at {}'.format(args.model_path))
    else:
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy}
        )

    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))

        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    # setting up for weight pruning/continual learning
    cl_version = args.cl_step
    cl_available_params = 0
    cl_total_params = 0
    cl_curr_sparsity = 0

    if cl_version is not None:
        for m in actor_critic.base.get_sparse_parameters():
            if args.no_update:
                m.mask_version = cl_version
            else:
                m.update_mask_version(cl_version)
            cl_available_params += int((m._mask_weight == cl_version).sum())
            cl_total_params += m._mask_weight.numel()

            if m.bias_shape is not None:
                cl_available_params += int((m._mask_bias == cl_version).sum())
                cl_total_params += m._mask_bias.numel()

    print('{}/{} ({:.2f}%) parameters are available to train.'.format(
        cl_available_params, cl_total_params,
        100.0 * cl_available_params / cl_total_params
    ))

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    for j in range(num_updates):
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step], deterministic=args.no_update)

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            if args.render:
                envs.render()

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        if not args.no_update:
            value_loss, action_loss, dist_entropy = agent.update(rollouts)
        else:
            value_loss, action_loss, dist_entropy = 0.0, 0.0, 0.0

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            if args.model_path is None:
                save_path = os.path.join(args.save_dir, args.algo)
                save_path = os.path.join(save_path, args.env_name + ".pt")
            else:
                save_path = args.model_path

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], save_path)

        # log every every once in a while
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        # evaluate every once in a while
        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

        # prune every once in a while. don't start pruning until some time
        # has passed
        if (args.max_prune_percent - cl_curr_sparsity > 0.001) and \
           (j >= 10 * args.prune_interval and j % args.prune_interval == 0):

            params = []
            for m in actor_critic.base.get_sparse_parameters():
                params.append((m._weight, m._mask_weight))
                if m.bias_shape is not None:
                    params.append((m._bias, m._mask_bias))

            sparse_params = 0
            for weight, mask in params:
                avail_idx = (mask == cl_version).nonzero().flatten()
                orig_count = ((mask == 0) | (mask == cl_version)).sum().item()
                # print(f'avail_idx = {len(avail_idx)} | orig_count = {orig_count} | total = {len(mask)}')
                remaining_pcnt = len(avail_idx) / orig_count if orig_count > 0 else 0

                prune_pcnt = min(remaining_pcnt - args.max_prune_percent, args.prune_percent)
                prune_pcnt = max(0, prune_pcnt)
                num_prune = int(prune_pcnt * orig_count)

                # sorting the weights by magnitude and getting their indices
                prune_idx = torch.argsort(weight[avail_idx].abs())[:num_prune]

                # setting those weights to zero in the mask
                mask[avail_idx[prune_idx]] = 0
                sparse_params += int((mask == 0).sum())

            cl_curr_sparsity = sparse_params / cl_available_params
            print(f'Pruned. Current sparsity = {100.0 * cl_curr_sparsity:.1f}%')


if __name__ == "__main__":
    main()
