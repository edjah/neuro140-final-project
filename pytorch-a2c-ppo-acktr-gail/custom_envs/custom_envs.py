import gym.envs.atari
from gym import spaces
from gym.envs.registration import register


# we need this env to be consistent with Pong and SpaceInvaders
# which both have 6 actions
class SixActionBreakout(gym.envs.atari.AtariEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.old_action_space = self.action_space
        self.action_space = spaces.Discrete(6)

    def step(self, a):
        a = a % self.old_action_space.n
        return super().step(a)

    @property
    def _n_actions(self):
        return self.action_space.n


def register_custom_envs():
    register(
        id='SixActionBreakoutNoFrameskip-v4',
        entry_point='custom_envs.custom_envs:SixActionBreakout',
        kwargs={'game': 'breakout', 'obs_type': 'image', 'frameskip': 1},
        max_episode_steps=4 * 100000,
        nondeterministic=False,
    )
