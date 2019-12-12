import argparse
import sys

import matplotlib.pyplot as plt # pour l'affichage

import gym
from gym import wrappers, logger

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


def plot_reward(reward, title="", file_name="error", mean_size=500):
    list_e = []
    i = 0
    while i < len(reward):
        max_mean = i + mean_size
        if max_mean >= len(reward):
            max_mean = len(reward) - 1
        list_e.append(sum(reward[i:max_mean])/(max_mean - i))
        i += mean_size
    plt.figure(figsize=(10, 10))
    plt.plot(range(len(list_e)), list_e, 'k')
    plt.ylabel("Valeur de l'erreur")
    plt.xlabel("Avancement de l'apprentisage")
    plt.title(title)
    plt.savefig(f"src/img/{file_name}.png")
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v1', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 1000
    reward = 0
    done = False

    rewards = []
    for i in range(episode_count):
        ob = env.reset()
        rewardSum = 0
        steps = 0
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            rewardSum += reward
            steps += 1
            if done:
                rewards.append(rewardSum)
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    plot_reward(rewards, "rewards", "rewards", mean_size=10)
    env.close()
