from collections import deque

import gym
from gym import wrappers, logger

env_name = 'CartPole-v1'


class Agent:
    def __init__(self, action_space):
        self.mem = deque(maxlen=10000)
        self.action_space = action_space

    def choose_action(self, _observation, _reward, _done):
        return self.action_space.sample()

    def experience(self, _state, _action, _reward, _next_state, _done):
        self.mem.append((_state, _action, _reward, _next_state, _done))


# You can set the level to logger.DEBUG or logger.WARN if you
# want to change the amount of output.
logger.set_level(logger.INFO)

env = gym.make(env_name)

# You provide the directory to write to (can be an existing
# directory, including one with existing data -- all monitor files
# will be namespaced). You can also dump to a tempdir if you'd
# like: tempfile.mkdtemp().
outdir = '/tmp/random-agent-results'
env = wrappers.Monitor(env, directory=outdir, force=True)
env.seed(0)
agent = Agent(env.action_space)

episode_count = 1000
reward = 0
done = False

rewards = []
for i in range(episode_count):
    ob = env.reset()
    rewardSum = 0
    steps = 0
    while not done:
        action = agent.choose_action(ob, reward, done)
        current_state = ob
        ob, reward, done, _ = env.step(action)
        rewardSum += reward
        steps += 1

        print(ob, current_state)
        # Experience replay
        agent.experience(current_state, action, reward, ob, done)

        if done:
            rewards.append(rewardSum)
            break
        # Note there's no env.render() here. But the environment still can open window and
        # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
        # Video is not recorded every episode, see capped_cubic_video_schedule for details.

# Close the env and write monitor result info to disk
env.close()
