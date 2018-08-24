import gym
import gym_dc
import numpy as np
from gym_dc.envs.q_learning import DeepQNetwork
from importlib import reload

reload(gym_dc.envs.q_learning)
#reload(gym_distributioncenter)

env = gym.make("DCEnv-v0")
env.timelimit = 100
# to initialize the environment we need to reset it , returns an integer which is the initial state
#env.reset()
# total number of states in space and action space
# print('size action space, size state space')
# print(env.action_space, env.observation_space)
# print(len(env.action_space.spaces))
# print('obs space size')
# print(env.observation_space)
# for i in range(len(env.action_space.spaces)):
#     print(env.action_space.spaces[i])+


# to visualize the current state
env.render()
# returns four variables , state, reward, done, info
# variables are new state, reward, whether the environment is terminated or done and info for debugging
# size of Q table determines the granularity of state action space discretization
G = 0
alpha = 0.618
# hyperparameters
gamma = 0.0
epsilon = 0.1
statesize = env.observation_space.sample().shape[0]
action_dimensions = env.action_space.sample().shape[0]
learningrate = 0.001
batch_size = 32
# q learning network from q_learning module
agent = DeepQNetwork(env, learningrate, statesize, action_dimensions, gamma, epsilon, batch_size)


# episode in this infinite MDP ends after a defined time step (see step function)
for episode in range(1, 100):
    done = False
    G, reward = 0,0
    # resets environment when done == True
    print('resetting environment for next episode ....')
    print('episode {}'.format(episode))
    state = env.reset()
    while done != True:
        # take maximum value action in tabulated Q values
        state = np.reshape(state, [1,state.shape[0]])
        action = agent.act(state)
        nextstate, reward, done, info = env.step(action)
        # remember previous state, action, reward and done
        agent.remember(state, action, reward, nextstate, done)


        state = nextstate
        G += reward
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    # fits NN to memory
    # before training
    agent.get_stats_training()

    # after training see if change for testing
    agent.get_stats_training()
    print('Episode {} Total Reward: {}'.format(episode, G))

    #agent.replay(32)
