from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
from random import sample
from collections import deque

# Q learning implementation
# Initialize Q(s,a) arbitrarily
# Repeat (for each generation):
# 	Initialize state s
# 	While (s is not a terminal state):
# 		Choose a from s using policy derived from Q
# 		Take action a, observe r, s'
# 		Q(s,a) += alpha * (r + gamma * max,Q(s') - Q(s,a))
# 		s = s'
#
# TODO: doesn't seem to train right now.
# 1. receive state and feed to NN and network that decides best action to execute
# 2. network trained millions of times via Q learning to maximize future expected reward

#https://keon.io/deep-q-learning/


class DeepQNetwork:

    def __init__(self, env, learningrate, state_size, action_dimensions, gamma, epsilon, batch_size):
        self.action_dimensions = action_dimensions
        self.state_size = state_size
        self.learningrate = learningrate
        # reward discount
        self.gamma = gamma
        # exploration rate
        self.epsilon = epsilon
        # memory contains state, action, reward, nextstate, done
        self.memory = deque(maxlen=2000)
        self.model = self._buildmodel()
        self.env = env
        self.time = self.env.time
        self.action_indizes = {}
        self.max_ind = 0

    def fit_NN(self, state, reward_value):
        ''' fit reward value from a certain state after carrying out an action decreases gap between prediction to target according to learning rate'''
        model.fit(state, reward_value, epochs=1, verbose=0)

    def predict_reward(self, state):
        '''predict reward in that state '''
        model.predict(state)

    def target_val(self, nextstate, reward):
        ''' compute discounted reward as value function target value estimate '''
        target = reward + self.gamma * np.amax(model.predict(nextstate))

    def remember(self, state, action, reward, nextstate, done):
        ''' function to append state, action, next state experiences to retrain model on previous experiences
        parameters:
        done = boolean if in final state '''
        self.memory.append((state, action, reward, nextstate, done))

    def replay(self, batch_size):
        ''' experience replay based on old experiences '''
        minibatch = sample(self.memory, batch_size)
        for state, action, reward, nextstate, done in minibatch:
            target = reward
            if not done:
                nextstate = np.reshape(nextstate, [1,nextstate.shape[0]])
                # reward + distcounted future reward
                target = reward + self.gamma * np.amax(self.model.predict(nextstate)[0])
            # make agent approximately map current state to future discounted reward
            target_f = self.model.predict(state)
            # reset target value for action to target
            #action_index = self.get_action_index(action)
            # print('target predicted')
            # print(target_f)
            # target_f[0][action_index] = target
            # train NN with state and approximate reward
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def get_action_index(self, action):
        if str(action) in self.action_indizes.keys():
            index = self.action_indizes.get(str(action))
            return index
        else:
            self.max_ind += 1
            index = self.max_ind
            self.action_indizes[str(action)] = index
            return index


    def act(self, state):
        ''' randomly select action at first by certain percentage corresponding to epsilon '''
        if np.random.rand() <= self.epsilon:
            # sample a random action from action space
            #print('exploration....')
            return self.env.action_space.sample()
        # print('state')
        # print(state)
        # print(state.shape)
        # predict the reward value based on given state
        act_values = self.model.predict(state)
        # pick action based on maximum reward
        # print('predicted action')
        #print(act_values)
        return act_values[0]

    def _buildmodel(self):
       NN = Sequential()
       # input layer takes state
       NN.add(Dense(24, input_dim=self.state_size, activation='relu'))
       # hidden layer
       NN.add(Dense(24, activation='relu'))
       # output layer - action dimensionality, value of input state action
       # TODO: remove hardcoded action size
       NN.add(Dense(4, activation='linear'))
       #
       NN.compile(loss='mse', optimizer=Adam(lr=self.learningrate))
       return NN

    def get_stats_training(self):
        ''' function to examine pogress  '''
        first_weights = self.model.layers[0].get_weights()[0]
        first_bias = self.model.layers[0].get_weights()[1]
        second_weights = self.model.layers[1].get_weights()[0]
        second_bias = self.model.layers[1].get_weights()[1]
        print('variance weights first layer  {}'.format(np.var(first_weights)))
        print('mean weights first layer  {}'.format(np.mean(first_weights)))
        print('variance weights second layer  {}'.format(np.var(first_weights)))
        print('mean weights second layer  {}'.format(np.mean(first_weights)))
