#!/usr/bin/env python3

import numpy as np, numpy.random as nr, gym
import matplotlib.pyplot as plt
np.set_printoptions(precision=3)


from crawler_env import CrawlingRobotEnv

env = CrawlingRobotEnv()

print("We can inspect the observation space and action space of this Gym Environment")
print("-----------------------------------------------------------------------------")
print("Action space:", env.action_space)
print("It's a discrete space with %i actions to take" % env.action_space.n)
print("Each action corresponds to increasing/decreasing the angle of one of the joints")
print("We can also sample from this action space:", env.action_space.sample())
print("Another action sample:", env.action_space.sample())
print("Another action sample:", env.action_space.sample())
print("Observation space:", env.observation_space, ", which means it's a 9x13 grid.")
print("It's the discretized version of the robot's two joint angles")

env = CrawlingRobotEnv(
    render=True, # turn render mode on to visualize random motion
)

#standard procedure for interfacing with a Gym environment
cur_state = env.reset() # reset environment and get initial state
ret = 0.
done = False
i = 0
while not done:
    action = env.action_space.sample() # sample an action randomly
    next_state, reward, done, info = env.step(action)
    ret += reward
    cur_state = next_state
    i += 1
    if i == 500:
        break # for the purpose of this visualization, let's only run for 1500 steps
        # also note the GUI won't close automatically
# you can close the visualization GUI with the following method
env.close_gui()

from collections import defaultdict
import random

# dictionary that maps from state, s, to a numpy array of Q values [Q(s, a_1), Q(s, a_2) ... Q(s, a_n)]
#   and everything is initialized to 0.
q_vals = defaultdict(lambda: np.array([0. for _ in range(env.action_space.n)]))

#print(env.action_space.n)
print("Q-values for state (0, 0): %s" % q_vals[(0, 0)], "which is a list of Q values for each action")
print("As such, the Q value of taking action 3 in state (1,2), i.e. Q((1,2), 3), can be accessed by q_vals[(1,2)][3]:", q_vals[(1,2)][3])

def eps_greedy(q_vals, eps, state):
    """
    Inputs:
        q_vals: q value tables
        eps: epsilon
        state: current state
    Outputs:
        random action with probability of eps; argmax Q(s, .) with probability of (1-eps)
    """
    # you might want to use random.random() to implement random exploration
    #   number of actions can be read off from len(q_vals[state])
    import random
    p = random.random()
    if p < eps:
        return random.randint(0,len(q_vals[state])-1)
    else:
        return np.argmax(q_vals[state])

# test 1
dummy_q = defaultdict(lambda: np.array([0. for _ in range(env.action_space.n)]))
test_state = (0, 0)
dummy_q[test_state][0] = 10.
trials = 100000
sampled_actions = [
    int(eps_greedy(dummy_q, 0.3, test_state))
    for _ in range(trials)
]
freq = np.sum(np.array(sampled_actions) == 0) / trials
tgt_freq = 0.3 / env.action_space.n + 0.7
if np.isclose(freq, tgt_freq, atol=1e-2):
    print("Test1 passed")
else:
    print("Test1: Expected to select 0 with frequency %.2f but got %.2f" % (tgt_freq, freq))

# test 2
dummy_q = defaultdict(lambda: np.array([0. for _ in range(env.action_space.n)]))
test_state = (0, 0)
dummy_q[test_state][2] = 10.
trials = 100000
sampled_actions = [
    int(eps_greedy(dummy_q, 0.5, test_state))
    for _ in range(trials)
]
freq = np.sum(np.array(sampled_actions) == 2) / trials
tgt_freq = 0.5 / env.action_space.n + 0.5
if np.isclose(freq, tgt_freq, atol=1e-2):
    print("Test2 passed")
else:
    print("Test2: Expected to select 2 with frequency %.2f but got %.2f" % (tgt_freq, freq))

def q_learning_update(gamma, alpha, q_vals, cur_state, action, next_state, reward):
    """
    Inputs:
        gamma: discount factor
        alpha: learning rate
        q_vals: q value table
        cur_state: current state
        action: action taken in current state
        next_state: next state results from taking `action` in `cur_state`
        reward: reward received from this transition

    Performs in-place update of q_vals table to implement one step of Q-learning
    """
    actionIdx = np.argmax(q_vals[next_state])
    target = reward + gamma * q_vals[next_state][actionIdx]
    #q_vals[cur_state][action] = (1-alpha) * q_vals[cur_state][action] + alpha * target
    q_vals[cur_state][action] += alpha * (target - q_vals[cur_state][action])

# testing your q_learning_update implementation
dummy_q = q_vals.copy()
test_state = (0, 0)
test_next_state = (0, 1)
dummy_q[test_state][0] = 10.
dummy_q[test_next_state][1] = 10.
q_learning_update(0.9, 0.1, dummy_q, test_state, 0, test_next_state, 1.1)
tgt = 10.01
if np.isclose(dummy_q[test_state][0], tgt,):
    print("Test passed")
else:
    print("Q(test_state, 0) is expected to be %.2f but got %.2f" % (tgt, dummy_q[test_state][0]))

# now with the main components tested, we can put everything together to create a complete q learning agent

env = CrawlingRobotEnv()
q_vals = defaultdict(lambda: np.array([0. for _ in range(env.action_space.n)]))
gamma = 0.9
alpha = 0.1
eps = 0.5
cur_state = env.reset()

def greedy_eval():
    """evaluate greedy policy w.r.t current q_vals"""
    test_env = CrawlingRobotEnv(horizon=np.inf)
    prev_state = test_env.reset()
    ret = 0.
    done = False
    H = 100
    for i in range(H):
        action = np.argmax(q_vals[prev_state])
        state, reward, done, info = test_env.step(action)
        ret += reward
        prev_state = state
    return ret / H

for itr in range(300000):
    action = eps_greedy(q_vals, eps, cur_state)
    next_state, reward, done, info = env.step(action)
    q_learning_update(gamma, alpha, q_vals, cur_state, action, next_state, reward)
    cur_state = next_state
    if itr % 50000 == 0: # evaluation
        print("Itr %i # Average speed: %.2f" % (itr, greedy_eval()))

# at the end of learning your crawler should reach a speed of >= 3
env = CrawlingRobotEnv(render=True, horizon=500)
prev_state = env.reset()
ret = 0.
done = False
while not done:
    action = np.argmax(q_vals[prev_state])
    state, reward, done, info = env.step(action)
    ret += reward
    prev_state = state

# you can close the visualization GUI with the following method
env.close_gui()
