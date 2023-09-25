from collections import deque
import pickle
import os
import random
import numpy as np

class ReplayMemory:
	def __init__(self, memory_size):
		self.buffer = deque()
		self.memory_size = memory_size
	
	def append(self, pre_state, action, reward, post_state, terminal):
		self.buffer.append((pre_state, action, reward, post_state, terminal))
		if len(self.buffer) >= self.memory_size:
			self.buffer.popleft()
	
	def sample(self, size):
		length = 3
		state_his = list()
		next_state_his = list()
		# index = random.shuffle(range(self.buffer.__len__()))
		buff_new = self.buffer.copy()
		for i in range(length):
			buff_new.popleft()
		# buff = self.buffer
		minibatch = random.sample(list(enumerate(buff_new)),size)
		# minibatch = random.sample(self.buffer, size)
		states = [data[0] for i, data in minibatch]
		actions = np.array([data[1] for i, data in minibatch])
		rewards = np.array([data[2] for i, data in minibatch])
		next_states = [data[3] for i, data in minibatch]
		terminals = np.array([data[4] for i, data in minibatch])
		index = np.array([i for i, data in minibatch]) + length
		for i in index:
			pre_states = list()
			for j in range(length):
				pre_obser = (self.buffer[i-j])[0]
				pre_states.append(pre_obser)
			state_his.append(pre_states)
		# state_his = [data[0] for data in state_his]
		for i in index:
			pre_next = list()
			for j in range(length):
				pre_obser = (self.buffer[i - j])[0]
				pre_next.append(pre_obser)
			next_state_his.append(pre_next)
		# next_state_his = [data[0] for data in next_state_his]
		return states, actions, rewards, next_states, terminals, state_his, next_state_his

	def save(self, dir):
		file = os.path.join(dir, 'replaymemory.pickle')
		with open(file, 'wb') as f:
			pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

	def load(self, dir):
		file = os.path.join(dir, 'replaymemory.pickle')
		with open(file, 'rb') as f:
			memory = pickle.load(f)
		return memory