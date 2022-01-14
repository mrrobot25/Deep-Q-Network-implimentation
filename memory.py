import numpy as np

class MemoryBuffer(object):

	def __init__(self,max_size, input_shape, n_actions):

		self.memory_size = max_size
		self.memory_cntr = 0
		self.new_state_memory = np.zeros((self.memory_size,*input_shape),dtype = np.float32)
		self.state_memory = np.zeros((self.memory_size,*input_shape),dtype = np.float32)
		self.action_memory = np.zeros((self.memory_size),dtype = np.int64)
		self.reward_memory = np.zeros((self.memory_size),dtype = np.float32)
		self.done_memory = np.zeros((self.memory_size),dtype = np.bool)
	

	def store_transition(self,state,action,reward,new_state,done):
		
		index = self.memory_cntr % self.memory_size
		
		self.state_memory[index] = state
		self.action_memory[index] = action
		
		self.new_state_memory[index] = new_state
		self.reward_memory[index] = reward
		self.done_memory[index] = done
		self.memory_cntr += 1


	def sample_buffer(self,batch_size):

		max_mmr = min(self.memory_cntr,self.memory_size)
		batch = np.random.choice(max_mmr,batch_size, replace = False)

		states = self.state_memory[batch]
		actions = self.action_memory[batch]
		new_states = self.new_state_memory[batch]
		rewards = self.reward_memory[batch]
		dones = self.done_memory[batch]

		return states,actions, new_states, rewards, dones


