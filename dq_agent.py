import numpy as np
import torch as T
from memory import MemoryBuffer
from network import DeepNetwork


class DQAgent(object):
	def __init__(self,gamma,epsilon,lr,n_actions,input_dims,
		mem_size,batch_size,eps_min = 0.01, eps_dec=5e-7,
		replace=100,algo=None,env_name=None,
		 chkpt_dir='chpt/'):
		
		self.gamma = gamma
		self.lr = lr
		self.epsilon = epsilon
		self.n_actions = n_actions
		self.input_dims = input_dims
		self.mem_size = mem_size
		self.batch_size = batch_size
		self.eps_min = eps_min	
		self.eps_dec = eps_dec
		self.replace_target_counter = replace
		self.algo = algo
		self.env_name = env_name
		self.chkpt_dir = chkpt_dir
		
		self.action_space = [i for i in range(n_actions)]
		self.learn_cntr = 0
		self.memory = MemoryBuffer(mem_size,input_dims,n_actions)

		self.Q = DeepNetwork(self.lr, self.input_dims, self.n_actions,
						name =self.env_name+'_'+self.algo+'q_eval',
						dir_path =self.chkpt_dir )

		self.Q_next = DeepNetwork(self.lr, self.input_dims, self.n_actions,
						name =self.env_name+'_'+self.algo+'q_next',
						dir_path =self.chkpt_dir )


	def choose_action(self,observation):
		
		if np.random.random() > self.epsilon:
			state = T.tensor([observation],dtype = T.float).to(self.Q.device)
			actions = self.Q.forward(state)
			action = T.argmax(actions).item()

		else:
			action = np.random.choice(self.action_space)
			

		return action	

	def store_transition(self,state, action,reward,state_,done):
		self.memory.store_transition(state,action,reward,state_,done)
	

	def sample_memory(self,batch_size):
		states,actions,new_states, rewards,dones = self.memory.sample_buffer(batch_size)
		
		states = T.tensor(states).to(self.Q.device)
		actions = T.tensor(actions).to(self.Q.device)
		new_states = T.tensor(new_states).to(self.Q.device)
		rewards = T.tensor(rewards).to(self.Q.device)
		dones = T.tensor(dones).to(self.Q.device)
		
		return states,actions, new_states, rewards, dones

	def target_network(self):
		if self.learn_cntr % self.replace_target_counter == 0:
			self.Q_next.load_state_dict(self.Q.state_dict())
			

	def decreament_epsilon(self):
		
		self.epsilon = self.epsilon - self.epsilon \
					if self.epsilon > self.eps_min else self.eps_min
					

	def save_models(self):
		
		self.Q.save_checkpoint()
		self.Q_next.save_checkpoint()

	def load_models(self):
		self.Q.load_checkpoint()
		self.Q_next.load_checkpoint()

	def learn(self):
		if self.memory.memory_cntr < self.batch_size:
			return
		
		self.Q.optimizer.zero_grad()
		self.target_network()
		states, actions,new_states, rewards, dones = self.sample_memory(self.batch_size)
	
		indices = np.arange(self.batch_size)
		q_pred = self.Q.forward(states)[indices,actions]
		
		q_next = self.Q_next(new_states).max(axis=1)[0]

		q_next[dones] = 0.0

		q_target = rewards + self.gamma*q_next
		loss = self.Q.loss(q_pred,q_next).to(self.Q.device)
		loss.backward()
		self.Q.optimizer.step()
		self.learn_cntr +=1

		self.decreament_epsilon()
































