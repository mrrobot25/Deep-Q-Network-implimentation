import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os


class DeepNetwork(nn.Module):
	def __init__(self,lr,input_dims, n_actions, name, dir_path):
		super(DeepNetwork,self).__init__()
		self.dir_path = dir_path
		self.checkpoint_file = os.path.join(self.dir_path, name)	
		self.lr = lr
		
		self.conv1 = nn.Conv2d(input_dims[0],32,8,stride = 4)
		self.conv2 = nn.Conv2d(32,64,4,stride = 2)
		self.conv3 = nn.Conv2d(64,64,3,stride = 1)

		fc_dims = self.find_conv_output_dims(input_dims)
		self.fc1 = nn.Linear(fc_dims,512)
		
		self.fc2 = nn.Linear(512, n_actions)
	
		self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
		self.loss = nn.MSELoss()
		
		self.device = T.device('cpu')
		self.to(self.device)
		
	def find_conv_output_dims(self,input_dims):
		state = T.zeros(1,*input_dims)
		dims = self.conv1(state)
		dims = self.conv2(dims)
		dims = self.conv3(dims)
		
		return int(np.prod(dims.size()))


	def forward(self,state):
		
		conv1 = F.relu(self.conv1(state))
		conv2 = F.relu(self.conv2(conv1))
		conv3 = F.relu(self.conv3(conv2))

		flatten = conv3.view(conv3.size()[0], -1)
		fc1 = F.relu(self.fc1(flatten))
		
		actions = self.fc2(fc1)
		
		return actions



	def save_checkpoint(self):
		print("...........saving checkpoint............")
		
		T.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		print("........loading checkpoint..........")

		self.load_state_dict(T.load(self.checkpoint_file))













		
