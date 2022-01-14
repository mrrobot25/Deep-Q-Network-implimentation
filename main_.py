import torch as T
import numpy as np
from dq_agent import DQAgent
from utility import make_env
import matplotlib.pyplot as plt

if __name__ == '__main__':
	
	env = make_env('PongNoFrameskip-v4')

	best_score = -np.inf
	load_checkpoint = False
	n_games = 500
	agent = DQAgent(gamma=0.99,epsilon=1.0, lr=0.0001,input_dims=(env.observation_space.shape),
					n_actions=env.action_space.n, mem_size=20000, eps_min=0.1,
					 batch_size=32, replace=1000, 
					eps_dec=1e-5,chkpt_dir='models/', algo='DQAgent',
					env_name='PongNoFrameskip-v4')

	file_name = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' + '_' + str(n_games) + 'games'

	figure_file = 'plots/' + file_name + '.png'
	n_steps = 0 
	scores, eps_his = [], []

	for i in range(n_games):
		done = False
		observation = env.reset()
		score = 0

		while not done:
			action = agent.choose_action(observation)
			observation_, reward, done, info = env.step(action)
			score += reward

			if not load_checkpoint:
				agent.store_transition(observation,action, reward, observation_, done)
				agent.learn()
			observation = observation_
			n_steps +=1

		scores.append(score)

		avg_score = np.mean(scores[-100:])
		

		if avg_score > best_score:
			if not load_checkpoint:
				agent.save_models()
			best_score = avg_score

	print('episode: ', i,'score: ', score,
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

	plt.plot(scores)
		
			
