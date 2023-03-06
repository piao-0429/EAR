import os
import glob
import time
from datetime import datetime
from PIL import Image
import csv
import sys

sys.path.append('.') 
import torch
import os
import time
import os.path as osp
import numpy as np
import torch.nn.functional as F
from torchrl.utils import get_args
from torchrl.utils import get_params
from torchrl.env import get_env
from torchrl.env.mujoco_envs import ENVS

import torchrl.policies as policies
import torchrl.networks as networks


args = get_args()
params = get_params(args.config)


LTE_1 = np.array([1,0,0])
LTE_2 = np.array([0,1,0])

env = ENVS[params['env_name']](mode='interpolate')

lse_shape = params['lse_shape']
lte_shape = params['lte_shape']
alpha = params['alpha']

params['state_net']['base_type']=networks.MLPBase
params['action_net']['base_type']=networks.MLPBase


pf_state = networks.NormNet(
	input_shape=env.observation_space.shape[0], 
	output_shape=lse_shape,
	**params['state_net']
)

pf_action=policies.EmergentActionRepresentationGuassianContPolicy(
	input_shape = lse_shape + lte_shape,
	output_shape = 2 * env.action_space.shape[0],
	**params['action_net'] 
)
experiment_id = str(args.id)
seed = args.seed

model_dir = 'log/' + experiment_id + '/' + params['env_name'] + '/' + str(args.seed) + '/model/'
experiment_id = experiment_id + '_interpolate'

pf_state.load_state_dict(torch.load(model_dir + 'model_pf_state_best.pth', map_location='cpu'))
pf_action.load_state_dict(torch.load(model_dir + 'model_pf_action_best.pth', map_location='cpu'))


def create_image_dir(exp_id, seed):
   
	gif_images_dir = 'gif_images' + '/'
	if not os.path.exists(gif_images_dir):
		os.makedirs(gif_images_dir)

	gif_images_dir = gif_images_dir + '/' + exp_id + '/'
	if not os.path.exists(gif_images_dir):
		os.makedirs(gif_images_dir)

	gif_images_dir = gif_images_dir + '/' + env_name + '/'
	if not os.path.exists(gif_images_dir):
		os.makedirs(gif_images_dir)
  
	gif_images_dir = gif_images_dir + '/' + str(seed) + '/'
	if not os.path.exists(gif_images_dir):
		os.makedirs(gif_images_dir)

	gif_dir = 'gifs' + '/'
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	gif_dir = gif_dir + '/' + exp_id + '/'
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	gif_dir = gif_dir + '/' + env_name  + '/'
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)
	
	gif_dir = gif_dir + '/' + str(seed)  + '/'
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)
	
	return gif_images_dir

def evaluate(env_name, max_ep_len=200):

	device = torch.device('cuda:{}'.format(args.device) if args.cuda else 'cpu')
	
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	if args.cuda:
		torch.backends.cudnn.deterministic=True

	gif_images_dir = create_image_dir(experiment_id, seed)
        
	lte = LTE_1 * alpha + LTE_2 * (1-alpha)
	lte = torch.from_numpy(lte).unsqueeze(0).float()
	lte = F.normalize(lte, dim=-1)
 
	ob = env.reset()
	sum_vel_x = 0
	sum_vel_y = 0
	
	with torch.no_grad():
		for t in range(max_ep_len):
			lse = pf_state.forward(torch.Tensor( ob ).to('cpu').unsqueeze(0))
			out = pf_action.explore(lse,lte)
			act = out['action']
			act = act.detach().numpy()
			next_ob, _, done, info = env.step(act)
			sum_vel_x += info['x_velocity']

			if params['env_name'] == 'Ant-Dir':
				sum_vel_y += info['y_velocity']
	
			img = env.render(mode = 'rgb_array')
			img = Image.fromarray(img)
			img.save(gif_images_dir + experiment_id + '_' + str(seed) + '_' + str(t).zfill(6) + '.jpg')
			
			ob = next_ob
			if done:
				break

		if params['env_name'] == 'Ant-Dir':
			x = info['x_position']
			y = info['y_position']
			dir = np.arctan(y/x)/ np.pi * 180
			if x<0 and y>0:
				dir+=180
			elif x<0 and y<0:
				dir+=180
			elif x>0 and y<0:
				dir+=360
			print("Interpolated Result:", dir)

		else:
			print("Interpolated Result:", sum_vel_x/t)

	env.close()

def save_gif(env_name, total_timesteps=200, step=1, frame_duration=60):  
     
	gif_images_dir = 'gif_images/' + experiment_id + '/' + env_name + '/' + str(seed) + '/'
	gif_images_path = gif_images_dir + '/*.jpg'

	gif_dir = 'gifs'
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	gif_dir = gif_dir + '/' + experiment_id + '/' + env_name + '/' + str(seed) + '/'
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)
	gif_path = gif_dir + experiment_id + '_' + str(seed) + '.gif'

	img_path = sorted(glob.glob(gif_images_path))
	img_path = img_path[:total_timesteps]
	img_path = img_path[::step]

	img, *imgs = [Image.open(f) for f in img_path]
	img.save(fp=gif_path, format='GIF', append_images=imgs, save_all=True, optimize=True, duration=frame_duration, loop=0)
	print('saved gif at : ', gif_path)



if __name__ == '__main__':
	env_name = params['env_name']
	evaluate(env_name)
	save_gif(env_name, total_timesteps = 200, step = 1, frame_duration = 60)

