import torch 
from torch import nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import yaml
import os

# input_obs_keys = ['theta_sin', 'theta_cos', 'ang_vels_z', 'linear_vels_x', 'linear_vels_y']
output_keys =  ['poses_x', 'poses_y', 'theta_sin', 'theta_cos', 'ang_vels_z', 'linear_vels_x', 'linear_vels_y']
mb_keys = output_keys + ['previous_action_steer', 'previous_action_speed']
class DynamicsModel(nn.Module):
    def __init__(self,hidden_size, dt, min_state, max_state, column_names,
                 lr=1e-4, weight_decay=1e-5):
        
        super().__init__()
        state_size = len(column_names)
        action_size = 2
        self.min_state = min_state
        self.max_state = max_state
        self.state_size, self.action_size, self.dt = state_size, action_size, dt
        A_size, B_size = state_size * state_size, state_size * action_size
        # hidden layers for A
        self.A_layers = nn.ModuleList()
        self.A_layers.append(self._make_layer(state_size + action_size, hidden_size[0]))
        for i in range(1, len(hidden_size)):
            self.A_layers.append(self._make_layer(hidden_size[i-1], hidden_size[i]))
        self.A_layers.append(self._make_layer(hidden_size[-1], A_size))

        # hidden layers for B
        self.B_layers = nn.ModuleList()
        self.B_layers.append(self._make_layer(state_size + action_size, hidden_size[0]))
        for i in range(1, len(hidden_size)):
            self.B_layers.append(self._make_layer(hidden_size[i-1], hidden_size[i]))
        self.B_layers.append(self._make_layer(hidden_size[-1], B_size))

        self.optimizer_dynamics = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.column_names = column_names
    
    def _make_layer(self, in_dim, out_dim):
        layer = nn.Linear(in_dim, out_dim)
        init.orthogonal_(layer.weight)
        return layer

    def forward(self, x, u):
        """
        Predict x_{t+1} = f(x_t, u_t)
        :param x: a batch of states
        :param u: a batch of actions
        """
        assert self.column_names is not None
        #in order to make learning easier apply to the u the clipping and scaling
        # u = torch.clip(u, -1, 1) * 0.05
        xu = torch.cat((x, u), -1)
        # set the x and y states to 0 
        x_column = self.column_names.index('poses_x')
        y_column = self.column_names.index('poses_y')

        xu[:,x_column] = 0.0  # Remove dependency in (x,y)
        xu[:,y_column] = 0.0  # Remove dependency in (x,y)
        for layer in self.A_layers[:-1]:
            xu = F.relu(layer(xu))

        A = self.A_layers[-1](xu)  # Last layer
        A = torch.reshape(A, (x.shape[0], self.state_size, self.state_size))
        
        # Reset and pass through B hidden layers
        xu = torch.cat((x, u), -1)
        xu[:, x_column] = 0.0
        xu[:, y_column] = 0.0
        for layer in self.B_layers[:-1]:  # All but the last layer
            xu = F.relu(layer(xu))
        
        B = self.B_layers[-1](xu)  # Last layer
        B = torch.reshape(B, (x.shape[0], self.state_size, self.action_size))
        
        dx = A @ x.unsqueeze(-1) + B @ u.unsqueeze(-1)
        x = x + dx.squeeze()*self.dt
        x_new = torch.clamp(x, self.min_state, self.max_state)
        return x_new
    
    def save(self, path, filename="model_based_torch_checkpoint.pth"):
        torch.save(self.state_dict(), os.path.join(path, filename))

        
    def load(self, path, filename="model_based_torch_checkpoint.pth"):
        self.load_state_dict(torch.load(os.path.join(path, filename)))

    def update(self, states, actions, next_states, masks, steps):
        self.optimizer_dynamics.zero_grad()
        pred_states = self(states, actions)
        dyn_loss = F.mse_loss(pred_states, next_states, reduction='none')
        dyn_loss = (dyn_loss).mean()
        dyn_loss.backward()
        self.optimizer_dynamics.step()
        return dyn_loss.item(), 0, 0 , 0

class ModelBased(object):
    def __init__(self,env, state_dim, action_dim, hidden_size, dt, min_state,max_state,
                 logger, dataset, fn_normalize, fn_unnormalize,obs_keys, use_reward_model=False,
                 learning_rate=1e-3, weight_decay=1e-4,target_reward=None):
        self.env = env
        self.device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.relevant_keys = output_keys
        self.state_dim, self.action_dim, self.hidden_size, self.dt = state_dim, action_dim, hidden_size, dt
        self.fn_normalize = fn_normalize
        self.fn_unnormalize = fn_unnormalize

        self.obs_keys = obs_keys
        # gather the relevant keys
        # add a 0 dim to min_states

        min_state, column_names = self.gather_relevant_keys(torch.unsqueeze(min_state, 0))
        max_state, _ = self.gather_relevant_keys(torch.unsqueeze(max_state, 0))

        self.min_state = min_state[0].to(self.device)
        self.max_state = max_state[0].to(self.device)
        print(self.min_state.shape)
        self.dynamics_model = DynamicsModel(hidden_size, dt, self.min_state, self.max_state, column_names=output_keys)
        self.dynamics_model.to(self.device)
        """
        self.dynamics_model = ModelBasedEnsemble(state_dim, 
                                                   action_dim,
                                                    hidden_size,
                                                    dt,
                                                    min_state,
                                                    max_state,
                                                    logger, fn_normalize, fn_unnormalize,
                                                    obs_keys,
                                                    learning_rate=learning_rate,
                                                    weight_decay=weight_decay,
                                                    N=3)
        """
        #config = Config(target_reward)
        #self.reward_model = GroundTruthRewardFast(dataset, 20,config)
        self.reward_config = target_reward
        self.writer=logger
        
        self.dynamics_optimizer_iterations = 0

    def update(self, original_states, actions, original_next_states, rewards, masks):
        states_in = self.env.get_specific_obs(original_states, output_keys)
        next_states_in = self.env.get_specific_obs(original_next_states, output_keys)
        curr_actions = self.env.get_specific_obs(original_next_states, ['previous_action_steer',
                                                                         'previous_action_speed'])
        loss, _ , _ , _ = self.dynamics_model.update(states_in, curr_actions, next_states_in, masks, self.dynamics_optimizer_iterations)
        # print(loss)
        self.dynamics_optimizer_iterations += 1
        return loss
    
    def __call__(self, states, actions):
        states_in = self.env.get_specific_obs(states, output_keys)
        action_states = self.env.get_specific_obs(states, ['previous_action_steer',
                                                      'previous_action_speed'])
        action_states_unnorm = self.fn_unnormalize(action_states, ['previous_action_steer',
                                                        'previous_action_speed'])
        actions = action_states_unnorm + actions
        
        # renormalize the actions
        actions = self.fn_normalize(actions, ['previous_action_steer',
                                                        'previous_action_speed'])
        next_states = self.dynamics_model(states_in, actions)
        # now we need to add the rest of the observations, these are the previous actions
        # lets append the filtered out observations
        # 1. the new previous actions
        next_states = torch.cat((next_states, actions), dim=1)

        assert states.shape == next_states.shape , "states and next states should have the same shape, but have {} and {}".format(states.shape, next_states.shape)
        return next_states
    
    def gather_relevant_keys(self, states ,keys = None):
        if keys is None:
            keys = self.relevant_keys
        # Find the indices of the relevant keys in obs_keys
        indices = [self.obs_keys.index(key) for key in keys if key in self.obs_keys]
        # Use these indices to extract the relevant columns from states
        relevant_states = states[:, indices]
        relevant_column_names = [self.obs_keys[i] for i in indices]
        return relevant_states, relevant_column_names


    def rollout(self, states, actions,get_target_action=None, horizon=10, batch_size=256, use_dynamics=True):
        with torch.no_grad():
            states_initial = states[:,0,:]
            state_batches = torch.split(states_initial, batch_size) # only do rollouts from timestep 0
            #print(states.shape)
            #print((0, horizon, states.shape[-1]))
            all_states = torch.zeros((0, horizon, states.shape[-1]))
            all_actions = torch.zeros((0, horizon, 2))
            
            for num_batch, state_batch in enumerate(state_batches):
                assert len(state_batch.shape) == 2
                assert state_batch.shape[0] <= batch_size
                all_state_batches = torch.zeros((state_batch.shape[0], 0, state_batch.shape[-1]))
                all_actions_batches = torch.zeros((state_batch.shape[0], 0, 2))
                for i in range(horizon):

                    if get_target_action is None:
                        if i == 0:
                            action = np.zeros((state_batch.shape[0],2),dtype=np.float32)
                            
                            action = torch.tensor(action)
                        else:
                            action = actions[num_batch*batch_size:batch_size*(num_batch+1),i -1,:] # (batch,2)
                            action = action.float()
                        #print(action.shape)
                        assert(action.shape[0] == state_batch.shape[0])
                        assert(action.shape[1]==2)
                    else:
 
                        action = get_target_action(state_batch, keys=mb_keys) #.to('cpu').numpy())

                        assert(action.shape[0] == state_batch.shape[0])
                        assert(action.shape[1]==2)
                        action = action
                        #make dtype float32
                        action = action.float()
                    # add the action to all_batch_actions along dim=1
                    #print("--")
                    #print(state_batch.unsqueeze(1).shape)
                    #print(action.shape)
                    #print(all_state_batches.shape)
                
                    all_actions_batches = torch.cat([all_actions_batches, action.unsqueeze(1)], dim=1)
                    all_state_batches = torch.cat([all_state_batches, state_batch.unsqueeze(1)], dim=1)
                    
                    if use_dynamics:
                        #print(state_batch.shape)
                        state_batch = self(state_batch, action)
                    elif horizon-1 != i:
                        state_batch = states[:,i+1,:]

                all_states = torch.cat([all_states, all_state_batches], dim=0)
                all_actions = torch.cat([all_actions, all_actions_batches], dim=0)
            return all_states, all_actions
    
    """
    @brief requires unnormalized states
    """
    def calculate_progress(self, states):
        from f110_orl_dataset.compute_progress import Progress, Track
        progress_obs_np = np.zeros((states.shape[0],states.shape[1],1))
        track_path = "/home/fabian/msc/f110_dope/ws_release/f1tenth_gym/gym/f110_gym/maps/Infsaal2/Infsaal2_centerline.csv"
        track = Track(track_path)
        progress = Progress(track, lookahead=200)
        pose = lambda traj_num, timestep: np.array([(states[traj_num,timestep,0],states[traj_num,timestep,1])])
        for i in range(0,states.shape[0]):
            # progress = Progress(states_inf[i,0,:])
            progress.reset(pose(i,0))
            for j in range(0,states.shape[1]):
                progress_obs_np[i,j,0] = progress.get_progress(pose(i,j))
        return progress_obs_np

    def estimate_mse_pose(self, states, get_target_action, horizon=250, batch_size=256):
        with torch.no_grad():
            inital_states = states[:,0,:]
            actions = torch.zeros((states.shape[0], 1, 2))
            all_states, all_actions = self.rollout(states, actions,get_target_action, horizon, batch_size)
            print(all_states.shape)
            print(states.shape)
            mse = torch.mean((all_states[:,:,:2] - states[:,0:horizon+1,:2])**2)
            return mse


    def estimate_returns(self, inital_states, get_target_action, horizon=250, discount=0.99, batch_size=256, plot=False):
        from f110_orl_dataset.fast_reward import MixedReward
        from f110_orl_dataset.config_new import Config
        map = "/home/fabian/msc/f110_dope/ws_release/f1tenth_gym/gym/f110_gym/maps/Infsaal2/Infsaal2_map.yaml"

        with torch.no_grad():
            states = inital_states.unsqueeze(1)
            actions = torch.zeros((states.shape[0], 1, 2))
            all_states, all_actions = self.rollout(states, actions,get_target_action, horizon, batch_size)
            config = Config(self.reward_config)
            mixedReward = MixedReward(self.env, config)
            # we need to massage all_states and all_actions into the right format
            unnormalized_states = self.fn_unnormalize(all_states, self.relevant_keys)
            progress_obs = self.calculate_progress(unnormalized_states)

            unnormalized_states = np.concatenate((unnormalized_states, progress_obs), axis=2)
            # add a zero columen, thats the format fast reward expects
            unnormalized_states = np.concatenate((unnormalized_states, np.zeros((unnormalized_states.shape[0],unnormalized_states.shape[1],1))), axis=2)
            
            num_trajectories = len(all_states)


            all_rewards = np.zeros((num_trajectories, horizon))
            for i in range(num_trajectories):
                # print("!")
                obs = unnormalized_states[i]
                action = self.env.get_specific_obs(obs, ["previous_action_steer","previous_action_speed"])
                col = ~self.is_drivable(map, obs[:,0:2])
                ter = col
                
                laser_scan = self.env.get_laser_scan(obs, 20)
                # add a dimension to all the inputs
                laser_scan =  np.expand_dims(laser_scan, axis=0)
                obs = np.expand_dims(obs, axis=0)
                action = np.expand_dims(action, axis=0)
                col = np.expand_dims(col, axis=0)
                ter = np.expand_dims(ter, axis=0)
                rewards, _ = mixedReward(obs, action,col, ter,laser_scan=laser_scan)
                all_rewards[i] = rewards
                # set rewards to zero where is not drivable / terminal
                #print(col)
                #print(col.shape)
                #print(rewards.shape)
                first_crash = np.where(col == True)[0]
                if first_crash.shape[0] > 0:
                    all_rewards[i, :first_crash[0]] = 0.0
            # for each trajectory plot the reward
            # plot all trajectories
            if plot:
                self.plot_trajectories_on_map(map, unnormalized_states)

                for i in range(num_trajectories):
                    plt.plot(all_rewards[i])
                    plt.show()
                    # also plot each trajectory on map
                    self.plot_poses_on_map(map, unnormalized_states[i], self.is_drivable(map, unnormalized_states[i][:,0:2]))
                # apply discount to the rewards
            discounted_rewards = np.zeros((num_trajectories,))
            for i in range(num_trajectories):
                discounted_rewards[i] = np.sum(all_rewards[i] * np.power(discount, np.arange(len(all_rewards[i]))))

            return np.mean(discounted_rewards), np.std(discounted_rewards)
            



    def plot_trajectories_on_map(self, yaml_path, poses):
        # Load map metadata
        with open(yaml_path, 'r') as file:
            map_metadata = yaml.safe_load(file)

        # Construct the path for the map image
        map_image_path = os.path.join(os.path.dirname(yaml_path), map_metadata['image'])
        map_image = Image.open(map_image_path)
        map_array = np.array(map_image)

        # Display the map
        plt.imshow(map_image, cmap='gray')

        # Map parameters
        resolution = map_metadata['resolution']
        origin = map_metadata['origin']

        # Number of trajectories
        num_trajectories = poses.shape[0]

        # Generate color map
        colors = cm.rainbow(np.linspace(0, 1, num_trajectories))

        # Plot each trajectory
        for i in range(num_trajectories):
            # Convert poses to pixel coordinates, invert y-axis
            pixel_poses = poses[i].copy()
            pixel_poses[:, 0] = (pixel_poses[:, 0] - origin[0]) / resolution
            pixel_poses[:, 1] = map_array.shape[0] - ((pixel_poses[:, 1] - origin[1]) / resolution)

            # Plot trajectory
            plt.plot(pixel_poses[:, 0], pixel_poses[:, 1], color=colors[i], label=f'Trajectory {i+1}')

        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.title('Trajectories on Map')
        plt.legend()
        plt.show()

    def is_drivable(self,yaml_path, poses):
        # Load map metadata
        with open(yaml_path, 'r') as file:
            map_metadata = yaml.safe_load(file)

        # Load the map image
        map_image_path = os.path.join(os.path.dirname(yaml_path), map_metadata['image'])
        map_image = Image.open(map_image_path)
        map_array = np.array(map_image)

        # Map parameters
        resolution = map_metadata['resolution']  # meters per pixel
        origin = map_metadata['origin']  # [x, y, theta]
        occupied_thresh = map_metadata['occupied_thresh']

        # Convert poses to pixel coordinates, invert y-axis
        pixel_poses = poses.copy()
        pixel_poses[:, 0] = (pixel_poses[:, 0] - origin[0]) / resolution
        pixel_poses[:, 1] = map_array.shape[0] - 1 - ((pixel_poses[:, 1] - origin[1]) / resolution)
        pixel_poses = pixel_poses.astype(int)

        # Check bounds
        in_bounds = (pixel_poses[:, 0] >= 0) & (pixel_poses[:, 0] < map_array.shape[1]) & \
                    (pixel_poses[:, 1] >= 0) & (pixel_poses[:, 1] < map_array.shape[0])

        # Check if the area is drivable (not occupied)
        # Assuming drivable area is white (high pixel value)
        drivable_threshold = int(255 * (1 - occupied_thresh))
        is_drivable = np.array([map_array[pixel_y, pixel_x] > drivable_threshold if in_bounds[i] else False 
                                for i, (pixel_x, pixel_y) in enumerate(pixel_poses)])
        first_false = np.where(is_drivable == False)[0]
        #print(first_false)
        if len(first_false) > 0:
            first_false = first_false[0]
            is_drivable[first_false:] = False
        return is_drivable
    
    def plot_poses_on_map(self, yaml_path, poses, is_drivable):
        import matplotlib.patches as mpatches
        # Load map metadata
        with open(yaml_path, 'r') as file:
            map_metadata = yaml.safe_load(file)

        # Construct the path for the map image
        map_image_path = os.path.join(os.path.dirname(yaml_path), map_metadata['image'])
        map_image = Image.open(map_image_path)
        map_array = np.array(map_image)

        # Display the map
        plt.imshow(map_image, cmap='gray')

        # Map parameters
        resolution = map_metadata['resolution']
        origin = map_metadata['origin']

        # Convert poses to pixel coordinates, invert y-axis
        pixel_poses = poses.copy()
        pixel_poses[:, 0] = (pixel_poses[:, 0] - origin[0]) / resolution
        pixel_poses[:, 1] = map_array.shape[0] - ((pixel_poses[:, 1] - origin[1]) / resolution)

        # Plot each pose
        for pose, drivable in zip(pixel_poses, is_drivable):
            if drivable:
                plt.plot(pose[0], pose[1], 'o', color='green')  # Drivable: green circle
            else:
                plt.plot(pose[0], pose[1], 'x', color='red')  # Not drivable: red x

        # Add legend
        drivable_patch = mpatches.Patch(color='green', label='Drivable')
        not_drivable_patch = mpatches.Patch(color='red', label='Not Drivable')
        plt.legend(handles=[drivable_patch, not_drivable_patch])

        plt.gca().invert_yaxis()  # Invert y-axis to match ROS map orientation
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.title('Poses on Map')
        plt.show()
    
    def save(self, save_path, filename="model_based2_torch_checkpoint.pth"):
        """
        Save the model's state dictionaries.
        
        Args:
        - save_path (str): The directory path where the model should be saved.
        - filename (str, optional): The name of the checkpoint file. Defaults to "model_based2_checkpoint.pth".
        """
        
        # Ensure the directory exists
        os.makedirs(save_path, exist_ok=True)

        # Define the checkpoint

        self.dynamics_model.save(save_path, filename)
    
    def load(self, save_path, filename="model_based2_torch_checkpoint.pth"):
        """
        Load the model's state dictionaries.
        
        Args:
        - save_path (str): The directory path where the model should be loaded from.
        - filename (str, optional): The name of the checkpoint file. Defaults to "model_based2_checkpoint.pth".
        """
        
        self.dynamics_model.load(save_path, filename)