import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def tf_to_torch(tf_tensor):
    """
    Convert a TensorFlow tensor to a PyTorch tensor.
    """
    # Convert TensorFlow tensor to NumPy
    numpy_array = tf_tensor.numpy()
    
    # Convert NumPy array to PyTorch tensor
    torch_tensor = torch.from_numpy(numpy_array)
    
    return torch_tensor

import torch.nn.init as init


"""
A network that takes in the current x,y state and outputs the sin and cos of the progress
"""
class ProgressNetwork(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=2):
        super(ProgressNetwork, self).__init__()
        # Define the architecture here
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_size, hidden_size) # Second fully connected layer
        self.fc3 = nn.Linear(hidden_size, output_size) # Output layer

    def forward(self, x):
        # Define the forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output layer without an activation function
        x = self.fc3(x)
        # Normalize the output to lie on the unit circle
        # This enforces the sin^2(theta) + cos^2(theta) = 1 constraint
        x = F.normalize(x, p=2, dim=1)
        return x

class DynamicsModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, dt, min_state, max_state):
        
        super().__init__()
        self.min_state = min_state
        self.max_state = max_state
        self.state_size, self.action_size, self.dt = state_size, action_size, dt
        A_size, B_size = state_size * state_size, state_size * action_size
        #self.A1 = nn.Linear(state_size + action_size, hidden_size)
        #self.A2 = nn.Linear(hidden_size, A_size)
        #self.B1 = nn.Linear(state_size + action_size, hidden_size)
        #self.B2 = nn.Linear(hidden_size, B_size)
        self.A_layers = nn.ModuleList()
        self.A_layers.append(self._make_layer(state_size + action_size, hidden_size[0]))
        for i in range(1, len(hidden_size)):
            self.A_layers.append(self._make_layer(hidden_size[i-1], hidden_size[i]))
        self.A_layers.append(self._make_layer(hidden_size[-1], A_size))

        # Construct hidden layers for B
        self.B_layers = nn.ModuleList()
        self.B_layers.append(self._make_layer(state_size + action_size, hidden_size[0]))
        for i in range(1, len(hidden_size)):
            self.B_layers.append(self._make_layer(hidden_size[i-1], hidden_size[i]))
        self.B_layers.append(self._make_layer(hidden_size[-1], B_size))
        
        self.STATE_X, self.STATE_Y = 0, 1
        self.STATE_PROGRESS_SIN = 9
        self.STATE_PROGRESS_COS = 10
        # maybe also remove this
        self.STATE_PREVIOUS_ACTION_STEERING = 7
        self.STATE_PREVIOUS_ACTION_VELOCITY = 8

        self.progress_model = ProgressNetwork(input_size=2, hidden_size=64, output_size=2)
    
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
        #in order to make learning easier apply to the u the clipping and scaling
        # u = torch.clip(u, -1, 1) * 0.05
        xu = torch.cat((x, u), -1)
        xu[:, self.STATE_X:self.STATE_Y+1] = 0  # Remove dependency in (x,y)
        xu[:, self.STATE_PROGRESS_SIN:self.STATE_PROGRESS_COS+1] = 0  # Remove dependency in progress
        # calculate the actual input action by using the previous action ob + the cliped and scaled action
        # but would also need unnormalization and then renomalization, so not at this point in time
        
        #A = self.A2(F.relu(self.A1(xu)))
        #A = torch.reshape(A, (x.shape[0], self.state_size, self.state_size))
        #B = self.B2(F.relu(self.B1(xu)))
        #B = torch.reshape(B, (x.shape[0], self.state_size, self.action_size))
        for layer in self.A_layers[:-1]:  # All but the last layer
            xu = F.relu(layer(xu))
        A = self.A_layers[-1](xu)  # Last layer
        A = torch.reshape(A, (x.shape[0], self.state_size, self.state_size))
        # Reset and pass through B hidden layers
        xu = torch.cat((x, u), -1)
        xu[:, self.STATE_X:self.STATE_Y+1] = 0
        xu[:, self.STATE_PROGRESS_SIN:self.STATE_PROGRESS_COS+1] = 0
        for layer in self.B_layers[:-1]:  # All but the last layer
            xu = F.relu(layer(xu))
        B = self.B_layers[-1](xu)  # Last layer
        B = torch.reshape(B, (x.shape[0], self.state_size, self.action_size))
        
        dx = A @ x.unsqueeze(-1) + B @ u.unsqueeze(-1)
        x = x + dx.squeeze()*self.dt

        # now apply the progress model to x
        progress = self.progress_model(x[:, self.STATE_X:self.STATE_Y+1])
        # Create a mask for the indices that you want to update
        x_new = x.clone()
        x_new[:, self.STATE_PROGRESS_SIN:self.STATE_PROGRESS_COS+1] = progress
        
        #x[:, self.STATE_PROGRESS_SIN:self.STATE_PROGRESS_COS+1] = progress
        # clip the state between min and maxstate
        x_new = torch.clamp(x_new, self.min_state, self.max_state)
        return x_new


class DynamicsModelPolicy(object):
    def __init__(self, state_dim, action_dim, hidden_size, dt,writer,
                 learning_rate=1e-3, weight_decay=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dynamics_model = DynamicsModel(state_dim, action_dim, hidden_size, dt)
        self.dynamics_model.to(self.device)
        self.optimizer_dynamics = optim.Adam(self.dynamics_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.writer=writer

    def update(self, states,actions,next_states, step):
        self.optimizer_dynamics.zero_grad()  # Reset gradients
        pred_states = self.dynamics_model(states, actions)
        dyn_loss = F.mse_loss(pred_states, next_states, reduction='none')
        dyn_loss = (dyn_loss).mean()
        dyn_loss.backward()  # Compute gradients
        self.optimizer_dynamics.step()
        self.writer.add_scalar('cond/train/dyn_loss', dyn_loss.item(), global_step=step)

    def __call__(self, states, actions):
        return self.dynamics_model(states, actions)

    def save(self, path, filename):
        torch.save(self.dynamics_model.state_dict(), os.path.join(path, filename))


class RewardModel(nn.Module):
    """A class that implements a reward model."""

    def __init__(self, state_dim, action_dim):
        super(RewardModel, self).__init__()

        # Define the neural network layers
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 1)

        # Initialize the layers with orthogonal initialization
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.fc3.weight)
        nn.init.orthogonal_(self.fc4.weight)
        nn.init.orthogonal_(self.fc5.weight)

    def forward(self, x, a):
        x_a = torch.cat([x, a], dim=-1)
        x = F.relu(self.fc1(x_a))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x).squeeze(-1)

from f110_orl_dataset.reward import MixedReward
import gymnasium as gym

from f110_orl_dataset.normalize_dataset import Normalize

class GroundTruthReward(object):
    def __init__(self, map, dataset,  subsample_laser, **reward_config):
       
        self.env = gym.make('f110_with_dataset-v0',
        # only terminals are available as of right now 
        **dict(name='f110_with_dataset-v0',
            config = dict(map="Infsaal", num_agents=1,
            params=dict(vmin=0.5, vmax=2.0)),
              render_mode="human")
    )
        
        self.reward = MixedReward(self.env, self.env.track, **reward_config)
        self.model_input_normalizer = Normalize()
        self.dataset= dataset
        self.subsample_laser = subsample_laser
    
    def __call__(self, observation, action):
        # the input observation and action are tensors
        collision = False
        done = False
        #print(observation)
        #print(action)
        states = self.dataset.unnormalize_states(observation)
        #print(states)
        # need to add the laser observation
        #print("unnormalized")
        #print(states)
        observation_dict = self.model_input_normalizer.unflatten_batch(states)
        laser_scan = self.env.get_laser_scan(states, self.subsample_laser) # TODO! rename f110env to dataset_env
        laser_scan = self.model_input_normalizer.normalize_laser_scan(laser_scan)
        observation_dict['lidar_occupancy'] = laser_scan
        # have to unsqueeze the batch dimension
        observation_dict = {key: value.squeeze(0) if value.ndim > 1 and value.shape[0] == 1 else value for key, value in observation_dict.items()}
        #print("previous action:", observation_dict["previous_action"])
        #print("current action:", np.clip(action, -1, 1) * 0.05)
        raw_action = observation_dict['previous_action'] + np.clip(action, -1, 1) * 0.05
        #print(f"calcuated raw_action {raw_action}")
        #print(np.array([observation_dict['poses_x'][0], observation_dict['poses_y'][0]]))
        reward, _ = self.reward(observation_dict, raw_action, 
                                      collision, done)
        # print("R:", reward)
        return reward
    
    def reset(self, pose , velocity=1.5):
        # add 9 empty states such that we can use the unnormalize function
        states = np.zeros((1, 11), dtype=np.float32)
        states[0, 0] = pose[0]
        states[0, 1] = pose[1]
        pose = self.dataset.unnormalize_states(states)[0][:2]
        self.reward.reset(pose, velocity=velocity)

progress_config = {
    "collision_penalty": 0.0,
    "progress_weight": 1.0,
    "raceline_delta_weight": 0.0,
    "velocity_weight": 0.0,
    "steering_change_weight": 0.0,
    "velocity_change_weight": 0.0,
    "pure_progress_weight": 0.0,
    "min_action_weight" : 0.0,
    "min_lidar_ray_weight" : 0.0, #missing
    "inital_velocity": 1.5,
    "normalize": False,
}

raceline_config = {
    "collision_penalty": 0.0,
    "progress_weight": 0.0,
    "raceline_delta_weight": 1.0,
    "velocity_weight": 0.0,
    "steering_change_weight": 0.0,
    "velocity_change_weight": 0.0,
    "pure_progress_weight": 0.0,
    "min_action_weight" : 0.0,
    "min_lidar_ray_weight" : 0.0, #missing
    "inital_velocity": 1.5,
    "normalize": False,
}


min_action_config = {
    "collision_penalty": 0.0,
    "progress_weight": 0.0,
    "raceline_delta_weight": 0.0,
    "velocity_weight": 0.0,
    "steering_change_weight": 0.0,
    "velocity_change_weight": 0.0,
    "pure_progress_weight": 0.0,
    "min_action_weight" : 1.0,
    "min_lidar_ray_weight" : 0.0, #missing
    "inital_velocity": 1.5,
    "normalize": False,
}


def dynamic_xavier_init(scale):
    def _initializer(m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight, gain=scale)
            if m.bias is not None:
                init.zeros_(m.bias)
    return _initializer




"""
The new dynamics model works like this:
0. the target action is computed from the previous action and the new action
1. delta x, delta y are predicted from previous theta, lin_vel, & target action
2. delta_theta_s, delta_theta_c are predicted with from the same states as above
(),out normalized to fullfill cos^2 + sin^2 = 1
3. Linear_vels are predicted from the same states as above (so no x,y, prev_action (rather target action))
4. Prev_action is computed outside the dynamics model
5. progress is computed by (x,y) Network
"""
class NewDynamicsModel(nn.Module):
    def __init__(self):
        # X,Y Prediction network
        super().__init__()
        pass
    def forward(self,x,u):
        #from x and u compute the target action
        target_action = x[:, 7:9] + u # this doesnt work we first need to unnormalize
        pass

class ModelBasedEnsemble(object):
    def __init__(self, state_dim, action_dim, hidden_size,  dt, min_state, max_state,
                 logger,
                 learning_rate=1e-3, weight_decay=1e-4, N=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim, self.action_dim, self.hidden_size, self.dt = state_dim, action_dim, hidden_size, dt
        self.N = N  # Number of models in the ensemble
        self.models = []  # This will store each of the ensemble members
        self.optimizers = []
        self.min_state = min_state
        self.max_state = max_state
        for _ in range(self.N):
            model = DynamicsModel(state_dim, action_dim, 
                                  hidden_size, dt, min_state,max_state).to(self.device)
            
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            # reduces performance a bit (n not large enough)
            #chosen_init = dynamic_xavier_init(scale=random.uniform(0.5, 1.5))
            #model.apply(chosen_init)
            self.models.append(model)
            self.optimizers.append(optimizer)

        self.writer = logger

    def forward(self, x, u):
        """
        Forward pass through all models of the ensemble.
        Returns predictions from all ensemble members.
        """
        predictions = []
        for model in self.models:
            prediction = model(x, u)
            predictions.append(prediction)
        
        return predictions
    
    def update(self, states, actions, next_states, step):
        # Convert input tensors
        #states = tf_to_torch(states).to(self.device)
        #actions = tf_to_torch(actions).to(self.device)
        #next_states = tf_to_torch(next_states).to(self.device)

        # Bootstrapping for each model in the ensemble
        batch_size = states.size(0)
        
        for model, optimizer in zip(self.models, self.optimizers):
            # Sample with replacement to create a bootstrapped batch
            indices = torch.randint(0, batch_size, (batch_size,)).to(self.device)
            bootstrapped_states = states[indices]
            bootstrapped_actions = actions[indices]
            bootstrapped_next_states = next_states[indices]
            
            # Reset gradients for the current model
            optimizer.zero_grad()
            
            # Predict using the current model and calculate loss
            pred_states = model(bootstrapped_states, bootstrapped_actions)
            dyn_loss = F.mse_loss(pred_states, bootstrapped_next_states, reduction='none')
            dyn_loss = (dyn_loss).mean()
            
            # Compute gradients and update model parameters
            dyn_loss.backward()
            optimizer.step()

        #self.dynamics_optimizer_iterations = 0
    def __call__(self,x,u):
        predictions = self.forward(x, u)
        avg_prediction = torch.mean(torch.stack(predictions), dim=0)
        return avg_prediction
    
    def save(self, path, filename):
        #print("saving model right now not implemented")
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), os.path.join(path, f"{filename}_{i}.pth"))

        #pass
        #for model in self.models:
    def load(self, path, filename):
        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(os.path.join(path, f"{filename}_{i}.pth")))

class ModelBased2(object):
    def __init__(self, state_dim, action_dim, hidden_size, dt,
                 logger, dataset, min_state, max_state, use_reward_model=False,
                 learning_rate=1e-3, weight_decay=1e-4,target_reward=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim, self.action_dim, self.hidden_size, self.dt = state_dim, action_dim, hidden_size, dt
        self.dynamics_model = ModelBasedEnsemble(state_dim, 
                                                   action_dim,
                                                    hidden_size,
                                                    dt,
                                                    tf_to_torch(min_state).to(self.device),
                                                    tf_to_torch(max_state).to(self.device),
                                                    logger,
                                                    learning_rate=learning_rate,
                                                    weight_decay=weight_decay,
                                                    N=3)
        #DynamicsModelPolicy(state_dim, action_dim, hidden_size, dt, logger,
                              #                    learning_rate=learning_rate, weight_decay=weight_decay)
        #self.min_state = 
        #self.max_state = 
        self.use_reward_model = use_reward_model
        if use_reward_model:
            self.reward_model = RewardModel(state_dim, action_dim)
            self.done_model = RewardModel(state_dim, action_dim)
            self.reward_model.to(self.device)
            self.done_model.to(self.device)
        
        else:
            # TODO! think about how to do better here
            if target_reward=="trajectories_td_prog.zarr":
                print("[mb] Using progress reward")
                self.reward_model = GroundTruthReward("Infsaal",dataset,20, **progress_config)
            elif target_reward=="trajectories_raceline.zarr":
                print("[mb] Using raceline reward")
                self.reward_model = GroundTruthReward("Infsaal",dataset,20, **raceline_config)
            elif target_reward=="trajectories_min_act.zarr":
                print("[mb]Using min action reward")
                self.reward_model = GroundTruthReward("Infsaal",dataset,20, **min_action_config)
            else:
                raise NotImplementedError

        self.writer=logger
        
        # self.dynamics_model.to(self.device)
        
        #self.optimizer_dynamics = optim.Adam(self.dynamics_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if use_reward_model:
            self.optimizer_reward = optim.Adam(self.reward_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            self.optimizer_done = optim.Adam(self.done_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.dynamics_optimizer_iterations = 0

    def update(self, states, actions, next_states, rewards, masks, weights):
        states = tf_to_torch(states)
        actions = tf_to_torch(actions)
        next_states = tf_to_torch(next_states)
        rewards = tf_to_torch(rewards)
        masks = tf_to_torch(masks)
        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        rewards = rewards.to(self.device)
        masks = masks.to(self.device)

        self.dynamics_model.update(states, actions, next_states, self.dynamics_optimizer_iterations)
        # Update model parameters
        
        if self.use_reward_model:
            # Update reward model
            self.optimizer_reward.zero_grad()  # Reset gradients
            pred_rewards = self.reward_model(states, actions)
            reward_loss = F.mse_loss(pred_rewards, rewards, reduction='none')
            reward_loss = (reward_loss).mean()
            reward_loss.backward()  # Compute gradients
            self.optimizer_reward.step()  # Update model parameters

            # Update done model
            self.optimizer_done.zero_grad()  # Reset gradients
            pred_dones = self.done_model(states, actions)
            done_loss = F.binary_cross_entropy_with_logits(pred_dones, masks, reduction='none')
            done_loss = (done_loss).mean()
            done_loss.backward()  # Compute gradients
            self.optimizer_done.step()
        
        if self.dynamics_optimizer_iterations % 1000 == 0:
            
            if self.use_reward_model:
                self.writer.add_scalar('cond/train/rew_loss', reward_loss.item(), global_step=self.dynamics_optimizer_iterations)
                self.writer.add_scalar('cond/train/done_loss', done_loss.item(), global_step=self.dynamics_optimizer_iterations)
        self.dynamics_optimizer_iterations += 1
    
    def plot_rollouts_fixed(self, states, actions, inital_mask, min_state, max_state, 
                            clip=True, horizon=1000, num_samples=20, 
                            path="logdir/plts/mb/rollouts_mb.png", get_target_action=None,scans=None):
        with torch.no_grad():
            states = tf_to_torch(states)
            actions = tf_to_torch(actions)
            states = states.to(self.device)
            actions = actions.to(self.device)
            min_state = tf_to_torch(min_state).to(self.device)
            max_state = tf_to_torch(max_state).to(self.device)

            
            #TODO! remove debug
            sampled_initial_indices = self.sample_initial_states(inital_mask, min_distance=horizon, num_samples=num_samples)
            #sampled_initial_indices = [1]#,1,2,3]
            #print(sampled_initial_indices)
            sampled_states = states[sampled_initial_indices]
            
            plt.figure(figsize=(12, 6))
            colors = plt.cm.jet(torch.linspace(0, 1, num_samples))

            # Plot all states as a grey background
            plt.scatter(states[:, 0].cpu().numpy(), states[:, 1].cpu().numpy(), color='grey', s=5, label='All states', alpha=0.5)
            
            for idx, start_idx in enumerate(sampled_initial_indices):
                state_trajectory = [states[start_idx][:2].cpu().numpy()]  # start with the sampled state

                # Plot "x" at the starting state
                plt.scatter(state_trajectory[0][0], state_trajectory[0][1], color=colors[idx], marker='x', s=60, label=f'Start {idx + 1}')
                
                # Plot the actual states using dashed lines
                actual_trajectory = states[start_idx:start_idx+horizon, :2].cpu().numpy()
                plt.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], '--', label=f"Ground Truth {idx + 1}", color=colors[idx])
                
                current_state = states[start_idx].unsqueeze(0).to(self.device)  # make it (1, state_dim)
                for i in range(horizon):
                    if get_target_action is None:
                        
                        action = actions[start_idx + i].unsqueeze(0).to(self.device)
                        #if i == 0:
                        #    print("State", current_state)
                        #    print("action", action)
                    else:
                        action = get_target_action(current_state.to('cpu').numpy() )
                        #if i == 0:
                        #    print("State", current_state)
                        #    print("action", action)
                        action = tf_to_torch(action).to(self.device)

                    current_state = self.dynamics_model(current_state, action)
                    if clip:
                        current_state = torch.clamp(current_state, min_state, max_state)

                    # Collect x, y for plotting
                    state_trajectory.append(current_state[0, :2].cpu().numpy())

                x_coords, y_coords = zip(*state_trajectory)
                plt.plot(x_coords, y_coords, label=f"Sample {idx + 1}", color=colors[idx])

            plt.title("Rollouts using Given Actions (torch)")
            plt.savefig(path)
            plt.show()
    
    def rollout(self, dataset,
                horizon=100, num_samples=20, discount=1.0, get_target_action=None):
        """
        Evaluate the rollouts using the learned dynamics model. And the fixed GT model.
        """
        with torch.no_grad():
            states, actions, rewards, inital_mask = dataset.states, dataset.actions, dataset.rewards, dataset.mask_inital
            raw_actions = dataset.raw_actions
            raw_actions = tf_to_torch(raw_actions).to(self.device)
            states = tf_to_torch(states).to(self.device)
            actions = tf_to_torch(actions).to(self.device)
            rewards = tf_to_torch(rewards).to(self.device)
            # inital_mask = tf_to_torch(inital_mask).to(self.device)
            sampled_initial_indices = self.sample_initial_states(inital_mask, min_distance=horizon, num_samples=num_samples)
            # first calculate the ground truth rewards
            sampled_initial_indices = torch.tensor(sampled_initial_indices)
            discount_factors = torch.tensor([discount**i for i in range(horizon)]).to(self.device)
            gt_rewards_segments = rewards[sampled_initial_indices[:, None] + torch.arange(horizon)].to(self.device)
            #print(gt_rewards_segments.cpu().numpy())
            #print(unnormalize_fn(gt_rewards_segments.cpu().numpy()))
            #print("aaaaa")
            # print(gt_rewards_segments.shape)
            gt_rewards = (gt_rewards_segments * discount_factors).sum(dim=1)
            mean_gt_rewards = unnormalize_fn(gt_rewards.mean().item())
            # print(f"Mean GT rewards: {mean_gt_rewards}")

            # next perform rollouts from the sampled_initial_indices with the dynamics model
            # use the GT reward model to label the rewards of the rollouts and compute the mean
            # print("Real first reward: ", rewards[sampled_initial_indices[0]])
            model_rewards = []
            for idx in sampled_initial_indices:
                state = states[idx].unsqueeze(0)
                rollout_rewards = []
                # TODO! read velocity from state, quo vadis velocity?, 
                # but should be unnecessary for used rewards aorn
                self.reward_model.reset(state[0, :2].cpu().numpy(), velocity=1.5)
                for i in range(horizon):
                    if get_target_action is None:
                        action = actions[idx + i].unsqueeze(0)
                    else:
                        action = get_target_action(state.to('cpu').numpy())
                        action = tf_to_torch(action).to(self.device)
                    next_state = self.dynamics_model(state, action)
                    # can calculate action_raw by taking current action adding to previous_action observation
                    pred_reward = self.reward_model(state.cpu().numpy(), action.cpu().numpy())
                    rollout_rewards.append(pred_reward * (discount**i))
                    state = next_state
                    #print(f"State {i} : {state}")
                    #print(f"Reward prediction: {pred_reward}")
                model_rewards.append(sum(rollout_rewards))

            mean_model_rewards = np.mean(np.asarray(model_rewards))
            # then compare the two means
            # write to tensorboard
            mean_gt_rewards = mean_gt_rewards.cpu().numpy()
            tag= ""
            if get_target_action is not None:
                tag = "_model_action"

            self.writer.add_scalar(f"test/mean_gt_rewards{tag}", mean_gt_rewards, global_step=self.dynamics_optimizer_iterations)
            self.writer.add_scalar(f"test/mean_model_rewards{tag}", mean_model_rewards, global_step=self.dynamics_optimizer_iterations)
            diff = mean_gt_rewards - mean_model_rewards
            self.writer.add_scalar(f"test/model_diff{tag}", diff, global_step=self.dynamics_optimizer_iterations)
            print(diff)
            print(mean_gt_rewards)
            print(mean_model_rewards)
        return mean_gt_rewards, mean_model_rewards

    def estimate_returns(self,inital_states, inital_weights, get_target_action, horizon=50, discount=0.99):
        with torch.no_grad():
            inital_states = tf_to_torch(inital_states)
            inital_states = inital_states.to('cpu')

            # loop over the inital states
            model_rewards = []
            # print(len(inital_states))
            j = 0
            for inital_state in inital_states: # loop over batch dimension
                j += 1
                #if j%10==0:
                #    print(j)
                state = inital_state.unsqueeze(0)
                rollout_rewards = []
                self.reward_model.reset(state[0, :2].cpu().numpy(), velocity=1.5)
                for i in range(horizon):
                    #print(state.shape)
                    action = get_target_action(state.to('cpu').numpy())
                
                    #print(action)
                    #print()
                    pred_reward = self.reward_model(state.cpu().numpy(), action.cpu().numpy())
                    action = tf_to_torch(action).to(self.device)
                    state = state.to(self.device)
                    next_state = self.dynamics_model(state, action)
                    rollout_rewards.append(pred_reward * (discount**i))
                    state = next_state
                model_rewards.append(sum(rollout_rewards))
                #break
                #if j ==100:
                #    break
            # print(model_rewards)
            mean_model_rewards = np.mean(np.asarray(model_rewards)) * (1 - discount)
            std_model_rewards = np.std(np.asarray(model_rewards)) * (1 - discount)
            return mean_model_rewards, std_model_rewards


    def sample_initial_states(self, initial_mask, min_distance=50, num_samples=20):
        # Find all initial state indices
        initial_indices = np.where(initial_mask == 1)[0]

        # Filter out the indices that don't have at least min_distance before the next initial state
        valid_indices = [idx for i, idx in enumerate(initial_indices[:-1]) if initial_indices[i+1] - idx >= min_distance]

        # If the last state is also valid (i.e., it has more than min_distance states until the end of the array)
        if len(initial_mask) - initial_indices[-1] >= min_distance:
            valid_indices.append(initial_indices[-1])

        # If there are fewer or equal valid indices than num_samples, return all valid indices
        if len(valid_indices) <= num_samples:
            return valid_indices
        
        # Determine the step size to take to get num_samples from the list of valid_indices
        step_size = len(valid_indices) // num_samples

        # Select the indices
        sampled_indices = valid_indices[::step_size][:num_samples]
        
        return sampled_indices
    
    def evaluate_rollouts(self, dataset,unnormalize_fn,
                           horizon=100, num_samples=20, discount=1.0, get_target_action=None ):
        """
        Evaluate the rollouts using the learned dynamics model. And the fixed GT model.
        """
        with torch.no_grad():
            states, actions, rewards, inital_mask = dataset.states, dataset.actions, dataset.rewards, dataset.mask_inital
            raw_actions = dataset.raw_actions
            raw_actions = tf_to_torch(raw_actions).to(self.device)
            states = tf_to_torch(states).to(self.device)
            actions = tf_to_torch(actions).to(self.device)
            rewards = tf_to_torch(rewards).to(self.device)
            # inital_mask = tf_to_torch(inital_mask).to(self.device)
            sampled_initial_indices = self.sample_initial_states(inital_mask, min_distance=horizon, num_samples=num_samples)
            # first calculate the ground truth rewards
            sampled_initial_indices = torch.tensor(sampled_initial_indices)
            discount_factors = torch.tensor([discount**i for i in range(horizon)]).to(self.device)
            gt_rewards_segments = rewards[sampled_initial_indices[:, None] + torch.arange(horizon)].to(self.device)
            #print(gt_rewards_segments.cpu().numpy())
            #print(unnormalize_fn(gt_rewards_segments.cpu().numpy()))
            #print("aaaaa")
            # print(gt_rewards_segments.shape)
            gt_rewards = (gt_rewards_segments * discount_factors).sum(dim=1)
            mean_gt_rewards = unnormalize_fn(gt_rewards.mean().item())
            # print(f"Mean GT rewards: {mean_gt_rewards}")

            # next perform rollouts from the sampled_initial_indices with the dynamics model
            # use the GT reward model to label the rewards of the rollouts and compute the mean
            # print("Real first reward: ", rewards[sampled_initial_indices[0]])
            
            
            """
            model_rewards = []
            for idx in sampled_initial_indices:
                print(idx)
                state = states[idx].unsqueeze(0)
                rollout_rewards = []
                # TODO! read velocity from state, quo vadis velocity?
                self.reward_model.reset(state[0, :2].cpu().numpy(), velocity=1.5)
                for i in range(horizon):
                    print("******")
                    action = actions[idx + i].unsqueeze(0)
                    # print("raw_action", raw_actions[idx + i].unsqueeze(0))
                    next_state = states[idx + i + 1].unsqueeze(0)
                    pred_reward = self.reward_model(state.cpu().numpy(), action.cpu().numpy())
                    rollout_rewards.append(pred_reward * (discount**i))
                    state = next_state
                    # print(f"State {i} : {state}")
                    print(f"Reward prediction: {pred_reward}")
                    print("+++++++")
                model_rewards.append(sum(rollout_rewards))
            print("----------")
            """
            
            model_rewards = []
            for idx in sampled_initial_indices:
                state = states[idx].unsqueeze(0)
                rollout_rewards = []
                # TODO! read velocity from state, quo vadis velocity?, 
                # but should be unnecessary for used rewards aorn
                self.reward_model.reset(state[0, :2].cpu().numpy(), velocity=1.5)
                for i in range(horizon):
                    if get_target_action is None:
                        action = actions[idx + i].unsqueeze(0)
                    else:
                        action = get_target_action(state.to('cpu').numpy())
                        action = tf_to_torch(action).to(self.device)

                    #print("action", action)
                    
                    #action_raw = raw_actions[idx + i].unsqueeze(0).cpu().numpy()
                    #print("raw action", action_raw)
                    next_state = self.dynamics_model(state, action)
                    # can calculate action_raw by taking current action adding to previous_action observation
                    pred_reward = self.reward_model(state.cpu().numpy(), action.cpu().numpy())
                    rollout_rewards.append(pred_reward * (discount**i))
                    state = next_state
                    #print(f"State {i} : {state}")
                    #print(f"Reward prediction: {pred_reward}")
                model_rewards.append(sum(rollout_rewards))

            mean_model_rewards = np.mean(np.asarray(model_rewards))
            # then compare the two means
            # write to tensorboard
            mean_gt_rewards = mean_gt_rewards.cpu().numpy()
            tag= ""
            if get_target_action is not None:
                tag = "_model_action"

            self.writer.add_scalar(f"test/mean_gt_rewards{tag}", mean_gt_rewards, global_step=self.dynamics_optimizer_iterations)
            self.writer.add_scalar(f"test/mean_model_rewards{tag}", mean_model_rewards, global_step=self.dynamics_optimizer_iterations)
            diff = mean_gt_rewards - mean_model_rewards
            self.writer.add_scalar(f"test/model_diff{tag}", diff, global_step=self.dynamics_optimizer_iterations)
            print(diff)
            print(mean_gt_rewards)
            print(mean_model_rewards)
        return mean_gt_rewards, mean_model_rewards

    def evaluate(self, states, actions, rewards, next_states, step, name, 
                clip=True, min_reward=-100, max_reward=100, min_state=-100, max_state=100):
        states = tf_to_torch(states)
        actions = tf_to_torch(actions)
        rewards = tf_to_torch(rewards)
        next_states = tf_to_torch(next_states)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)

        # only floats
        min_reward = float(min_reward.numpy())
        max_reward = float(max_reward.numpy())
        min_state = tf_to_torch(min_state).to(self.device)
        max_state = tf_to_torch(max_state).to(self.device)

        # Ensure no gradients are computed
        with torch.no_grad():
            pred_state, pred_rewards, logits = self.forward(states, actions, 
                                                            min_reward, 
                                                            max_reward, 
                                                            min_state, 
                                                            max_state, 
                                                            clip)

            # Compute the loss for dynamics
            loss = nn.MSELoss()(next_states, pred_state)
            loss_mean = loss.mean().item()  # Get scalar value from tensor

            # You can log using your preferred logging framework; Here's an example with tensorboardX
           
            self.writer.add_scalar(f'test/dyn_loss_{name}', loss_mean, global_step=self.dynamics_optimizer_iterations)
            
            if self.use_reward_model:
                # Compute the loss for rewards
                loss_reward = nn.MSELoss()(rewards, pred_rewards)
                loss_reward_mean = loss_reward.mean().item()

                self.writer.add_scalar(f'test/reward_loss_{name}', loss_reward_mean, global_step=self.dynamics_optimizer_iterations)
                # writer.close()

    def forward(self, states, actions, min_reward, max_reward, min_state, max_state, clip):
        """
        Forward pass through the model.
        """
        pred_rewards = None
        logits = None
        if self.use_reward_model:
            pred_rewards = self.reward_model(states, actions)  # Use the PyTorch call method

            if clip:
                pred_rewards = torch.clamp(pred_rewards, min_reward, max_reward)

            logits = self.done_model(states, actions)


        states = self.dynamics_model(states, actions)

        if clip:
            states = torch.clamp(states, min_state, max_state)

        return states, pred_rewards, logits
    
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
        if self.use_reward_model:
            # throw not impleemneted error
            raise NotImplementedError
        else:
            self.dynamics_model.save(save_path, filename)


    def load(self, checkpoint_path, filename):
        """
        Load the model's state dictionaries from a checkpoint.
        
        Args:
        - checkpoint_path (str): The path to the saved checkpoint file.
        """

        # Load the checkpoint
        # checkpoint = torch.load(checkpoint_path)

        # Restore the state dictionaries
        self.dynamics_model.load(checkpoint_path, filename) #checkpoint["dynamics_model_state_dict"])
        """
        if self.use_reward_model:
            self.reward_model.load_state_dict(checkpoint["reward_model_state_dict"])
            self.done_model.load_state_dict(checkpoint["done_model_state_dict"])
            
        # Move models to the appropriate device
        self.dynamics_model.to(self.device)
        if self.use_reward_model:
            self.reward_model.to(self.device)
            self.done_model.to(self.device)
        """