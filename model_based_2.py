import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import os

def tf_to_torch(tf_tensor):
    """
    Convert a TensorFlow tensor to a PyTorch tensor.
    """
    # Convert TensorFlow tensor to NumPy
    numpy_array = tf_tensor.numpy()
    
    # Convert NumPy array to PyTorch tensor
    torch_tensor = torch.from_numpy(numpy_array)
    
    return torch_tensor

class DynamicsModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, dt):
        
        super().__init__()
        self.state_size, self.action_size, self.dt = state_size, action_size, dt
        A_size, B_size = state_size * state_size, state_size * action_size
        self.A1 = nn.Linear(state_size + action_size, hidden_size)
        self.A2 = nn.Linear(hidden_size, A_size)
        self.B1 = nn.Linear(state_size + action_size, hidden_size)
        self.B2 = nn.Linear(hidden_size, B_size)
        self.STATE_X, self.STATE_Y = 0, 1
        self.STATE_PROGRESS_SIN = 9
        self.STATE_PROGRESS_COS = 10
        # maybe also remove this
        self.STATE_PREVIOUS_ACTION_STEERING = 7
        self.STATE_PREVIOUS_ACTION_VELOCITY = 8

    def forward(self, x, u):
        """
            Predict x_{t+1} = f(x_t, u_t)
        :param x: a batch of states
        :param u: a batch of actions
        """
        #print(x.shape)
        #print(u.shape)

        xu = torch.cat((x, u), -1)
        xu[:, self.STATE_X:self.STATE_Y+1] = 0  # Remove dependency in (x,y)
        xu[:, self.STATE_PROGRESS_SIN:self.STATE_PROGRESS_COS+1] = 0  # Remove dependency in progress
        A = self.A2(F.relu(self.A1(xu)))
        A = torch.reshape(A, (x.shape[0], self.state_size, self.state_size))
        B = self.B2(F.relu(self.B1(xu)))
        B = torch.reshape(B, (x.shape[0], self.state_size, self.action_size))
        dx = A @ x.unsqueeze(-1) + B @ u.unsqueeze(-1)
        return x + dx.squeeze()*self.dt


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
        observation_dict = self.model_input_normalizer.unflatten_batch(states)
        laser_scan = self.env.get_laser_scan(states, self.subsample_laser) # TODO! rename f110env to dataset_env
        laser_scan = self.model_input_normalizer.normalize_laser_scan(laser_scan)
        observation_dict['lidar_occupancy'] = laser_scan
        # have to unsqueeze the batch dimension
        observation_dict = {key: value.squeeze(0) if value.ndim > 1 and value.shape[0] == 1 else value for key, value in observation_dict.items()}
        #print(observation_dict)
        #print(np.array([observation_dict['poses_x'][0], observation_dict['poses_y'][0]]))
        reward, _ = self.reward(observation_dict, action, 
                                      collision, done)
        return reward
    
    def reset(self, pose , velocity=1.5):
        # add 9 empty states such that we can use the unnormalize function
        states = np.zeros((1, 11), dtype=np.float32)
        states[0, 0] = pose[0]
        states[0, 1] = pose[1]
        pose = self.dataset.unnormalize_states(states)[0][:2]
        self.reward.reset(pose, velocity=velocity)

standard_config = {
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

class ModelBased2(object):
    def __init__(self, state_dim, action_dim, hidden_size, dt,
                 logger, dataset,use_reward_model=False,
                 learning_rate=1e-3, weight_decay=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim, self.action_dim, self.hidden_size, self.dt = state_dim, action_dim, hidden_size, dt
        self.dynamics_model = DynamicsModel(state_dim, action_dim, hidden_size, dt)
        self.use_reward_model = use_reward_model
        if use_reward_model:
            self.reward_model = RewardModel(state_dim, action_dim)
            self.done_model = RewardModel(state_dim, action_dim)
            self.reward_model.to(self.device)
            self.done_model.to(self.device)
        
        else:
            self.reward_model = GroundTruthReward("Infsaal",dataset,20, **standard_config)

        self.writer=logger
        
        self.dynamics_model.to(self.device)
        
        self.optimizer_dynamics = optim.Adam(self.dynamics_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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

        
        self.optimizer_dynamics.zero_grad()  # Reset gradients
        pred_states = self.dynamics_model(states, actions)
        dyn_loss = F.mse_loss(pred_states, next_states, reduction='none')
        dyn_loss = (dyn_loss).mean()
        dyn_loss.backward()  # Compute gradients
        self.optimizer_dynamics.step()  # Update model parameters
        
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
            self.writer.add_scalar('train/dyn_loss', dyn_loss.item(), global_step=self.dynamics_optimizer_iterations)
            if self.use_reward_model:
                self.writer.add_scalar('train/rew_loss', reward_loss.item(), global_step=self.dynamics_optimizer_iterations)
                self.writer.add_scalar('train/done_loss', done_loss.item(), global_step=self.dynamics_optimizer_iterations)
        self.dynamics_optimizer_iterations += 1
    
    def plot_rollouts_fixed(self, states, actions, inital_mask, min_state, max_state, 
                            clip=True, horizon=1000, num_samples=20, 
                            path="logdir/plts/mb/rollouts_mb.png"):
        with torch.no_grad():
            states = tf_to_torch(states)
            actions = tf_to_torch(actions)
            states = states.to(self.device)
            actions = actions.to(self.device)
            min_state = tf_to_torch(min_state).to(self.device)
            max_state = tf_to_torch(max_state).to(self.device)

            
            
            sampled_initial_indices = self.sample_initial_states(inital_mask, min_distance=horizon, num_samples=num_samples)
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
                    action = actions[start_idx + i].unsqueeze(0).to(self.device)

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
                           horizon=100, num_samples=20, discount=1.0, ):
        """
        Evaluate the rollouts using the learned dynamics model. And the fixed GT model.
        """
        with torch.no_grad():
            states, actions, rewards, inital_mask = dataset.states, dataset.actions, dataset.rewards, dataset.mask_inital
            states = tf_to_torch(states).to(self.device)
            actions = tf_to_torch(actions).to(self.device)
            rewards = tf_to_torch(rewards).to(self.device)
            # inital_mask = tf_to_torch(inital_mask).to(self.device)
            sampled_initial_indices = self.sample_initial_states(inital_mask, min_distance=horizon, num_samples=num_samples)
            # first calculate the ground truth rewards
            sampled_initial_indices = torch.tensor(sampled_initial_indices)
            discount_factors = torch.tensor([discount**i for i in range(horizon)]).to(self.device)
            gt_rewards_segments = rewards[sampled_initial_indices[:, None] + torch.arange(horizon)].to(self.device)
            print(gt_rewards_segments.cpu().numpy())
            print(unnormalize_fn(gt_rewards_segments.cpu().numpy()))
            print("aaaaa")
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
                # TODO! read velocity from state, quo vadis velocity?
                self.reward_model.reset(state[0, :2].cpu().numpy(), velocity=1.5)
                for i in range(horizon):
                    action = actions[idx + i].unsqueeze(0)
                    next_state = states[idx + i + 1].unsqueeze(0)
                    pred_reward = self.reward_model(state.cpu().numpy(), action.cpu().numpy())
                    rollout_rewards.append(pred_reward * (discount**i))
                    state = next_state
                    print(f"State {i} : {state}")
                    print(f"Reward prediction: {pred_reward}")
                model_rewards.append(sum(rollout_rewards))
            print("----------")
            
            model_rewards = []
            for idx in sampled_initial_indices:
                state = states[idx].unsqueeze(0)
                rollout_rewards = []
                # TODO! read velocity from state, quo vadis velocity?
                self.reward_model.reset(state[0, :2].cpu().numpy(), velocity=1.5)
                for i in range(horizon):
                    action = actions[idx + i].unsqueeze(0)
                    next_state = self.dynamics_model(state, action)
                    pred_reward = self.reward_model(state.cpu().numpy(), action.cpu().numpy())
                    rollout_rewards.append(pred_reward * (discount**i))
                    state = next_state
                    print(f"State {i} : {state}")
                    print(f"Reward prediction: {pred_reward}")
                model_rewards.append(sum(rollout_rewards))
            mean_model_rewards = np.mean(np.asarray(model_rewards))
            # then compare the two means
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
            checkpoint = {
                "dynamics_model_state_dict": self.dynamics_model.state_dict(),
                "reward_model_state_dict": self.reward_model.state_dict(),
                "done_model_state_dict": self.done_model.state_dict(),
                # Add other things to save if necessary
            }
        else:
            checkpoint = {
                "dynamics_model_state_dict": self.dynamics_model.state_dict(),
                # Add other things to save if necessary
            }
        # Save the checkpoint
        torch.save(checkpoint, os.path.join(save_path, filename))
    
    def load(self, checkpoint_path):
        """
        Load the model's state dictionaries from a checkpoint.
        
        Args:
        - checkpoint_path (str): The path to the saved checkpoint file.
        """

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)

        # Restore the state dictionaries
        self.dynamics_model.load_state_dict(checkpoint["dynamics_model_state_dict"])
        if self.use_reward_model:
            self.reward_model.load_state_dict(checkpoint["reward_model_state_dict"])
            self.done_model.load_state_dict(checkpoint["done_model_state_dict"])
            
        # Move models to the appropriate device
        self.dynamics_model.to(self.device)
        if self.use_reward_model:
            self.reward_model.to(self.device)
            self.done_model.to(self.device)