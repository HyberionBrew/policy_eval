from f110_sim_env.base_env import make_base_env
from f110_orl_dataset import fast_reward

class Sim:
    def __init__(self, dataset, agent_config, target_reward_config, enviroment_config, normalizer=None, map= "Infsaal"):
        
        self.env = make_base_env(random_start =False, 
                                 pose_start=True,
                                 map=map,
                                 min_vel = enviroment_config.min_vel,
                                 max_vel = enviroment_config.max_vel,
                                 max_acceleration=enviroment_config.max_acceleration,
                                 max_delta_steering=enviroment_config.max_delta_steering,
                                 )
        
        self.reward_model = fast_reward
