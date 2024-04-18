import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class UniformContextDistribution():
    def __init__(self, low, high, num_dimension):
        self.low = low
        self.high = high
        self.num_dimension = num_dimension

    def get_random_state(self):
        return np.random.uniform(
            low=self.low, 
            high=self.high, 
            size=self.num_dimension,
        )


class ContextualBanditsEnv(gym.Env):
    """ A Context Bandit environment
    """
    def __init__(
        self,
        action_config,
        observation_config,
        reward_function,
    ):
        self.setup_action_space(action_config=action_config)
        self.setup_observation_space(observation_config=observation_config)
        self.reward_function = reward_function
        self._seed()
        self.reset()

    def setup_action_space(self, action_config):
        self.action_config = action_config
        if self.action_config["action_space_type"] == "discrete":
            self.action_space = spaces.Discrete(
                self.action_config["num_actions"]
            )
        elif self.action_config["action_space_type"] == "continuous":
            low_value = self.action_config["low"]
            high_value = self.action_config["high"]
            num_dimension = self.action_config.get("num_dimension", 1)

            self.action_low = np.array([low_value for i in range(num_dimension)], dtype=np.float64)
            self.action_high = np.array([high_value for i in range(num_dimension)], dtype=np.float64)
            self.action_space = spaces.Box(
                low=self.action_low,
                high=self.action_high,
            )
        else:
            raise ValueError("Given action config is not supported.")
        
    def setup_observation_space(self, observation_config):
        self.observation_config = observation_config
        if self.observation_config["observation_space_type"] == "discrete":
            self.observation_space = spaces.Discrete(
                self.observation_config["num_states"]
            )
        if self.observation_config["observation_space_type"] == "continuous":
            low_value = self.observation_config["low"]
            high_value = self.observation_config["high"]
            num_dimension = self.observation_config.get("num_dimension", 1)

            low = np.array([low_value for i in range(num_dimension)], dtype=np.float64)
            high = np.array([high_value for i in range(num_dimension)], dtype=np.float64)
            self.observation_space = spaces.Box(
                low=low,
                high=high,
            )
        else:
            raise ValueError("Given observation config is not supported.")
        
        self.context_distribution = self.observation_config["context_distribution"]

    def reset(self):
        self.state = self.context_distribution.get_random_state()
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human', close=False):
        pass
    
    def step(self, action):
        """ steps forward one trial
        
        Args:
            action - action comes from the algorithm file
        """
        assert self.action_space.contains(action)
        
        done = True
        reward = self.reward_function(context=self.state, action=action)
        self.reset()
        
        return self.state, reward, done, {}
    
    def reward(self, state, action):
        return self.reward_function(context=state, action=action)
    

class TransformerEnv(gym.Env):
    def __init__(
        self,
        max_new_tokens,
        vocab_size,
        reward_function,
    ):
        self.action_space = spaces.Discrete(
            vocab_size,
        )
        self.observation_space = spaces.Discrete(
            vocab_size,
        )
        self.reward_function = reward_function
        self.vocab_size = vocab_size
        self.max_new_tokens = max_new_tokens
        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        self.state = np.random.randint(0, self.vocab_size)
        return self.state
    
    def render(self, mode='human', close=False):
        pass

    def step(self, action):
        """ steps forward one trial
        
        Args:
            action - action comes from the algorithm file
        """
        
        done = True
        reward = self.reward_function(context=self.state, action=action)
        self.reset()
        
        return self.state, reward, done, {}
    
    def reward(self, state, action):
        return self.reward_function(context=state, action=action)