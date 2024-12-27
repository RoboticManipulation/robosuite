"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like
interface.
"""

import numpy as np
import gymnasium as gym

from robosuite.wrappers import Wrapper

from collections import OrderedDict

class IndexEnvWrapper(Wrapper, gym.Env):

    def __init__(self, env, index, keys=None):
        # Run super method
        # super().__init__(env=env)
        super().__init__(env=env)
        # super(IndexEnvWrapper, self).__init__(env=env)
        self.index = index

    # def __init__(self, env, index):
    #     super(IndexEnvWrapper, self).__init__(env)
    #     self.index = index

    def reset(self, **kwargs):
        # Pass all arguments to the underlying environment's reset method
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["env_index"] = self.index  # Include environment index in info
        return obs, reward, terminated, truncated, info
    

"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like
interface.
"""
class GymWrapperDictObs(Wrapper, gym.Env):
    metadata = None
    render_mode = None
    """
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, keys=None):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        self.observation_keys = keys

        self.keys = []
        for key in self.observation_keys:
            # Add object obs if requested
            if key == "object":
                self.keys.append("object-state")
            # Add image obs if requested
            elif key == "camera":
                for cam_name in self.env.camera_names:
                    self.keys.append(f"{cam_name}_image")
            # Iterate over all robots to add to state
            elif key == "robot_proprio":
                for idx in range(len(self.env.robots)):
                    self.keys.append("robot{}_proprio-state".format(idx))
            # Iterate over all robots to add to state
            elif key == "eef_pos":
                for idx in range(len(self.env.robots)):
                    self.keys.append("robot{}_eef_pos".format(idx))
            elif key == "eef_quat":
                for idx in range(len(self.env.robots)):
                    self.keys.append("robot{}_eef_quat".format(idx))
            elif key == "eef_vel_lin":
                for idx in range(len(self.env.robots)):
                    self.keys.append("robot{}_eef_vel_lin".format(idx))
            elif key == "eef_vel_ang":
                for idx in range(len(self.env.robots)):
                    self.keys.append("robot{}_eef_vel_ang".format(idx))
            else:
                self.keys.append(key)

        # Gym specific attributes
        self.env.spec = None

        # set up observation and action spaces
        obs = self.env.reset()
        # obs_by_modality = OrderedDict((key, value) for key, value in obs.items() if key.endswith("-state"))
        
        self.modality_dims = {key: obs[key].shape for key in self.keys}

        observation_space = OrderedDict()
        for key in self.keys:
            shape = self.modality_dims[key]
            observation_space[key] = self.build_obs_space(shape=shape, low=-np.inf, high=np.inf)
        self.observation_space = gym.spaces.Dict(observation_space)

        low, high = self.env.action_spec
        self.action_space = gym.spaces.Box(low, high)

    def filter_obs_dict_by_keys(self, obs_dict, keys):
        observations = OrderedDict()
        for key in keys:
            observations[key] = obs_dict[key]
        return observations

    def build_obs_space(self, shape, low, high):
        """
        Helper function that builds individual observation spaces.

        :param shape: shape of the space
        :param low: lower bounds of the space
        :param high: higher bounds of the space
        """
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Returns observations and optionally resets seed

        Returns:
            gym.spaces.Dict: observation space
        """
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        ob_dict = self.env.reset()
        observations = self.filter_obs_dict_by_keys(ob_dict, self.keys)
        return observations, {} # observation, reset_info

    def step(self, action):
        """
        Converts 4-tuple to 5-tuple

        Args:
            action (np.array): Action to take in environment

        Returns:
            5-tuple:

                - (gym.spaces.Dict) observations from the environment
                - (float) reward from the environment
                - (bool) episode ending after reaching an env terminal state
                - (bool) episode ending after an externally defined condition
                - (dict) misc information
        """
        ob_dict, reward, terminated, info = self.env.step(action)
        observations = self.filter_obs_dict_by_keys(ob_dict, self.keys)
        return observations, reward, terminated, False, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()


# Old GymWrapperDictObs
# class GymWrapperDictObs(Wrapper, gym.Env):
#     metadata = None
#     render_mode = None
#     """
#     Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
#     found in the gym.core module

#     Args:
#         env (MujocoEnv): The environment to wrap.
#         keys (None or list of str): If provided, each observation will
#             consist of concatenated keys from the wrapped environment's
#             observation dictionary. Defaults to proprio-state and object-state.

#     Raises:
#         AssertionError: [Object observations must be enabled if no keys]
#     """

#     def __init__(self, env, keys=None):
#         # Run super method
#         super().__init__(env=env)
#         # Create name for gym
#         robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
#         self.name = robots + "_" + type(self.env).__name__

#         if keys is None:
#             keys = []
#             # Add object obs if requested
#             if self.env.use_object_obs:
#                 keys += ["object-state"]
#             # Add image obs if requested
#             if self.env.use_camera_obs:
#                 keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
#             # Iterate over all robots to add to state
#             for idx in range(len(self.env.robots)):
#                 keys += ["robot{}_proprio-state".format(idx)]
#         self.keys = keys

#         # Gym specific attributes
#         self.env.spec = None

#         # set up observation and action spaces
#         # TODO: Change to preserve dictionary shapes when e.g. passing images
#         obs = self.env.reset()
#         self.modality_dims = {key: obs[key].shape for key in self.keys}

#         observation_space = OrderedDict()
#         for key in self.keys:
#             shape = self.modality_dims[key]
#             observation_space[key] = self.build_obs_space(shape=shape, low=-np.inf, high=np.inf)
#         self.observation_space = gym.spaces.Dict(observation_space)

#         low, high = self.env.action_spec
#         self.action_space = gym.spaces.Box(low, high)

#     def build_obs_space(self, shape, low, high):
#         """
#         Helper function that builds individual observation spaces.

#         :param shape: shape of the space
#         :param low: lower bounds of the space
#         :param high: higher bounds of the space
#         """
#         return gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

#     def reset(self, seed=None, options=None):
#         """
#         Returns observations and optionally resets seed

#         Returns:
#             gym.spaces.Dict: observation space
#         """
#         if seed is not None:
#             if isinstance(seed, int):
#                 np.random.seed(seed)
#             else:
#                 raise TypeError("Seed must be an integer type!")
#         ob_dict = self.env.reset()
#         return ob_dict, {}

#     def step(self, action):
#         """
#         Converts 4-tuple to 5-tuple

#         Args:
#             action (np.array): Action to take in environment

#         Returns:
#             5-tuple:

#                 - (gym.spaces.Dict) observations from the environment
#                 - (float) reward from the environment
#                 - (bool) episode ending after reaching an env terminal state
#                 - (bool) episode ending after an externally defined condition
#                 - (dict) misc information
#         """
#         ob_dict, reward, terminated, info = self.env.step(action)
#         return ob_dict, reward, terminated, False, info

#     def compute_reward(self, achieved_goal, desired_goal, info):
#         """
#         Dummy function to be compatible with gym interface that simply returns environment reward

#         Args:
#             achieved_goal: [NOT USED]
#             desired_goal: [NOT USED]
#             info: [NOT USED]

#         Returns:
#             float: environment reward
#         """
#         # Dummy args used to mimic Wrapper interface
#         return self.env.reward()
    
"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like
interface.
"""
class GymWrapper(Wrapper, gym.Env):
    metadata = None
    render_mode = None
    """
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, keys=None):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        if keys is None:
            keys = []
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # Add image obs if requested
            if self.env.use_camera_obs:
                keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys

        # Gym specific attributes
        self.env.spec = None

        # set up observation and action spaces
        obs = self.env.reset()
        self.modality_dims = {key: obs[key].shape for key in self.keys}
        flat_ob = self._flatten_obs(obs)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = gym.spaces.Box(low, high)
        low, high = self.env.action_spec
        self.action_space = gym.spaces.Box(low, high)

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)

    def reset(self, seed=None, options=None):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict and optionally resets seed

        Returns:
            2-tuple:
                - (np.array) flattened observations from the environment
                - (dict) an empty dictionary, as part of the standard return format
        """
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        ob_dict = self.env.reset()
        return self._flatten_obs(ob_dict), {}

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) episode ending after reaching an env terminal state
                - (bool) episode ending after an externally defined condition
                - (dict) misc information
        """
        ob_dict, reward, terminated, info = self.env.step(action)
        return self._flatten_obs(ob_dict), reward, terminated, False, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()