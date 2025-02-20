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

    def __init__(self, env, keys=None, info_keys=None, replay_buffer_keys=None, norm_obs=False, norm_limits=[-1.0, 1.0], imitate_cams=False):
        # Run super method
        super().__init__(env=env)

        self.imitate_cams = imitate_cams
        # self.cam_obs_names = ["left_depth", "right_depth"]
        self.cam_obs_names = ["left_depth"]

        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        self.norm_obs = norm_obs
        self.norm_limits = norm_limits

        self.observation_keys = keys
        self.info_observation_keys = info_keys

        if keys is not None:
            self.keys = self.create_key_list_from_mapping(self.observation_keys)
        
        if info_keys is not None:
            self.info_keys = self.create_key_list_from_mapping(self.info_observation_keys)

        if replay_buffer_keys is not None:
                if replay_buffer_keys["replay_buffer_type"] == "HerReplayBuffer":
                    her_obs = OrderedDict()

                    self.replay_buffer_keys = {}
                    self.replay_buffer_keys["replay_buffer_type"] = replay_buffer_keys["replay_buffer_type"]
                    self.replay_buffer_keys["observation"] = self.create_key_list_from_mapping(replay_buffer_keys["observation"])
                    self.replay_buffer_keys["achieved_goal"] = self.create_key_list_from_mapping(replay_buffer_keys["achieved_goal"])
                    self.replay_buffer_keys["desired_goal"] = self.create_key_list_from_mapping(replay_buffer_keys["desired_goal"])
        else:
            self.replay_buffer_keys = {}
            self.replay_buffer_keys["replay_buffer_type"] = ""
            self.replay_buffer_keys["observation"] = []
            self.replay_buffer_keys["achieved_goal"] = []
            self.replay_buffer_keys["desired_goal"] = []

        # Gym specific attributes
        self.env.spec = None

        # set up observation and action spaces
        obs = self.env.reset()
        # obs_by_modality = OrderedDict((key, value) for key, value in obs.items() if key.endswith("-state"))
        
        # self.modality_dims = {obs_key: obs[obs_key].shape for obs_key in self.keys}
        self.modality_dims = {obs_key: obs[obs_key].shape for obs_key in obs}

        observation_space = OrderedDict()
        if self.replay_buffer_keys["replay_buffer_type"] == "HerReplayBuffer":
            her_obs = self.map_her_obs(obs, self.replay_buffer_keys)

            if type(self.keys) == dict:
                for key, value in her_obs.items():
                    if self.norm_obs:
                        low, high = self.norm_limits[0], self.norm_limits[1]
                    else:
                        lows, highs = [], []
                        for limit_key, limit_value in self.replay_buffer_keys[key].items():
                            lows.append(limit_value["limits"][0])
                            highs.append(limit_value["limits"][1])
                        low = min(lows)
                        high = max(highs)
                    observation_space[key] = self.build_obs_space(shape=value.shape, low=low, high=high)
            elif type(self.keys) == list:
                for key, value in her_obs.items():
                    observation_space[key] = self.build_obs_space(shape=value.shape, low=-np.inf, high=np.inf)
        else:
            if type(self.keys) == dict:
                for key, value in self.keys.items():
                    shape = self.modality_dims[key]
                    if self.norm_obs:
                        low, high = self.norm_limits[0], self.norm_limits[1]
                    else:
                        low, high = value["limits"]
                    observation_space[key] = self.build_obs_space(shape=shape, low=low, high=high)
            elif type(self.keys) == list:
                for key in self.keys:
                    shape = self.modality_dims[key]
                    observation_space[key] = self.build_obs_space(shape=shape, low=-np.inf, high=np.inf)
        self.observation_space = gym.spaces.Dict(observation_space)

        low, high = self.env.action_spec
        self.action_space = gym.spaces.Box(low, high)

    def map_her_obs(self, obs, her_keys):
        her_obs = OrderedDict()
        her_obs["observation"] = self.concat_obs(self.filter_obs_dict_by_keys(obs, her_keys["observation"]))
        her_obs["achieved_goal"] = self.concat_obs(self.filter_obs_dict_by_keys(obs, her_keys["achieved_goal"]))
        her_obs["desired_goal"] = self.concat_obs(self.filter_obs_dict_by_keys(obs, her_keys["desired_goal"]))
        return her_obs

    # def create_key_list_from_mapping(self, keys):
    #     temp_list = []
    #     for key in keys:
    #         temp_list.extend(self.key_mapping(key))
    #     return temp_list

    def create_key_list_from_mapping(self, keys):
        if type(keys) == dict:
            parsed_keys = {}
            for key, value in keys.items():
                mapped_keys = self.key_mapping(key)
                for mapped_key in mapped_keys:
                    parsed_keys[mapped_key] = {}
                    if self.norm_obs:
                        parsed_keys[mapped_key]["limits"] = value["limits"]
                    else:
                        parsed_keys[mapped_key]["limits"] = [-np.inf, np.inf]
            return parsed_keys
        elif type(keys) == list:
            temp_list = []
            for key in keys:
                temp_list.extend(self.key_mapping(key))
            return temp_list
    
    def key_mapping(self, key):
        temp_list = []
        # Add object obs if requested
        if key == "object":
            temp_list.append("object-state")
        # Add image obs if requested
        elif key == "camera":
            for cam_name in self.env.camera_names:
                temp_list.append(f"{cam_name}_image")
        # Iterate over all robots to add to state
        elif key == "robot_proprio":
            for idx in range(len(self.env.robots)):
                temp_list.append("robot{}_proprio-state".format(idx))
        # Iterate over all robots to add to state
        elif key == "eef_pos":
            for idx in range(len(self.env.robots)):
                temp_list.append("robot{}_eef_pos".format(idx))
        elif key == "eef_quat":
            for idx in range(len(self.env.robots)):
                temp_list.append("robot{}_eef_quat".format(idx))
        elif key == "eef_vel_lin":
            for idx in range(len(self.env.robots)):
                temp_list.append("robot{}_eef_vel_lin".format(idx))
        elif key == "eef_vel_ang":
            for idx in range(len(self.env.robots)):
                temp_list.append("robot{}_eef_vel_ang".format(idx))
        else:
            temp_list.append(key)
        
        return temp_list

    def concat_obs(self, obs_dict, verbose=False):
        ob_lst = []
        for key in obs_dict.keys():
            ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)
    
    def filter_obs_dict_by_keys(self, obs_dict, keys):
        observations = OrderedDict()
        for key in keys:
            observations[key] = obs_dict[key]
        if self.norm_obs:
            observations = self.normalize_dict(observations, keys)
        return observations

    def check_dict_for_nan(self, observations, raise_error=True):       
        nan_detected = False
        for key, value in observations.items():
            if np.isnan(value).any():
                nan_detected = True
                print()
                print(f"NaN detected in observation '{key}'!")
                print(f"Observation '{key}': {value}")
        if nan_detected and raise_error:
            raise ValueError
    
    def build_obs_space(self, shape, low, high):
        """
        Helper function that builds individual observation spaces.

        :param shape: shape of the space
        :param low: lower bounds of the space
        :param high: higher bounds of the space
        """
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

    def normalize_value(self, value, c_min, c_max, normed_min, normed_max, key=""):
        if np.any((value < c_min) | (value > c_max)):
            print("Incorrect normalization input in:", key)
            if np.any((value < c_min)):
                print(f"value_min:{value.min()} < {c_min}")
            if np.any((value > c_max)):
                print(f"value_max:{value.max()} > {c_max}")
            
        v_normed = (value - c_min) / (c_max - c_min)
        v_normed = v_normed * (normed_max - normed_min) + normed_min

        if np.any((v_normed < normed_min) | (v_normed > normed_max)):
            print("Incorrect normalization output in:", key)
            if np.any((v_normed < normed_min)):
                print(f"normed_min:{v_normed.min()} < {normed_min}")
            if np.any((v_normed > normed_max)):
                print(f"normed_max:{v_normed.max()} > {normed_max}")
        
        return v_normed

    def denormalize_value(self, v_normed, c_min, c_max, normed_min, normed_max, key=""):
        value = (v_normed - normed_min)/(normed_max - normed_min)
        value = value * (c_max - c_min) + c_min
        return value
    
    def normalize_dict(self, input_dict, keys):
        output_dict = OrderedDict()
        for key, value in input_dict.items():
            # print(f"Key={key}\nValue={value}")
            low, high = keys[key]["limits"]
            normed_value = self.normalize_value(value, low, high, self.norm_limits[0], self.norm_limits[1], key)
            output_dict[key] = normed_value
        return output_dict
    
    def denormalize_dict(self, input_dict, keys):
        output_dict = OrderedDict()
        for key, value in input_dict.items():
            # print(f"Key={key}\nValue={value}")
            low, high = keys[key]["limits"]
            output_dict[key] = self.denormalize_value(value, low, high, self.norm_limits[0], self.norm_limits[1], key)
        return output_dict

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
        self.check_dict_for_nan(ob_dict)
        if self.replay_buffer_keys["replay_buffer_type"] == "HerReplayBuffer":
            observations = self.map_her_obs(ob_dict, self.replay_buffer_keys)
        else:
            observations = self.filter_obs_dict_by_keys(ob_dict, self.keys)

        if self.imitate_cams:
            info = dict(
                (key, cam_ob) 
                for key, cam_ob in ob_dict.items() 
                if any(sub in key for sub in ("depth",))
            )
            # info = dict(
            #     (key, cam_ob) 
            #     for key, cam_ob in ob_dict.items() 
            #     if any(sub in key for sub in ("image", "depth", "segmentation"))
            # )
            # info = dict((key, cam_ob.flatten().tolist()) for key, cam_ob in ob_dict.items() if any(sub in key for sub in ("image", "depth", "segmentation")))
            # info = {}
        else:
            info = dict()
        return observations, info # observation, reset_info

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
        self.check_dict_for_nan(ob_dict)
        if self.replay_buffer_keys["replay_buffer_type"] == "HerReplayBuffer":
            observations = self.map_her_obs(ob_dict, self.replay_buffer_keys)
        else:
            observations = self.filter_obs_dict_by_keys(ob_dict, self.keys)

        if self.imitate_cams:
            info.update({
                key: cam_ob
                for key, cam_ob in ob_dict.items()
                if any(sub in key for sub in ("depth",))
            })
            # info.update({
            #     key: cam_ob
            #     for key, cam_ob in ob_dict.items()
            #     if any(sub in key for sub in ("image", "depth", "segmentation"))
            # })
            # info.update({key: cam_ob.flatten().tolist() for key, cam_ob in ob_dict.items() if any(sub in key for sub in ("image", "depth", "segmentation"))})
            if terminated:
                # print("here")
                observations.update({
                    key: cam_ob
                    for key, cam_ob in ob_dict.items()
                    if any(sub in key for sub in self.cam_obs_names)
                })
        
        return observations, reward, terminated, False, info

    def map_her_goal(self, goal, goal_name):
        goal_dict = OrderedDict()
        shapes = {}
        index = 0
        for key in self.replay_buffer_keys[goal_name]:
            shape = self.modality_dims[key][0]
            start = index
            end = index+shape
            shapes[key] = [start, end]
            index += shape
        for key in self.replay_buffer_keys[goal_name]:
            start = shapes[key][0]
            end = shapes[key][1]
            goal_dict[key] = goal[:, start:end]
        return goal_dict
    
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
        if self.replay_buffer_keys["replay_buffer_type"] == "HerReplayBuffer":           
            # Convert goals to dicts           
            achieved_goal_dict = self.map_her_goal(achieved_goal, "achieved_goal")
            desired_goal_dict = self.map_her_goal(desired_goal, "desired_goal")

            # Check for NaN values
            self.check_dict_for_nan(achieved_goal_dict)
            self.check_dict_for_nan(desired_goal_dict)

            # Denormalize values if they were normalized
            if self.norm_obs:
                achieved_goal_dict = self.denormalize_dict(achieved_goal_dict, self.replay_buffer_keys["achieved_goal"])
                desired_goal_dict = self.denormalize_dict(desired_goal_dict, self.replay_buffer_keys["desired_goal"])
            
            reward = self.env.reward(achieved_goal=achieved_goal_dict, desired_goal=desired_goal_dict)
        else:
            reward = self.env.reward()
        
        if np.isnan(reward).any():
            print(f"NaN detected in reward")
            print(f"reward: {reward}")
            raise ValueError
        return reward


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