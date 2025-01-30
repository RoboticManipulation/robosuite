import numpy as np
import robosuite as suite

# create environment instance
env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="UR5e",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    hard_reset=False,
    mujoco_passive_viewer=True
)

# reset the environment
env.reset()

while True:
    action = np.random.randn(*env.action_spec[0].shape) * 0.1
    obs, reward, done, info = env.step(action)  # take action in the environment
    if env.has_renderer or env.has_offscreen_renderer:
        env.render()  # render on display