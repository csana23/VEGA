import cv2
import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import vizdoom.gymnasium_wrapper  # noqa
from sb_ppo_basic_train import ObservationWrapper

DEFAULT_ENV = "VizdoomBasic-v0"
AVAILABLE_ENVS = [env for env in gymnasium.envs.registry.keys() if "Vizdoom" in env]
ENV = "VizdoomBasic-v0"
# Height and width of the resized image
IMAGE_SHAPE = (60, 80)
TB_LOG="../runs/sb_ppo_tensorboard/"

def main():
    def wrap_env(env):
        env = ObservationWrapper(env)
        env = gymnasium.wrappers.TransformReward(env, lambda r: r * 0.01)
        return env
    
    envs = make_vec_env(ENV, n_envs=1, wrapper_class=wrap_env)

    model = PPO.load("sb_ppo")

    obs = envs.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = envs.step(action)
        envs.render("human")

if __name__ == "__main__":
    main()