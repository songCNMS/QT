import gym
import d4rl
from envs_info.meta_info import env_state_contexts, env_obs_names, env_task_descriptions, offline_dataset_names
# from gym.wrappers.normalize import NormalizeReward



def get_env(env_name="Hopper-v2"):
    env = gym.make(offline_dataset_names[env_name])
    return env


def get_state_description(env, obss, env_name):
    context = env_state_contexts[env_name]
    obs_names = env_obs_names[env_name]
    for i, o_name in enumerate(obs_names):
        context += f"{o_name}: {obss[i]}\n"
    return context


def get_task_desc(env, env_name):
    return env_task_descriptions[env_name]

