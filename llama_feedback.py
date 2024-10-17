import pickle
from envs_info.m_envs import get_state_description
from envs_info.meta_info import *
import urllib.request
import json
import os
import ssl
import seaborn as sns
import torch
from datasets import load_dataset, Dataset
from transformers import LlamaConfig, AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from string import Template
import time
from openai import OpenAI
from omegaconf import OmegaConf
import random
from transformers import AutoModelForCausalLM, AutoTokenizer




class LLMEmbModel(nn.Module):
    def __init__(self, llm_model, llm_dim, llm_layers):
        super(LLMEmbModel, self).__init__()
        self.d_llm = llm_dim
        self.llm_model = llm_model
        self.llm_layers = llm_layers

        if self.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('meta-llama/Meta-Llama-3-8B')
            self.llama_config.num_hidden_layers = self.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    # 'huggyllama/llama-7b',
                    "meta-llama/Meta-Llama-3-8B",
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except Exception:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    # 'huggyllama/llama-7b',
                    "meta-llama/Meta-Llama-3-8B",
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    ignore_mismatched_sizes=True
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    # 'huggyllama/llama-7b',
                    "meta-llama/Meta-Llama-3-8B",
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    # 'huggyllama/llama-7b',
                    "meta-llama/Meta-Llama-3-8B",
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif self.llm_model == "GPT2":
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
            self.gpt2_config.num_hidden_layers = self.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                'openai-community/gpt2',
                trust_remote_code=True,
                local_files_only=False,
                config=self.gpt2_config,
            )
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                'openai-community/gpt2',
                trust_remote_code=True,
                local_files_only=False
            )
 
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        
    def get_emb(self, prompt):
        prompt = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = torch.mean(self.llm_model.get_input_embeddings()(prompt.to(self.llm_model.device)), axis=0)
        return prompt_embeddings[0].detach().cpu().numpy()
    

    
def get_local_model_resp(model, tokenizer, input):
    try:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": input}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except Exception as ex:
        print(ex)
        response = "ERROR"
    return response

if __name__ == "__main__":
    cfg = OmegaConf.from_cli()
    idx = cfg.idx
    total_split = cfg.split
    verbose = cfg.get("verbose", False)
    model_name = cfg.get("llm", "Qwen/Qwen2.5-32B-Instruct")
    device = f"cuda"
    model_emb_size = {
        "LLAMA": {"emb_size": 4096, "num_hidden_layer": 32},
        "GPT2":  {"emb_size": 768, "num_hidden_layer": 12},
    }
    llm_model_name = "GPT2"
    llmemb_model = LLMEmbModel(llm_model_name, model_emb_size[llm_model_name]['emb_size'], model_emb_size[llm_model_name]["num_hidden_layer"]).to(device)    
        
    question_template = Template("""
    Based on the following environment description and a given observation, please give a summary of the observation and instructions for improving. In your summary, please include your comments on whether it is a good or bad observation and why. ###Environment Description### "    
    The environment aims to increase the number of independent state and control variables as compared to the classic control environments. The hopper is a two-dimensional one-legged figure that consist of four main body parts - the torso at the top, the thigh in the middle, the leg in the bottom, and a single foot on which the entire body rests. The goal is to make hops that move in the forward (right) direction by applying torques on the three hinges connecting the four body parts.    
    Action Space    
    The action space is a Box(-1, 1, (3,), float32). An action represents the torques applied at the hinge joints.    
    Num	  Action	Control Min	Control Max	Name (in corresponding XML file)	Joint	Unit    
    0	Torque applied on the thigh rotor	-1	1	thigh_joint	hinge	torque (N m)    
    1	Torque applied on the leg rotor	-1	1	leg_joint	hinge	torque (N m)    
    2	Torque applied on the foot rotor	-1	1	foot_joint	hinge	torque (N m)    
    Observation Space    
    Observations consist of positional values of different body parts of the hopper, followed by the velocities of those individual parts (their derivatives) with all the positions ordered before all the velocities.    
    By default, observations do not include the x-coordinate of the hopper. It may be included by passing exclude_current_positions_from_observation=False during construction. In that case, the observation space will be Box(-Inf, Inf, (12,), float64) where the first observation represents the x-coordinate of the hopper. Regardless of whether exclude_current_positions_from_observation was set to true or false, the x-coordinate will be returned in info with key "x_position".    
    However, by default, the observation is a Box(-Inf, Inf, (11,), float64) where the elements correspond to the following:    
    Num	Observation	Min	Max	Name (in corresponding XML file)	Joint	Unit    
    0	z-coordinate of the torso (height of hopper)	-Inf	Inf	rootz	slide	position (m)    
    1	angle of the torso	-Inf	Inf	rooty	hinge	angle (rad)    
    2	angle of the thigh joint	-Inf	Inf	thigh_joint	hinge	angle (rad)    
    3	angle of the leg joint	-Inf	Inf	leg_joint	hinge	angle (rad)    
    4	angle of the foot joint	-Inf	Inf	foot_joint	hinge	angle (rad)    
    5	velocity of the x-coordinate of the torso	-Inf	Inf	rootx	slide	velocity (m/s)    
    6	velocity of the z-coordinate (height) of the torso	-Inf	Inf	rootz	slide	velocity (m/s)    
    7	angular velocity of the angle of the torso	-Inf	Inf	rooty	hinge	angular velocity (rad/s)    
    8	angular velocity of the thigh hinge	-Inf	Inf	thigh_joint	hinge	angular velocity (rad/s)    
    9	angular velocity of the leg hinge	-Inf	Inf	leg_joint	hinge	angular velocity (rad/s)    
    10	angular velocity of the foot hinge	-Inf	Inf	foot_joint	hinge	angular velocity (rad/s)    
    excluded	x-coordinate of the torso	-Inf	Inf	rootx	slide	position (m)    
    Rewards    
    The reward consists of three parts:    
    •	healthy_reward: Every timestep that the hopper is healthy (see definition in section “Episode Termination”), it gets a reward of fixed value healthy_reward.    
    •	forward_reward: A reward of hopping forward which is measured as forward_reward_weight * (x-coordinate before action - x-coordinate after action)/dt. dt is the time between actions and is dependent on the frame_skip parameter (fixed to 4), where the frametime is 0.002 - making the default dt = 4 * 0.002 = 0.008. This reward would be positive if the hopper hops forward (positive x direction).    
    •	ctrl_cost: A cost for penalising the hopper if it takes actions that are too large. It is measured as ctrl_cost_weight * sum(action2) where ctrl_cost_weight is a parameter set for the control and has a default value of 0.001    
    The total reward returned is reward = healthy_reward + forward_reward - ctrl_cost and info will also contain the individual reward terms    
    Starting State    
    All observations start in state (0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) with a uniform noise in the range of [-reset_noise_scale, reset_noise_scale] added to the values for stochasticity.    
    Episode End    
    The hopper is said to be unhealthy if any of the following happens:    
    1.	An element of observation[1:] (if exclude_current_positions_from_observation=True, else observation[2:]) is no longer contained in the closed interval specified by the argument healthy_state_range    
    2.	The height of the hopper (observation[0] if exclude_current_positions_from_observation=True, else observation[1]) is no longer contained in the closed interval specified by the argument healthy_z_range (usually meaning that it has fallen)    
    3.	The angle (observation[1] if exclude_current_positions_from_observation=True, else observation[2]) is no longer contained in the closed interval specified by the argument healthy_angle_range    
    If terminate_when_unhealthy=True is passed during construction (which is the default), the episode ends when any of the following happens:    
    1.	Truncation: The episode duration reaches a 1000 timesteps    
    2.	Termination: The hopper is unhealthy    
    If terminate_when_unhealthy=False is passed, the episode is ended only when 1000 timesteps are exceeded.    
    The current observation is $obs.
    Please give your summary, feedback, and instructions. 
    Please limit your input to 2000 words.
    Think step by step.
    """)

    
    env_name = "Hopper-v2"
    llama_paths = []

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    
    data_dir_loc = os.path.join(os.getenv('AMLT_DATA_DIR', "./"))
    if os.path.exists(f"{data_dir_loc}/D4RL/hopper-medium-expert-v2-ldesc_{idx}.pkl"):
        with open(f"{data_dir_loc}/D4RL/hopper-medium-expert-v2-ldesc_{idx}.pkl", "rb") as f:
            llama_paths = pickle.load(f)
    
    output_dir_loc = os.path.join(os.getenv('AMLT_OUTPUT_DIR', "./"))
    os.makedirs(f"{output_dir_loc}/D4RL/", exist_ok=True)
    with open(f"{data_dir_loc}/D4RL/hopper-medium-expert-v2.pkl", "rb") as f:
        paths = pickle.load(f)
        rewards_list = [np.sum(path["rewards"]) for path in paths]    
        path_reward_thd = np.percentile(rewards_list, 70)
        path_num = 0
        existing_path_num = len(llama_paths)
        for i, path in tqdm(enumerate(paths)):
            if i % total_split != idx: continue
            path_reward = np.sum(path["rewards"])
            if path_reward <= path_reward_thd: continue
            if path_num < existing_path_num:
                path_num += 1
                continue
            new_obss = []
            new_rewards = []
            new_terminals = []
            new_actions = []
            prompts = []
            for j, obs in tqdm(enumerate(path["observations"]), leave=False):
                input = question_template.substitute(obs=str(obs))
                prompt = get_local_model_resp(model, tokenizer, input)
                if verbose: print(prompt)
                # if prompt == "ERROR":
                #     continue
                prompt_emb = llmemb_model.get_emb(prompt)
                obs_emb = np.concatenate([obs, prompt_emb], axis=0)
                new_obss.append(obs_emb)
                new_rewards.append(path["rewards"][j])
                new_actions.append(path["actions"][j])
                new_terminals.append(path["terminals"][j])
                prompts.append(prompt)
                # time.sleep(2)
                
                if j == len(path["observations"]) - 1 or j % 10 == 0:
                    path = {
                        "observations": np.array(new_obss),
                        "rewards": np.array(new_rewards),
                        "actions": np.array(new_actions),
                        "terminals": np.array(new_terminals),
                        "prompts": prompts}
                    llama_paths.append(path)
                    
                    with open(f"{output_dir_loc}/D4RL/hopper-medium-expert-v2-ldesc_{idx}.pkl", "wb") as fw:
                        pickle.dump(llama_paths, fw)
                
        

