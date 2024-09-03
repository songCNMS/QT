import gym
import numpy as np

import collections
import pickle
from torch import nn
import d4rl
from envs_info.m_envs import get_state_description
from tqdm import tqdm

from transformers import (
    LlamaConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2Model,
    GPT2Tokenizer,
)
import torch

datasets = []
names = [ 'walker2d-medium-replay-v2']




class LLMEmbModel(nn.Module):
    def __init__(self, llm_model, llm_dim, llm_layers, device):
        super(LLMEmbModel, self).__init__()
        self.d_llm = llm_dim
        self.llm_model = llm_model
        self.llm_layers = llm_layers
        self.device = device

        if self.llm_model == "LLAMA":
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained(
                "meta-llama/Meta-Llama-3-8B"
            )
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
                    ignore_mismatched_sizes=True,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    # 'huggyllama/llama-7b',
                    "meta-llama/Meta-Llama-3-8B",
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except (
                EnvironmentError
            ):  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    # 'huggyllama/llama-7b',
                    "meta-llama/Meta-Llama-3-8B",
                    trust_remote_code=True,
                    local_files_only=False,
                )
        elif self.llm_model == "GPT2":
            self.gpt2_config = GPT2Config.from_pretrained("openai-community/gpt2")
            self.gpt2_config.num_hidden_layers = self.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                "openai-community/gpt2",
                trust_remote_code=True,
                local_files_only=False,
                config=self.gpt2_config,
            )
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                "openai-community/gpt2", trust_remote_code=True, local_files_only=False
            )

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = "[PAD]"
            self.tokenizer.add_special_tokens({"pad_token": pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.word_embeddings = self.llm_model.get_input_embeddings().weight.to(self.device)
        self.vocab_size = self.word_embeddings.shape[0]

    def get_emb(self, prompt):
        prompt = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).input_ids
        prompt_embeddings = torch.mean(
            self.llm_model.get_input_embeddings()(prompt.to(self.device)), axis=0
        )
        return prompt_embeddings[0].detach().cpu().numpy()


# for env_name in names:
def download_dataset(env_name, device, file_loc, with_emb=False):
    if with_emb:
        llm_model = LLMEmbModel('GPT2', 768, 12, device).to(device)
    name = env_name
    env = gym.make(name)
    print(env.observation_space)
    dataset = env.get_dataset()
    print(dataset.keys())
    
    if 'hopper' in env_name:
        ori_env_name = "Hopper-v2"
    elif 'halfcheetah' in env_name:
        ori_env_name = "HalfCheetah-v2"
    elif 'walker' in env_name:
        ori_env_name = "Walker2d-v2"
    else:
        raise NotImplemented

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    paths = []
    for i in tqdm(range(N)):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == 1000-1)
        for k in ['observations', 'actions', 'rewards', 'terminals']:
            if k == 'observations' and with_emb:
                prompt = get_state_description(env, dataset[k][i], ori_env_name)
                prompt_emb = llm_model.get_emb(prompt)
                obs_emb = np.concatenate([dataset[k][i], prompt_emb], axis=0)
                data_[k].append(obs_emb)
            else:
                data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            paths.append(episode_data)
            data_ = collections.defaultdict(list)
        episode_step += 1

    returns = np.array([np.sum(p['rewards']) for p in paths])
    num_samples = np.sum([p['rewards'].shape[0] for p in paths])
    print(f'Number of samples collected: {num_samples}')
    print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

    with open(file_loc, 'wb') as f:
        pickle.dump(paths, f)
