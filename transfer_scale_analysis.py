#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch 
import argparse
from utils import dotdict
from activation_dataset import setup_token_data
import wandb
import json
from datetime import datetime
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt

cfg = dotdict()
# models: "EleutherAI/pythia-70m-deduped", "EleutherAI/pythia-6.9b", "usvsnsp/pythia-6.9b-ppo", "lomahony/eleuther-pythia6.9b-hh-sft", "reciprocate/dahoas-gptj-rm-static"
cfg.model_name="EleutherAI/pythia-6.9b"
cfg.target_name="lomahony/eleuther-pythia6.9b-hh-sft"
cfg.layers=[10]
cfg.setting="residual"
# cfg.tensor_name="gpt_neox.layers.{layer}"
cfg.tensor_name="transformer.h.{layer}"
cfg.target_tensor_name="gpt_neox.layers.{layer}"
original_l1_alpha = 8e-4
cfg.l1_alpha=original_l1_alpha
cfg.sparsity=None
cfg.num_epochs=10
cfg.model_batch_size=8
cfg.lr=1e-3
cfg.kl=False
cfg.reconstruction=False
# cfg.dataset_name="NeelNanda/pile-10k"
cfg.dataset_name="Elriggs/openwebtext-100k"
cfg.device="cuda:0"
cfg.ratio = 4
cfg.seed = 0
# cfg.device="cpu"


# In[ ]:


base_name = "rm"  # base, rm
capitalized_base_name = "RM" # Base, RM
target_name = "ppo"  # sft, ppo
finetuning_type = "RLHF"  # SFT, RLHF


# In[ ]:


tensor_names = [cfg.tensor_name.format(layer=layer) for layer in cfg.layers]
target_tensor_names = [cfg.target_tensor_name.format(layer=layer) for layer in cfg.layers]


# In[ ]:


# Load base and target autoencoders
from autoencoders.learned_dict import TiedSAE, UntiedSAE, AnthropicSAE, TransferSAE
from torch import nn


# save_name = f"rm_sae_gptj" if base_name=="rm" else f"base_sae_6b"  # trim year
# autoencoder = torch.load(f"trained_models/{save_name}.pt")
# print(f"autoencoder loaded from {save_name}")
# autoencoder.to_device(cfg.device)

# save_name = f"ppo_sae_6b" if target_name=="ppo" else f"sft_sae_6b"
# target_autoencoder = torch.load(f"trained_models/{save_name}.pt")
# print(f"target_autoencoder loaded from {save_name}")
# target_autoencoder.to_device(cfg.device)


# In[ ]:


# Initialize New transfer autoencoder
# from autoencoders.learned_dict import TiedSAE, UntiedSAE, AnthropicSAE, TransferSAE
from torch import nn

modes = ["scale", "rotation", "bias", "free"]
mode = "free"

mode_tsae = torch.load(f"trained_models/transfer_{base_name}_{target_name}_6b_{mode}.pt")
mode_tsae.to_device(cfg.device)



# In[ ]:


dead_features = torch.load(f"trained_models/{base_name}_dead_features.pt")
live_features = dead_features > 0


# In[ ]:


import matplotlib.pyplot as plt
scales = mode_tsae.get_feature_scales()[live_features]
# feature_vectors = transfer_free.get_learned_dict()[live_features]
list_scales = list(scales.abs().detach().cpu().numpy())
print(sorted(list_scales))
# plt.plot(scales.detach().cpu().numpy())


# In[ ]:


plt.plot(sorted(list_scales))
plt.title(f"Feature Scaling from {capitalized_base_name} to {finetuning_type}'d LLM")
plt.xlabel("Feature")
plt.savefig(f"feature_scaling_{base_name}_{target_name}_{mode}")

