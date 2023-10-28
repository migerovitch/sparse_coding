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
# models: "EleutherAI/pythia-70m-deduped", "usvsnsp/pythia-6.9b-ppo", "lomahony/eleuther-pythia6.9b-hh-sft"
cfg.model_name="EleutherAI/pythia-6.9b"
cfg.target_name="lomahony/eleuther-pythia6.9b-hh-sft"
cfg.layers=[10]
cfg.setting="residual"
cfg.tensor_name="gpt_neox.layers.{layer}"
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


# In[3]:


tensor_names = [cfg.tensor_name.format(layer=layer) for layer in cfg.layers]


# In[4]:


# Load in the model
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
model = model.to(cfg.device)
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)


# In[6]:


# Download the dataset
cfg.max_length = 256
token_loader = setup_token_data(cfg, tokenizer, model, seed=cfg.seed, split="train")
num_tokens = cfg.max_length*cfg.model_batch_size*len(token_loader)
print(f"Number of tokens: {num_tokens}")


# In[8]:


# Run 1 datapoint on model to get the activation size
from baukit import Trace

text = "1"
tokens = tokenizer(text, return_tensors="pt").input_ids.to(cfg.device)
# Your activation name will be different. In the next cells, we will show you how to find it.
with torch.no_grad():
    with Trace(model, tensor_names[0]) as ret:
        _ = model(tokens)
        representation = ret.output
        # check if instance tuple
        if(isinstance(representation, tuple)):
            representation = representation[0]
        activation_size = representation.shape[-1]
print(f"Activation size: {activation_size}")


# In[9]:


# Set target sparsity to 10% of activation_size if not set
if cfg.sparsity is None:
    cfg.sparsity = int(activation_size*0.05)
    print(f"Target sparsity: {cfg.sparsity}")

target_lower_sparsity = cfg.sparsity * 0.9
target_upper_sparsity = cfg.sparsity * 1.1
adjustment_factor = 0.1  # You can set this to whatever you like


# In[10]:


# Load base and target autoencoders
from autoencoders.learned_dict import TiedSAE, UntiedSAE, AnthropicSAE, TransferSAE
from torch import nn


target_model = AutoModelForCausalLM.from_pretrained(cfg.target_name).cpu()

save_name = f"base_sae_6b"  # trim year
autoencoder = torch.load(f"trained_models/{save_name}.pt")
print(f"autoencoder loaded from f{save_name}")
autoencoder.to_device(cfg.device)

save_name = f"sft_sae_6b" 
target_autoencoder = torch.load(f"trained_models/{save_name}.pt")
print(f"target_autoencoder loaded from f{save_name}")
target_autoencoder.to_device(cfg.device)


# In[11]:


# Initialize New transfer autoencoder
from autoencoders.learned_dict import TiedSAE, UntiedSAE, AnthropicSAE, TransferSAE
from torch import nn

modes = ["scale", "rotation", "bias", "free"]
transfer_autoencoders = []
for mode in modes:
    # mode_tsae = TransferSAE(
    #     # n_feats = n_dict_components, 
    #     # activation_size=activation_size,
    #     autoencoder,
    #     decoder=autoencoder.get_learned_dict().detach().clone(),
    #     decoder_bias=autoencoder.shift_bias.detach().clone(),
    #     mode=mode,
    # )
    mode_tsae = torch.load(f"trained_models/transfer_base_sft_6b_{mode}.pt")
    mode_tsae.to_device(cfg.device)
    transfer_autoencoders.append(mode_tsae)



# In[32]:


# Wandb setup
secrets = json.load(open("secrets.json"))
wandb.login(key=secrets["wandb_key"])
start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
wandb_run_name = f"testing_{cfg.target_name}_transfer_{start_time[4:]}"  # trim year
print(f"wandb_run_name: {wandb_run_name}")
wandb.init(project="sparse coding", config=dict(cfg), name=wandb_run_name)


# In[14]:


def compute_activations(model, inputs):
    acts = []
    for tokens in inputs:
        with torch.no_grad(): # As long as not doing KL divergence, don't need gradients for model
            with Trace(model, tensor_names[0]) as ret:
                _ = model(tokens)
                representation = ret.output
                if(isinstance(representation, tuple)):
                    representation = representation[0]
        layer_activations = rearrange(representation, "b seq d_model -> (b seq) d_model")
        acts.append(layer_activations.cpu())
    return acts


# In[15]:


def generate_activations(model, target_model, token_loader, cfg, model_on_gpu=True, num_batches=500):
    saved_inputs = []
    for k, (batch) in enumerate(token_loader):
        saved_inputs.append(batch["input_ids"].to(cfg.device))
        
        if (k+1)%num_batches==0:
            # compute base and target model activations
            if model_on_gpu:
                base_activations = compute_activations(model, saved_inputs)
                model = model.cpu()
                target_model = target_model.to(cfg.device)
            target_activations = compute_activations(target_model, saved_inputs)
            if not model_on_gpu:
                target_model = target_model.cpu()
                model = model.to(cfg.device)
                base_activations = compute_activations(model, saved_inputs)
            model_on_gpu = not model_on_gpu
            
            for base_activation, target_activation in zip(base_activations, target_activations):
                yield base_activation, target_activation

            # wipe saved inputs
            saved_inputs = []
    pass


# In[48]:


# Testing transfer autoencoders
token_loader = setup_token_data(cfg, tokenizer, model, seed=cfg.seed, split="train")
dead_features = torch.zeros(autoencoder.encoder.shape[0])
target_dead_features = torch.zeros(autoencoder.encoder.shape[0])
sft_dead_features = torch.zeros(autoencoder.encoder.shape[0])

max_num_tokens = 300_000
log_every=100
# Freeze model parameters 
target_model = target_model.cpu()
target_model.eval()
model = model.to(cfg.device)
model.eval()

target_model.requires_grad_(False)
model.requires_grad_(False)

last_decoders = dict([(modes[i],transfer_autoencoders[i].decoder.clone().detach()) for i in range(len(transfer_autoencoders))])
model_on_gpu = True

saved_inputs = []
i = 0 # counts all optimization steps
num_saved_so_far = 0
print("starting loop")

auto_total_loss = 0
auto_base_loss = 0
auto_sft_loss = 0

target_base_loss = 0
target_sft_loss = 0

target_total_loss = 0
total_losses = dict((mode,0) for mode in modes)

for (base_activation, target_activation) in tqdm(generate_activations(model, target_model, token_loader, cfg, model_on_gpu=model_on_gpu, num_batches=100), 
                                                 total=int(max_num_tokens/(cfg.max_length*cfg.model_batch_size))):
    
    with torch.no_grad():
        c = autoencoder.encode(base_activation.to(cfg.device))
        x_hat = autoencoder.decode(c)
        autoencoder_loss = (x_hat - target_activation.to(cfg.device)).pow(2).mean()
        auto_total_loss += autoencoder_loss
        auto_base_loss += (x_hat - base_activation.to(cfg.device)).pow(2).mean()
        c_sft = autoencoder.encode(target_activation.to(cfg.device))
        auto_sft_loss += (autoencoder.decode(c_sft) - target_activation.to(cfg.device)).pow(2).mean()
        dead_features += c.sum(dim=0).cpu()
        
        
        target_c = target_autoencoder.encode(base_activation.to(cfg.device))
        target_x_hat = target_autoencoder.decode(c)
        target_autoencoder_loss = (target_x_hat - target_activation.to(cfg.device)).pow(2).mean()
        target_total_loss += target_autoencoder_loss
        target_base_loss += (target_x_hat - base_activation.to(cfg.device)).pow(2).mean()
        target_c_sft = target_autoencoder.encode(target_activation.to(cfg.device))
        target_sft_loss += (target_autoencoder.decode(target_c_sft) - target_activation.to(cfg.device)).pow(2).mean()
        target_dead_features += target_c.sum(dim=0).cpu()
        sft_dead_features += target_c_sft.sum(dim=0).cpu()
    
    wandb_log = {}
    
    for tsae, mode in zip(transfer_autoencoders, modes):
        with torch.no_grad():
            x_hat = tsae.decode(c)
        
        reconstruction_loss = (x_hat - target_activation.to(cfg.device)).pow(2).mean()
        total_loss = reconstruction_loss # NO L1 LOSS
        total_losses[mode] += total_loss

        if (i % log_every == 0): # Check here so first check is model w/o change
            self_similarity = torch.cosine_similarity(tsae.decoder, last_decoders[mode], dim=-1).mean().cpu().item()
            last_decoders[mode] = tsae.decoder.clone().detach()
            num_tokens_so_far = i*cfg.max_length*cfg.model_batch_size
            with torch.no_grad():
                sparsity = (c != 0).float().mean(dim=0).sum().cpu().item()
            print(f"Reconstruction Loss: {reconstruction_loss:.2f} | Tokens: {num_tokens_so_far} | Self Similarity: {self_similarity:.2f}")
            wandb_log.update({
                f'{mode} Reconstruction Loss': reconstruction_loss.item(),
                f'{mode} Self Similarity': self_similarity
            })

    if (i % log_every == 0):
        with torch.no_grad():
            sparsity = (c != 0).float().mean(dim=0).sum().cpu().item()
            num_dead_features = (dead_features == 0).sum().item()
            
            target_sparsity = (target_c != 0).float().mean(dim=0).sum().cpu().item()
            target_num_dead_features = (target_dead_features == 0).sum().item()
            
        print(f"Sparsity: {sparsity:.1f} | Dead Features: {num_dead_features} | Reconstruction Loss: {autoencoder_loss:.2f} | Tokens: {num_tokens_so_far}")
        
        wandb_log.update({  # Base SAE log
                f'SAE Sparsity': sparsity,
                f'Dead Features': num_dead_features,
                f'SAE Reconstruction Loss': autoencoder_loss.item(),
                f'Tokens': num_tokens_so_far,
            })
        
        wandb_log.update({  # Target SAE log
                f'Target SAE Sparsity': target_sparsity,
                f'Target Dead Features': target_num_dead_features,
                f'Target SAE Reconstruction Loss': target_autoencoder_loss.item(),
            })
        
        # Non transfer statistics (only base, or only sft)
        with torch.no_grad():
            sft_sparsity = (c_sft != 0).float().mean(dim=0).sum().cpu().item()            
            target_sft_sparsity = (target_c_sft != 0).float().mean(dim=0).sum().cpu().item()
            num_sft_dead_features = (sft_dead_features == 0).sum().item()
            
        wandb_log.update({  # Base only and Target only losses
                f'Sparsity on SFT': sft_sparsity,
                f'Target Sparsity on SFT': target_sft_sparsity,
                f'SFT Dead Features': num_sft_dead_features,
            })
        wandb.log(wandb_log)
    i+=1
    
                
    
    num_tokens_so_far = i*cfg.max_length*cfg.model_batch_size
    if(num_tokens_so_far > max_num_tokens):
        print(f"Reached max number of tokens: {max_num_tokens}")
        break
    


# In[49]:


# log total average loss and finish wandb
wandb_log = {
    'SAE Average Loss': auto_total_loss/i,
    'Target SAE Average Loss': target_total_loss/i,
    
    'SAE Average Loss on Base': auto_base_loss/i,
    'Target SAE Average Loss on Base': target_base_loss/i,
    
    'SAE Average Loss on SFT': auto_sft_loss/i,
    'Target SAE Average Loss on SFT': target_sft_loss/i,
    }
for mode in modes:
    wandb_log.update({  # Target SAE log
                    f'{mode} Average Loss': total_losses[mode]/i,
                })
    
wandb.log(wandb_log)
wandb.finish()


# In[51]:


import pprint

# Prints the nicely formatted dictionary
pprint.pprint(wandb_log)


# In[46]:


auto_and_target_losses = [
    auto_total_loss,
    auto_base_loss,
    auto_sft_loss,
    target_base_loss,
    target_sft_loss,
    target_total_loss
]

print([x/i for x in auto_and_target_losses])


# In[59]:


# save dead features
import os
if not os.path.exists("trained_models"):
    os.makedirs("trained_models")
# Save model
torch.save(dead_features, f"trained_models/base_dead_features.pt")
torch.save(target_dead_features, f"trained_models/target_dead_features.pt")
torch.save(sft_dead_features, f"trained_models/sft_dead_features.pt")


# In[ ]:




