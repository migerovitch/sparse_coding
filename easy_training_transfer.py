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
# models: "EleutherAI/pythia-70m-deduped", "usvsnsp/pythia-6.9b-ppo", "lomahony/eleuther-pythia6.9b-hh-sft", "reciprocate/dahoas-gptj-rm-static"
cfg.model_name="reciprocate/dahoas-gptj-rm-static"
cfg.target_name="usvsnsp/pythia-6.9b-ppo"
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


tensor_names = [cfg.tensor_name.format(layer=layer) for layer in cfg.layers]
target_tensor_names = [cfg.target_tensor_name.format(layer=layer) for layer in cfg.layers]


# In[ ]:


# Load in the model
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
model = model.to(cfg.device)
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)


# In[ ]:


# Download the dataset
# TODO iteratively grab dataset?
cfg.max_length = 256
token_loader = setup_token_data(cfg, tokenizer, model, seed=cfg.seed)
num_tokens = cfg.max_length*cfg.model_batch_size*len(token_loader)
print(f"Number of tokens: {num_tokens}")


# In[ ]:


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


# In[ ]:


# Set target sparsity to 10% of activation_size if not set
if cfg.sparsity is None:
    cfg.sparsity = int(activation_size*0.05)
    print(f"Target sparsity: {cfg.sparsity}")

target_lower_sparsity = cfg.sparsity * 0.9
target_upper_sparsity = cfg.sparsity * 1.1
adjustment_factor = 0.1  # You can set this to whatever you like


# In[ ]:


# Initialize New autoencoder
from autoencoders.learned_dict import TiedSAE, UntiedSAE, AnthropicSAE, TransferSAE
from torch import nn


target_model = AutoModelForCausalLM.from_pretrained(cfg.target_name).cpu()

model_save_name = cfg.model_name.split("/")[-1]
save_name = f"{model_save_name}_sp{cfg.sparsity}_r{cfg.ratio}_{tensor_names[0]}"  # trim year
autoencoder = torch.load(f"trained_models/{save_name}.pt")
print(f"autoencoder loaded from f{save_name}")

autoencoder.to_device(cfg.device)


# In[ ]:


# Initialize New transfer autoencoder
from autoencoders.learned_dict import TiedSAE, UntiedSAE, AnthropicSAE, TransferSAE
from torch import nn

modes = ["scale", "rotation", "bias", "free"]
transfer_autoencoders = []
for mode in modes:
    mode_tsae = TransferSAE(
        # n_feats = n_dict_components, 
        # activation_size=activation_size,
        autoencoder,
        decoder=autoencoder.get_learned_dict().detach().clone(),
        decoder_bias=autoencoder.shift_bias.detach().clone(),
        mode=mode,
    )
    mode_tsae.to_device(cfg.device)
    transfer_autoencoders.append(mode_tsae)

optimizers = []

# Set gradient to true for decoder only- only training decoder on transfer

for tsae in transfer_autoencoders:
    tsae.set_grad()
    optimizers.append(
        torch.optim.Adam(tsae.parameters(), lr=cfg.lr)
    )



# In[ ]:


# Wandb setup
secrets = json.load(open("secrets.json"))
wandb.login(key=secrets["wandb_key"])
start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
wandb_run_name = f"{cfg.target_name}_transfer_{start_time[4:]}_{cfg.sparsity}"  # trim year
print(f"wandb_run_name: {wandb_run_name}")
wandb.init(project="sparse coding", config=dict(cfg), name=wandb_run_name)


# In[ ]:


def compute_activations(model, inputs, layer_name):
    acts = []
    for tokens in inputs:
        with torch.no_grad(): # As long as not doing KL divergence, don't need gradients for model
            with Trace(model, layer_name) as ret:
                _ = model(tokens)
                representation = ret.output
                if(isinstance(representation, tuple)):
                    representation = representation[0]
        layer_activations = rearrange(representation, "b seq d_model -> (b seq) d_model")
        acts.append(layer_activations.cpu())
    return acts


# In[ ]:


def generate_activations(model, target_model, token_loader, cfg, model_on_gpu=True, num_batches=500):
    saved_inputs = []
    for k, (batch) in enumerate(token_loader):
        saved_inputs.append(batch["input_ids"].to(cfg.device))
        
        if (k+1)%num_batches==0:
            # compute base and target model activations
            if model_on_gpu:
                base_activations = compute_activations(model, saved_inputs, layer_name=tensor_names[0])
                model = model.cpu()
                target_model = target_model.to(cfg.device)
            target_activations = compute_activations(target_model, saved_inputs, layer_name=target_tensor_names[0])
            if not model_on_gpu:
                target_model = target_model.cpu()
                model = model.to(cfg.device)
                base_activations = compute_activations(model, saved_inputs, layer_name=tensor_names[0])
            model_on_gpu = not model_on_gpu
            
            for base_activation, target_activation in zip(base_activations, target_activations):
                yield base_activation, target_activation

            # wipe saved inputs
            saved_inputs = []
    pass


# In[ ]:


# Training transfer autoencoder
token_loader = setup_token_data(cfg, tokenizer, model, seed=cfg.seed)
dead_features = torch.zeros(autoencoder.encoder.shape[0])
# auto_dead_features = torch.zeros(autoencoder.encoder.shape[0])

max_num_tokens = 30_000_000
log_every=100
# Freeze model parameters 
model = model.to(cfg.device)
target_model = target_model.cpu()
target_model.eval()
target_model.requires_grad_(False)

last_decoders = dict([(modes[i],transfer_autoencoders[i].decoder.clone().detach()) for i in range(len(transfer_autoencoders))])
model_on_gpu = True

saved_inputs = []
i = 0 # counts all optimization steps
num_saved_so_far = 0
print("starting loop")
for (base_activation, target_activation) in tqdm(generate_activations(model, target_model, token_loader, cfg, model_on_gpu=model_on_gpu, num_batches=500), 
                                                 total=int(max_num_tokens/(cfg.max_length*cfg.model_batch_size))):
    c = autoencoder.encode(base_activation.to(cfg.device))
    x_hat = autoencoder.decode(c)
    
    autoencoder_loss = (x_hat - target_activation.to(cfg.device)).pow(2).mean()
    dead_features += c.sum(dim=0).cpu()
    
    wandb_log = {}
    
    for tsae, mode, optimizer in zip(transfer_autoencoders, modes, optimizers):
        x_hat = tsae.decode(c)
        
        reconstruction_loss = (x_hat - target_activation.to(cfg.device)).pow(2).mean()
        total_loss = reconstruction_loss # NO L1 LOSS

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

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    if (i % log_every == 0):
        with torch.no_grad():
            sparsity = (c != 0).float().mean(dim=0).sum().cpu().item()
            # Count number of dead_features are zero
            num_dead_features = (dead_features == 0).sum().item()
        print(f"Sparsity: {sparsity:.1f} | Dead Features: {num_dead_features} | Reconstruction Loss: {autoencoder_loss:.2f} | Tokens: {num_tokens_so_far}")
        dead_features = torch.zeros(autoencoder.encoder.shape[0])
        wandb_log.update({
                f'SAE Sparsity': sparsity,
                f'Dead Features': num_dead_features,
                f'SAE Reconstruction Loss': autoencoder_loss.item(),
                f'Tokens': num_tokens_so_far,
            })
        wandb.log(wandb_log)
    i+=1
    
    if ((i+2) % 2000==0): # save periodically but before big changes
        for tsae, mode in zip(transfer_autoencoders, modes):
            model_save_name = cfg.model_name.split("/")[-1]
            save_name = f"{model_save_name}_transfer_{mode}_sp{cfg.sparsity}_r{cfg.ratio}_{tensor_names[0]}_ckpt{num_saved_so_far}" 

            # Make directory traiend_models if it doesn't exist
            import os
            if not os.path.exists("trained_models"):
                os.makedirs("trained_models")
            # Save model
            torch.save(tsae, f"trained_models/{save_name}.pt")
        
        num_saved_so_far += 1
                
    
    num_tokens_so_far = i*cfg.max_length*cfg.model_batch_size
    if(num_tokens_so_far > max_num_tokens):
        print(f"Reached max number of tokens: {max_num_tokens}")
        break
    


# In[ ]:


# Save model at end

for tsae, mode in zip(transfer_autoencoders, modes):
    model_save_name = cfg.model_name.split("/")[-1]
    save_name = f"{model_save_name}_transfer_{mode}_sp{cfg.sparsity}_r{cfg.ratio}_{tensor_names[0]}" 

    # Make directory traiend_models if it doesn't exist
    import os
    if not os.path.exists("trained_models"):
        os.makedirs("trained_models")
    # Save model
    torch.save(tsae, f"trained_models/{save_name}.pt")


# In[ ]:


wandb.finish()

