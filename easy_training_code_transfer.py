#!/usr/bin/env python
# coding: utf-8

# In[44]:


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
cfg.model_name="lomahony/eleuther-pythia6.9b-hh-sft"
cfg.target_name="usvsnsp/pythia-6.9b-ppo"
cfg.layers=[24]
cfg.setting="residual"
cfg.tensor_name="gpt_neox.layers.{layer}"
original_l1_alpha = 5e-3
cfg.l1_alpha=original_l1_alpha
cfg.sparsity=None
cfg.num_epochs=10
cfg.model_batch_size=4
cfg.lr=1e-2
cfg.kl=False
cfg.reconstruction=False
# cfg.dataset_name="NeelNanda/pile-10k"
cfg.dataset_name="Elriggs/openwebtext-100k"
cfg.device="cuda:0"
cfg.ratio = 4
cfg.seed = 0
# cfg.device="cpu"


# In[45]:


tensor_names = [cfg.tensor_name.format(layer=layer) for layer in cfg.layers]


# In[46]:


# Load in the model
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
model = model.to(cfg.device)
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)


# In[ ]:


# Download the dataset
# TODO iteratively grab dataset?
cfg.max_length = 256
cfg.model_batch_size = 8
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


# Initialize New autoencoder
from autoencoders.learned_dict import TiedSAE, UntiedSAE, AnthropicSAE
from torch import nn
params = dict()
n_dict_components = activation_size*cfg.ratio
params["encoder"] = torch.empty((n_dict_components, activation_size), device=cfg.device)
nn.init.xavier_uniform_(params["encoder"])

params["decoder"] = torch.empty((n_dict_components, activation_size), device=cfg.device)
nn.init.xavier_uniform_(params["decoder"])

params["encoder_bias"] = torch.empty((n_dict_components,), device=cfg.device)
nn.init.zeros_(params["encoder_bias"])

params["shift_bias"] = torch.empty((activation_size,), device=cfg.device)
nn.init.zeros_(params["shift_bias"])

autoencoder = AnthropicSAE(  # TiedSAE, UntiedSAE, AnthropicSAE
    # n_feats = n_dict_components, 
    # activation_size=activation_size,
    encoder=params["encoder"],
    encoder_bias=params["encoder_bias"],
    decoder=params["decoder"],
    shift_bias=params["shift_bias"],
)
autoencoder.to_device(cfg.device)
autoencoder.set_grad()
# autoencoder.encoder.requires_grad = True
# autoencoder.encoder_bias.requires_grad = True
# autoencoder.decoder.requires_grad = True
# autoencoder.shift_bias.requires_grad = True
optimizer = torch.optim.Adam(
    [
        autoencoder.encoder, 
        autoencoder.encoder_bias,
        autoencoder.decoder,
        autoencoder.shift_bias,
    ], lr=cfg.lr)


# In[ ]:


# Set target sparsity to 10% of activation_size if not set
if cfg.sparsity is None:
    cfg.sparsity = int(activation_size*0.05)
    print(f"Target sparsity: {cfg.sparsity}")

target_lower_sparsity = cfg.sparsity * 0.9
target_upper_sparsity = cfg.sparsity * 1.1
adjustment_factor = 0.1  # You can set this to whatever you like


# In[ ]:


original_bias = autoencoder.encoder_bias.clone().detach()
# Wandb setup
secrets = json.load(open("secrets.json"))
wandb.login(key=secrets["wandb_key"])
start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
wandb_run_name = f"{cfg.model_name}_{start_time[4:]}_{cfg.sparsity}"  # trim year
print(f"wandb_run_name: {wandb_run_name}")


# In[ ]:


wandb.init(project="sparse coding", config=dict(cfg), name=wandb_run_name)


# In[ ]:


dead_features = torch.zeros(autoencoder.encoder.shape[0])
total_activations = torch.zeros(autoencoder.encoder.shape[0])
max_num_tokens = 100_000_000
save_every = 5_000_000
num_saved_so_far = 0
# Freeze model parameters 
model.eval()
model.requires_grad_(False)
model.to(cfg.device)
last_encoder = autoencoder.encoder.clone().detach()
for i, batch in enumerate(tqdm(token_loader)):
    tokens = batch["input_ids"].to(cfg.device)
    with torch.no_grad(): # As long as not doing KL divergence, don't need gradients for model
        with Trace(model, tensor_names[0]) as ret:
            _ = model(tokens)
            representation = ret.output
            if(isinstance(representation, tuple)):
                representation = representation[0]
    layer_activations = rearrange(representation, "b seq d_model -> (b seq) d_model")
    # activation_saver.save_batch(layer_activations.clone().cpu().detach())

    c = autoencoder.encode(layer_activations)
    x_hat = autoencoder.decode(c)
    
    reconstruction_loss = (x_hat - layer_activations).pow(2).mean()
    l1_loss = torch.norm(c, 1, dim=-1).mean()
    total_loss = reconstruction_loss + cfg.l1_alpha*l1_loss

    dead_features += c.sum(dim=0).cpu()
    total_activations += c.sum(dim=0).cpu()
    if (i % 20 == 0): # Check here so first check is model w/o change
        # self_similarity = torch.cosine_similarity(c, last_encoder, dim=-1).mean().cpu().item()
        # Above is wrong, should be similarity between encoder and last encoder
        self_similarity = torch.cosine_similarity(autoencoder.encoder, last_encoder, dim=-1).mean().cpu().item()
        last_encoder = autoencoder.encoder.clone().detach()
        num_tokens_so_far = i*cfg.max_length*cfg.model_batch_size
        with torch.no_grad():
            sparsity = (c != 0).float().mean(dim=0).sum().cpu().item()
            # Count number of dead_features are zero
            num_dead_features = (dead_features == 0).sum().item()
        print(f"Sparsity: {sparsity:.1f} | Dead Features: {num_dead_features} | Total Loss: {total_loss:.2f} | Reconstruction Loss: {reconstruction_loss:.2f} | L1 Loss: {cfg.l1_alpha*l1_loss:.2f} | l1_alpha: {cfg.l1_alpha:.2e} | Tokens: {num_tokens_so_far} | Self Similarity: {self_similarity:.2f}")
        wandb.log({
            'Sparsity': sparsity,
            'Dead Features': num_dead_features,
            'Total Loss': total_loss.item(),
            'Reconstruction Loss': reconstruction_loss.item(),
            'L1 Loss': (cfg.l1_alpha*l1_loss).item(),
            'l1_alpha': cfg.l1_alpha,
            'Tokens': num_tokens_so_far,
            'Self Similarity': self_similarity
        })
        
        dead_features = torch.zeros(autoencoder.encoder.shape[0])
        
        if(num_tokens_so_far > max_num_tokens):
            print(f"Reached max number of tokens: {max_num_tokens}")
            break
        
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    resample_period = 10000
    if (i % resample_period == 0):
        # RESAMPLING
        with torch.no_grad():
            # Count number of dead_features are zero
            num_dead_features = (total_activations == 0).sum().item()
            
        if num_dead_features > 0:
            print("Resampling!")
            # hyperparams:
            max_resample_tokens = 1000 # the number of token activations that we consider for inserting into the dictionary
            # compute loss of model on random subset of inputs
            resample_loader = setup_token_data(cfg, tokenizer, model, seed=i)
            num_resample_data = 0

            resample_activations = torch.empty(0, activation_size)
            resample_losses = torch.empty(0)

            for resample_batch in resample_loader:
                resample_tokens = resample_batch["input_ids"].to(cfg.device)
                with torch.no_grad(): # As long as not doing KL divergence, don't need gradients for model
                    with Trace(model, tensor_names[0]) as ret:
                        _ = model(resample_tokens)
                        representation = ret.output
                        if(isinstance(representation, tuple)):
                            representation = representation[0]
                layer_activations = rearrange(representation, "b seq d_model -> (b seq) d_model")
                resample_activations = torch.cat((resample_activations, layer_activations.detach().cpu()), dim=0)

                c = autoencoder.encode(layer_activations)
                x_hat = autoencoder.decode(c)
                
                reconstruction_loss = (x_hat - layer_activations).pow(2).mean(dim=-1)
                l1_loss = torch.norm(c, 1, dim=-1)
                temp_loss = reconstruction_loss + cfg.l1_alpha*l1_loss
                
                resample_losses = torch.cat((resample_losses, temp_loss.detach().cpu()), dim=0)
                
                num_resample_data +=layer_activations.shape[0]
                if num_resample_data > max_resample_tokens:
                    break

                
            # sample num_dead_features vectors of input activations
            probabilities = resample_losses**2
            probabilities /= probabilities.sum()
            sampled_indices = torch.multinomial(probabilities, num_dead_features, replacement=True)
            new_vectors = resample_activations[sampled_indices]

            # calculate average encoder norm of alive neurons
            alive_neurons = list((total_activations!=0))
            modified_columns = total_activations==0
            avg_norm = autoencoder.encoder.data[alive_neurons].norm(dim=-1).mean()

            # replace dictionary and encoder weights with vectors
            new_vectors = new_vectors / new_vectors.norm(dim=1, keepdim=True)
            
            params_to_modify = [autoencoder.encoder, autoencoder.encoder_bias]

            current_weights = autoencoder.encoder.data
            current_weights[modified_columns] = (new_vectors.to(cfg.device) * avg_norm * 0.02)
            autoencoder.encoder.data = current_weights

            current_weights = autoencoder.encoder_bias.data
            current_weights[modified_columns] = 0
            autoencoder.encoder_bias.data = current_weights
            
            if hasattr(autoencoder, 'decoder'):
                current_weights = autoencoder.decoder.data
                current_weights[modified_columns] = new_vectors.to(cfg.device)
                autoencoder.decoder.data = current_weights
                params_to_modify += [autoencoder.decoder]

            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    if any(param is d_ for d_ in params_to_modify):
                        # Extract the corresponding rows from m and v
                        m = optimizer.state[param]['exp_avg']
                        v = optimizer.state[param]['exp_avg_sq']
                        
                        # Update the m and v values for the modified columns
                        m[modified_columns] = 0  # Reset moving average for modified columns
                        v[modified_columns] = 0  # Reset squared moving average for modified columns
        
        total_activations = torch.zeros(autoencoder.encoder.shape[0])

    if ((num_tokens_so_far+1) % 5_000_000==0):
        model_save_name = cfg.model_name.split("/")[-1]
        save_name = f"{model_save_name}_sp{cfg.sparsity}_r{cfg.ratio}_{tensor_names[0]}_ckpt{num_saved_so_far}"  # trim year

        # Make directory traiend_models if it doesn't exist
        import os
        if not os.path.exists("trained_models"):
            os.makedirs("trained_models")
        # Save model
        torch.save(autoencoder, f"trained_models/{save_name}.pt")
        
        num_saved_so_far += 1

    # Running sparsity check
    if(num_tokens_so_far > 200000):
        if(i % 100 == 0):
            with torch.no_grad():
                sparsity = (c != 0).float().mean(dim=0).sum().cpu().item()
            if sparsity > target_upper_sparsity:
                cfg.l1_alpha *= (1 + adjustment_factor)
            elif sparsity < target_lower_sparsity:
                cfg.l1_alpha *= (1 - adjustment_factor)
            # print(f"Sparsity: {sparsity:.1f} | l1_alpha: {cfg.l1_alpha:.2e}")


# In[24]:


model_save_name = cfg.model_name.split("/")[-1]
save_name = f"{model_save_name}_sp{cfg.sparsity}_r{cfg.ratio}_{tensor_names[0]}"  # trim year

# Make directory traiend_models if it doesn't exist
import os
if not os.path.exists("trained_models"):
    os.makedirs("trained_models")
# Save model
torch.save(autoencoder, f"trained_models/{save_name}.pt")


# In[18]:


wandb.finish()


# In[69]:


target_model = AutoModelForCausalLM.from_pretrained(cfg.target_name).cpu()

model_save_name = cfg.model_name.split("/")[-1]
save_name = f"{model_save_name}_sp{cfg.sparsity}_r{cfg.ratio}_{tensor_names[0]}"  # trim year
autoencoder = torch.load(f"trained_models/{save_name}.pt")


# In[70]:


# Initialize New autoencoder
from autoencoders.learned_dict import TiedSAE, UntiedSAE, AnthropicSAE, TransferSAE
from torch import nn

# params["decoder"] = torch.empty((n_dict_components, activation_size), device=cfg.device)
# nn.init.xavier_uniform_(params["decoder"])

params["decoder_bias"] = torch.empty((activation_size,), device=cfg.device)
nn.init.zeros_(params["decoder_bias"])

transfer_autoencoder = TransferSAE(
    # n_feats = n_dict_components, 
    # activation_size=activation_size,
    autoencoder,
    decoder=autoencoder.get_learned_dict().detach().clone(),
    decoder_bias=autoencoder.shift_bias.detach().clone(),
)
transfer_autoencoder.to_device(cfg.device)

# Set gradient to true for decoder only- only training decoder on transfer
transfer_autoencoder.set_grad()
optimizer = torch.optim.Adam(
    [
        transfer_autoencoder.decoder,
        transfer_autoencoder.decoder_bias,
    ], lr=cfg.lr)


# In[71]:


# Wandb setup
secrets = json.load(open("secrets.json"))
wandb.login(key=secrets["wandb_key"])
start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
wandb_run_name = f"{cfg.target_name}_transfer_{start_time[4:]}_{cfg.sparsity}"  # trim year
print(f"wandb_run_name: {wandb_run_name}")
wandb.init(project="sparse coding", config=dict(cfg), name=wandb_run_name)


# In[72]:


def get_activations(model, inputs):
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


# In[73]:


# Training transfer autoencoder
token_loader = setup_token_data(cfg, tokenizer, model, seed=cfg.seed)
dead_features = torch.zeros(transfer_autoencoder.encoder.shape[0])
max_num_tokens = 100_000_000
# Freeze model parameters 
model = model.to(cfg.device)
target_model = target_model.cpu()
target_model.eval()
target_model.requires_grad_(False)

last_decoder = transfer_autoencoder.decoder.clone().detach()
model_on_gpu = True

saved_inputs = []
i = 0 # counts all optimization steps
for k, (batch) in enumerate(token_loader):
    saved_inputs.append(batch["input_ids"].to(cfg.device))
    
    if (k+1)%500==0:
        # compute base and target model activations
        if model_on_gpu:
            base_activations = get_activations(model, saved_inputs)
            model = model.cpu()
            target_model = target_model.to(cfg.device)
        target_activations = get_activations(target_model, saved_inputs)
        if not model_on_gpu:
            target_model = target_model.cpu()
            model = model.to(cfg.device)
            base_activations = get_activations(model, saved_inputs)
        model_on_gpu = not model_on_gpu

        # wipe saved inputs
        saved_inputs = []
        
        # train autoencoder on activations:
        for (base_activation, target_activation) in (zip(base_activations, target_activations)):
            c = transfer_autoencoder.encode(base_activation.to(cfg.device))
            x_hat = transfer_autoencoder.decode(c)
            
            reconstruction_loss = (x_hat - target_activation.to(cfg.device)).pow(2).mean()
            total_loss = reconstruction_loss # NO L1 LOSS

            dead_features += c.sum(dim=0).cpu()
            if (i % 500 == 0): # Check here so first check is model w/o change
                self_similarity = torch.cosine_similarity(transfer_autoencoder.decoder, last_decoder, dim=-1).mean().cpu().item()
                last_decoder = transfer_autoencoder.decoder.clone().detach()
                num_tokens_so_far = i*cfg.max_length*cfg.model_batch_size
                with torch.no_grad():
                    sparsity = (c != 0).float().mean(dim=0).sum().cpu().item()
                    # Count number of dead_features are zero
                    num_dead_features = (dead_features == 0).sum().item()
                print(f"Sparsity: {sparsity:.1f} | Dead Features: {num_dead_features} | Reconstruction Loss: {reconstruction_loss:.2f} | Tokens: {num_tokens_so_far} | Self Similarity: {self_similarity:.2f}")
                wandb.log({
                    'Sparsity': sparsity,
                    'Dead Features': num_dead_features,
                    'Reconstruction Loss': reconstruction_loss.item(),
                    'Tokens': num_tokens_so_far,
                    'Self Similarity': self_similarity
                })
                dead_features = torch.zeros(transfer_autoencoder.encoder.shape[0])
                
                if(num_tokens_so_far > max_num_tokens):
                    print(f"Reached max number of tokens: {max_num_tokens}")
                    break

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            i+=1

            # # Running sparsity check
            # if(num_tokens_so_far > 500000):
            #     if(i % 200 == 0):
            #         with torch.no_grad():
            #             sparsity = (c != 0).float().mean(dim=0).sum().cpu().item()
            #         if sparsity > target_upper_sparsity:
            #             cfg.l1_alpha *= (1 + adjustment_factor)
            #         elif sparsity < target_lower_sparsity:
            #             cfg.l1_alpha *= (1 - adjustment_factor)
            #         # print(f"Sparsity: {sparsity:.1f} | l1_alpha: {cfg.l1_alpha:.2e}")


# In[ ]:


model_save_name = cfg.target_name.split("/")[-1]
save_name = f"{model_save_name}_transfer_sp{cfg.sparsity}_r{cfg.ratio}_{tensor_names[0]}"  # trim year

# Make directory traiend_models if it doesn't exist
import os
if not os.path.exists("trained_models"):
    os.makedirs("trained_models")
# Save model
torch.save(transfer_autoencoder, f"trained_models/{save_name}.pt")


# In[74]:


wandb.finish()


# In[75]:


# Autoencoder vs Transfer Autoencoder Similarity
torch.cosine_similarity(autoencoder.decoder,transfer_autoencoder.decoder, dim=0)

