import torch 
import argparse
from utils import dotdict
from activation_dataset import setup_token_data
import wandb
import json
from datetime import datetime
from tqdm import tqdm
from einops import rearrange
import numpy as np

# from standard_metrics import run_with_model_intervention, perplexity_under_reconstruction, mean_nonzero_activations
# Create 
# # make an argument parser directly below
# parser = argparse.ArgumentParser()
# parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m-deduped")
# parser.add_argument("--layer", type=int, default=4)
# parser.add_argument("--setting", type=str, default="residual")
# parser.add_argument("--l1_alpha", type=float, default=3e-3)
# parser.add_argument("--num_epochs", type=int, default=10)
# parser.add_argument("--model_batch_size", type=int, default=4)
# parser.add_argument("--lr", type=float, default=1e-3)
# parser.add_argument("--kl", type=bool, default=False)
# parser.add_argument("--reconstruction", type=bool, default=False)
# parser.add_argument("--dataset_name", type=str, default="NeelNanda/pile-10k")
# parser.add_argument("--device", type=str, default="cuda:4")

# args = parser.parse_args()
cfg = dotdict()
# cfg.model_name="EleutherAI/pythia-70m-deduped"
# cfg.model_name="usvsnsp/pythia-6.9b-rm-full-hh-rlhf"
cfg.model_name="reciprocate/dahoas-gptj-rm-static"
cfg.total_layers = 10
cfg.layers=[i for i in range(cfg.total_layers)]
cfg.setting="residual"
# cfg.tensor_name="gpt_neox.layers.{layer}"
cfg.tensor_name="transformer.h.{layer}"
cfg.l1_alpha = np.linspace(6e-4, 1e-3, cfg.total_layers)
cfg.sparsity=None
cfg.model_batch_size=4
cfg.lr=1e-3
cfg.kl=False
cfg.reconstruction=False
# cfg.dataset_name="NeelNanda/pile-10k"
cfg.dataset_name="Elriggs/openwebtext-100k"
# cfg.dataset_name="Skylion007/openwebtext"
cfg.device="cuda:0"
cfg.ratio = 4
# cfg.device="cpu"
tensor_names = [cfg.tensor_name.format(layer=layer) for layer in cfg.layers]
# Load in the model
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(cfg.device)
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
model
# Download the dataset
# TODO iteratively grab dataset?
cfg.max_length = 256
cfg.model_batch_size = 4
token_loader = setup_token_data(cfg, tokenizer, model)
num_tokens = cfg.max_length*cfg.model_batch_size*len(token_loader)
print(f"Number of tokens: {num_tokens}")
# Run 1 datapoint on model to get the activation size (cause don't want to deal w/ different naming schemes in config files)
from baukit import Trace, TraceDict

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
from torch import nn
from torchtyping import TensorType


class TiedSAE(nn.Module):
    def __init__(self, activation_size, n_dict_components):
        super().__init__()
        self.encoder = nn.Parameter(torch.empty((n_dict_components, activation_size)))
        nn.init.xavier_uniform_(self.encoder)
        self.encoder_bias = nn.Parameter(torch.zeros((n_dict_components,)))

    def get_learned_dict(self):
        norms = torch.norm(self.encoder, 2, dim=-1)
        return self.encoder / torch.clamp(norms, 1e-8)[:, None]

    def encode(self, batch):
        c = torch.einsum("nd,bd->bn", self.encoder, batch)
        c = c + self.encoder_bias
        c = torch.clamp(c, min=0.0)
        return c

    def decode(self, code: TensorType["_batch_size", "_n_dict_components"]) -> TensorType["_batch_size", "_activation_size"]:
        learned_dict = self.get_learned_dict()
        x_hat = torch.einsum("nd,bn->bd", learned_dict, code)
        return x_hat

    def forward(self, batch: TensorType["_batch_size", "_activation_size"]) -> TensorType["_batch_size", "_activation_size"]:
        c = self.encode(batch)
        x_hat = self.decode(c)
        return x_hat, c

    def n_dict_components(self):
        return self.get_learned_dict().shape[0]

n_dict_components = activation_size*cfg.ratio
all_autoencoders = [TiedSAE(activation_size, n_dict_components).to(cfg.device) for _ in range(len(tensor_names))]
optimizers = [torch.optim.Adam(autoencoder.parameters(), lr=cfg.lr) for autoencoder in all_autoencoders]
# Wandb setup
secrets = json.load(open("secrets.json"))
wandb.login(key=secrets["wandb_key"])
start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
wandb_run_name = f"{cfg.model_name}_{start_time[4:]}_{cfg.sparsity}"  # trim year
print(f"wandb_run_name: {wandb_run_name}")
wandb.init(project="sparse coding", config=dict(cfg), name=wandb_run_name, entity="sparse_coding")
import numpy as np
# Make directory trained_models if it doesn't exist
import os
if not os.path.exists("trained_models"):
    os.makedirs("trained_models")
model_save_name = cfg.model_name.split("/")[-1]

num_batch = len(token_loader)
log_space = np.logspace(0, np.log10(num_batch), 11)  # 11 to get 10 intervals
save_batches = [int(x) for x in log_space[1:]]  # Skip the first (0th) interval

dead_features = [torch.zeros(n_dict_components) for _ in range(len(tensor_names))]
last_encoders = [autoencoder.encoder.clone().detach() for autoencoder in all_autoencoders]
# max_num_tokens = 100000000
# Freeze model parameters 
model.eval()
model.requires_grad_(False)
for i, batch in enumerate(token_loader):
    tokens = batch["input_ids"].to(cfg.device)
    with torch.no_grad(): # As long as not doing KL divergence, don't need gradients for model
        with TraceDict(model, tensor_names) as ret:
            _ = model(tokens)
    if(i > 1400):
        print("reached text limit")
        break

    for auto_ind in range(len(tensor_names)):
        # Index into correct autoencoder, optimizer, and tensor_name
        autoencoder = all_autoencoders[auto_ind]
        optimizer = optimizers[auto_ind]
        tensor_name = tensor_names[auto_ind]
        dead_feature = dead_features[auto_ind]
        last_encoder = last_encoders[auto_ind]
        l1_alpha = cfg.l1_alpha[auto_ind]

        # Get intermediate layer activations
        representation = ret[tensor_name].output
        if(isinstance(representation, tuple)):
            representation = representation[0]
        layer_activations = rearrange(representation, "b seq d_model -> (b seq) d_model")
        
        # Run through autoencoder
        c = autoencoder.encode(layer_activations)
        x_hat = autoencoder.decode(c)

        # Calculate loss
        reconstruction_loss = (x_hat - layer_activations).pow(2).mean()
        l1_loss = torch.norm(c, 1, dim=-1).mean()
        total_loss = reconstruction_loss + l1_alpha*l1_loss

        # Update dead features
        dead_feature += c.sum(dim=0).cpu()
        
        # Log
        if (i % 200 == 0): # Check here so first check is model w/o change
            # self_similarity = torch.cosine_similarity(c, last_encoder, dim=-1).mean().cpu().item()
            # Above is wrong, should be similarity between encoder and last encoder
            self_similarity = torch.cosine_similarity(autoencoder.encoder, last_encoder, dim=-1).mean().cpu().item()
            last_encoder = autoencoder.encoder.clone().detach()
            last_encoders[auto_ind] = last_encoder
            num_tokens_so_far = i*cfg.max_length*cfg.model_batch_size
            with torch.no_grad():
                sparsity = (c != 0).float().mean(dim=0).sum().cpu().item()
                # Count number of dead_features are zero
                num_dead_features = (dead_feature == 0).sum().item()
            print(f"Layer {auto_ind} | Sparsity: {sparsity:.1f} | Dead Features: {num_dead_features} | Total Loss: {total_loss:.2f} | Reconstruction Loss: {reconstruction_loss:.2f} | L1 Loss: {l1_alpha*l1_loss:.2f} | l1_alpha: {l1_alpha:.2e} | Tokens: {num_tokens_so_far} | Self Similarity: {self_similarity:.2f}")
            wandb.log({f"Layer {auto_ind}": {
                'Sparsity': sparsity,
                'Dead Features': num_dead_features,
                'Total Loss': total_loss.item(),
                'Reconstruction Loss': reconstruction_loss.item(),
                'L1 Loss': (l1_alpha*l1_loss).item(),
                'l1_alpha': l1_alpha,
                'Tokens': num_tokens_so_far,
                'Self Similarity': self_similarity,
                'step': i}
            })
            # wandb.log({f"Layer_{auto_ind}": {
            #     f"Sparsity": sparsity,
            #     f"Dead Features": num_dead_features,
            #     f"Total Loss": total_loss.item(),
            #     f"Reconstruction Loss": reconstruction_loss.item(),
            #     f"L1 Loss": (cfg.l1_alpha * l1_loss).item(),
            #     f"l1_alpha": cfg.l1_alpha,
            #     f"Tokens": num_tokens_so_far,
            #     f"Self Similarity": self_similarity
            # }, step=i})


            # wandb.log({f"Layer_{auto_ind}/sparsity": sparsity_value, "step": step})

            dead_feature = torch.zeros(autoencoder.encoder.shape[0])
        # if i in save_batches:
        #     save_name = f"{model_save_name}_sp{cfg.sparsity}_r{cfg.ratio}_{tensor_names[0]}_{i}"  # trim year
        #     torch.save(autoencoder, f"trained_models/{save_name}.pt")
        #     print(f"Saved model to trained_models/{save_name}")
            

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # # Running sparsity check
    # if(num_tokens_so_far > 5000000):
    #     if(i % 200 == 0):
    #         with torch.no_grad():
    #             sparsity = (c != 0).float().mean(dim=0).sum().cpu().item()
    #         if sparsity > target_upper_sparsity:
    #             cfg.l1_alpha *= (1 + adjustment_factor)
    #         elif sparsity < target_lower_sparsity:
    #             cfg.l1_alpha *= (1 - adjustment_factor)
            # print(f"Sparsity: {sparsity:.1f} | l1_alpha: {cfg.l1_alpha:.2e}")
wandb.finish()
for autoencoder_ind in range(len(tensor_names)):
    autoencoder = all_autoencoders[autoencoder_ind]
    tensor_name = tensor_names[autoencoder_ind]
    save_name = f"{model_save_name}_sp{cfg.sparsity}_r{cfg.ratio}_{tensor_name}"  # trim year

    # Save model
    torch.save(autoencoder, f"trained_models/{save_name}.pt")