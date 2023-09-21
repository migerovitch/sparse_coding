# Create 
import torch
from transformer_lens import HookedTransformer
import numpy as np
import argparse
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from functools import partial

from collections import defaultdict
import matplotlib.pyplot as plt
from utils import dotdict
from einops import rearrange
import os

def main():
    # make an argument parser directly below
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--feature_list", type=list, default=None)
    parser.add_argument("--first_n_features", type=int, default=None)
    parser.add_argument("--autoencoder_path", type=str, default="/mnt/ssd-cluster/longrun2408/tied_residual_l2_r6/_31/learned_dicts.pt")
    parser.add_argument("--autoencoder_index", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="features/")
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--setting", type=str, default="residual")
    parser.add_argument("--max_seq_length", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    cfg = dotdict(vars(args))

    if(cfg.first_n_features is None):
        assert (cfg.feature_list is not None), "Must specify either feature_list or first_n_features"

    all_autoencoders = torch.load(cfg.autoencoder_path)
    autoencoder, hyperparams = all_autoencoders[cfg.autoencoder_index]
    autoencoder.to_device(cfg.device)
    print(f"Loaded autoencoder w/ {hyperparams} on {cfg.device}")

    model = HookedTransformer.from_pretrained_no_processing(cfg.model_name, device = cfg.device)

    def download_dataset(dataset_name, tokenizer, max_length=256, num_datapoints=None):
        if(num_datapoints):
            split_text = f"train[:{num_datapoints}]"
        else:
            split_text = "train"
        dataset = load_dataset(dataset_name, split=split_text).map(
            lambda x: tokenizer(x['text']),
            batched=True,
        ).filter(
            lambda x: len(x['input_ids']) > max_length
        ).map(
            lambda x: {'input_ids': x['input_ids'][:max_length]}
        )
        return dataset

    print(f"Downloading {cfg.dataset_name}")
    dataset = download_dataset(cfg.dataset_name, tokenizer= model.tokenizer, max_length=cfg.max_seq_length, num_datapoints=None) # num_datapoints grabs all of them if None

    d_model = model.cfg.d_model
    assert (d_model == autoencoder.encoder.shape[-1]), f"Model and autoencoder must have same hidden size. Model: {d_model}, Autoencoder: {autoencoder.encoder.shape[-1]}"

    # Now we can use the model to get the activations

    def get_dictionary_activations(model, dataset, cache_name, autoencoder, batch_size=32):
        num_features, d_model = autoencoder.encoder.shape
        datapoints = dataset.num_rows
        dictionary_activations = torch.zeros((datapoints*cfg.max_seq_length, num_features))
        token_list = torch.zeros((datapoints*cfg.max_seq_length), dtype=torch.int64)
        with torch.no_grad(), dataset.formatted_as("pt"):
            dl = DataLoader(dataset["input_ids"], batch_size=batch_size)
            for i, batch in enumerate(tqdm(dl)):
                token_list[i*batch_size*cfg.max_seq_length:(i+1)*batch_size*cfg.max_seq_length] = rearrange(batch, "b s -> (b s)")
                _, cache = model.run_with_cache(batch.to(device))
                batched_neuron_activations = rearrange(cache[cache_name], "b s n -> (b s) n" )
                batched_dictionary_activations = autoencoder.encode(batched_neuron_activations)
                dictionary_activations[i*batch_size*cfg.max_seq_length:(i+1)*batch_size*cfg.max_seq_length,:] = batched_dictionary_activations.cpu()
        return dictionary_activations, token_list

    print("Getting dictionary activations")
    dictionary_activations, tokens_for_each_datapoint = get_dictionary_activations(model, dataset, cfg.cache_name, autoencoder, batch_size=32)


    def ablate_feature_direction(model, dataset, cache_name, autoencoder, feature, batch_size=32):
        def less_than_rank_1_ablate(value, hook):
            # Only ablate the feature direction up to the negative bias
            # ie Only subtract when it activates above that negative bias.

            # Rearrange to fit autoencoder
            int_val = rearrange(value, 'b s h -> (b s) h')
            # Run through the autoencoder
            act = autoencoder.encode(int_val)
            dictionary_for_this_autoencoder = autoencoder.get_learned_dict()
            feature_direction = torch.outer(act[:, feature].squeeze(), dictionary_for_this_autoencoder[feature].squeeze())
            batch, seq_len, hidden_size = value.shape
            feature_direction = rearrange(feature_direction, '(b s) h -> b s h', b=batch, s=seq_len)
            value -= feature_direction
            return value

        datapoints = dataset.num_rows
        logit_diffs = torch.zeros((datapoints*cfg.max_seq_length))
        with torch.no_grad(), dataset.formatted_as("pt"):
            dl = DataLoader(dataset["input_ids"], batch_size=batch_size)
            for i, batch in enumerate(tqdm(dl)):
                original_logits = model(batch.to(device)).log_softmax(dim=-1)
                ablated_logits = model.run_with_hooks(batch.to(device), fwd_hooks=[(cache_name, less_than_rank_1_ablate)]).log_softmax(dim=-1)
                diff_logits = ablated_logits  - original_logits# ablated > original -> negative diff
                gather_tokens = rearrange(batch[:,1:].to(device), "b s -> b s 1")
                gathered = diff_logits[:, :-1].gather(-1,gather_tokens)
                # append all 0's to the beggining of gathered
                gathered = torch.cat([torch.zeros((gathered.shape[0],1,1)).to(device), gathered], dim=1)
                diff = rearrange(gathered, "b s n -> (b s n)")
                # Add one to the first position of logit diff, so we're always skipping over the first token (since it's not predicted)
                logit_diffs[i*batch_size*cfg.max_seq_length:(i+1)*batch_size*cfg.max_seq_length] = diff.cpu()
        return logit_diffs
    feature = 1
    logit_diffs = ablate_feature_direction(model, dataset, cfg.cache_name, autoencoder, feature = feature, batch_size=32)

    from IPython.display import display, HTML
    import imgkit

    def make_colorbar(min_value, max_value, white = 245, red_blue_ness = 250, positive_threshold = 0.01, negative_threshold = 0.01):
        # Add color bar
        colorbar = ""
        num_colors = 4
        if(min_value < -negative_threshold):
            for i in range(num_colors, 0, -1):
                ratio = i / (num_colors)
                value = round((min_value*ratio),1)
                text_color = "255,255,255" if ratio > 0.5 else "0,0,0"
                colorbar += f'<span style="background-color:rgba(255, {int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},1); color:rgb({text_color})">&nbsp{value}&nbsp</span>'
        # Do zero
        colorbar += f'<span style="background-color:rgba({white},{white},{white},1);color:rgb(0,0,0)">&nbsp0.0&nbsp</span>'
        # Do positive
        if(max_value > positive_threshold):
            for i in range(1, num_colors+1):
                ratio = i / (num_colors)
                value = round((max_value*ratio),1)
                text_color = "255,255,255" if ratio > 0.5 else "0,0,0"
                colorbar += f'<span style="background-color:rgba({int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},255,1);color:rgb({text_color})">&nbsp{value}&nbsp</span>'
        return colorbar

    def value_to_color(activation, max_value, min_value, white = 245, red_blue_ness = 250, positive_threshold = 0.01, negative_threshold = 0.01):
        if activation > positive_threshold:
            ratio = activation/max_value
            text_color = "0,0,0" if ratio <= 0.5 else "255,255,255"  
            background_color = f'rgba({int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},255,1)'
        elif activation < -negative_threshold:
            ratio = activation/min_value
            text_color = "0,0,0" if ratio <= 0.5 else "255,255,255"  
            background_color = f'rgba(255, {int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},1)'
        else:
            text_color = "0,0,0"
            background_color = f'rgba({white},{white},{white},1)'
        return text_color, background_color

    def convert_token_array_to_list(array):
        if isinstance(array, torch.Tensor):
            if array.dim() == 1:
                array = [array.tolist()]
            elif array.dim()==2:
                array = array.tolist()
            else: 
                raise NotImplementedError("tokens must be 1 or 2 dimensional")
        elif isinstance(array, list):
            # ensure it's a list of lists
            if isinstance(array[0], int):
                array = [array]
        return array

    def tokens_and_activations_to_html(toks, activations, tokenizer, logit_diffs=None):
        toks = convert_token_array_to_list(toks)
        activations = convert_token_array_to_list(activations)
        # toks = [[tokenizer.decode(t).replace('Ġ', '&nbsp').replace('\n', '↵') for t in tok] for tok in toks]
        toks = [[tokenizer.decode(t).replace('Ġ', '&nbsp').replace('\n', '\\n') for t in tok] for tok in toks]
        highlighted_text = []
        max_value = max([max(activ) for activ in activations])
        min_value = min([min(activ) for activ in activations])
        if(logit_diffs):
            logit_max_value = max([max(activ) for activ in logit_diffs])
            logit_min_value = min([min(activ) for activ in logit_diffs])

        # Add color bar
        highlighted_text.append("Token Activations: " + make_colorbar(min_value, max_value))
        if(logit_diffs):
            highlighted_text.append('<br><br>')
            highlighted_text.append("Logit Diff: " + make_colorbar(logit_min_value, logit_max_value))
            
        highlighted_text.append('<br><br>')
        for seq_ind, (act, tok) in enumerate(zip(activations, toks)):
            for act_ind, (a, t) in enumerate(zip(act, tok)):
                if(logit_diffs):
                    highlighted_text.append('<div style="display: inline-block;">')
                text_color, background_color = value_to_color(a, max_value, min_value)
                highlighted_text.append(f'<span style="background-color:{background_color};color:rgb({text_color})">{t.replace(" ", "&nbsp")}</span>')
                if(logit_diffs):
                    logit_diffs_act = logit_diffs[seq_ind][act_ind]
                    _, logit_background_color = value_to_color(logit_diffs_act, logit_max_value, logit_min_value)
                    highlighted_text.append(f'<div style="display: block; height: 10px; background-color:{logit_background_color}; text-align: center;"></div></div>')
            highlighted_text.append('<br><br>')
        highlighted_text = ''.join(highlighted_text)
        return highlighted_text

    def display_tokens(tokens, activations, tokenizer, logit_diffs=None):
        return display(HTML(tokens_and_activations_to_html(tokens, activations, tokenizer, logit_diffs)))

    def save_token_display(tokens, activations, tokenizer, path, logit_diffs=None):
        html = tokens_and_activations_to_html(tokens, activations, tokenizer, logit_diffs)
        imgkit.from_string(html, path)
        # print(f"Saved to {path}")
        return

    def get_feature_indices(feature_index, dictionary_activations, tokenizer, token_amount, dataset, k=10, setting="max"):
        best_feature_activations = dictionary_activations[:, feature_index]
        # Sort the features by activation, get the indices
        if setting=="max":
            found_indices = torch.argsort(best_feature_activations, descending=True)[:k]
        elif setting=="uniform":
            # min_value = torch.min(best_feature_activations)
            min_value = torch.min(best_feature_activations)
            max_value = torch.max(best_feature_activations)

            # Define the number of bins
            num_bins = k

            # Calculate the bin boundaries as linear interpolation between min and max
            bin_boundaries = torch.linspace(min_value, max_value, num_bins + 1)

            # Assign each activation to its respective bin
            bins = torch.bucketize(best_feature_activations, bin_boundaries)

            # Initialize a list to store the sampled indices
            sampled_indices = []

            # Sample from each bin
            for bin_idx in torch.unique(bins):
                if(bin_idx==0): # Skip the first one. This is below the median
                    continue
                # Get the indices corresponding to the current bin
                bin_indices = torch.nonzero(bins == bin_idx, as_tuple=False).squeeze(dim=1)
                
                # Randomly sample from the current bin
                sampled_indices.extend(np.random.choice(bin_indices, size=1, replace=False))

            # Convert the sampled indices to a PyTorch tensor & reverse order
            found_indices = torch.tensor(sampled_indices).long().flip(dims=[0])
        else: # random
            # get nonzero indices
            nonzero_indices = torch.nonzero(best_feature_activations)[:, 0]
            # shuffle
            shuffled_indices = nonzero_indices[torch.randperm(nonzero_indices.shape[0])]
            found_indices = shuffled_indices[:k]
        return found_indices
    def get_feature_datapoints(found_indices, best_feature_activations, tokenizer, token_amount, dataset):
        num_datapoints = dataset.num_rows
        datapoint_indices =[np.unravel_index(i, (num_datapoints, token_amount)) for i in found_indices]
        all_activations = best_feature_activations.reshape(num_datapoints, token_amount).tolist()
        full_activations = []
        partial_activations = []
        text_list = []
        full_text = []
        token_list = []
        full_token_list = []
        for i, (md, s_ind) in enumerate(datapoint_indices):
            md = int(md)
            s_ind = int(s_ind)
            full_tok = torch.tensor(dataset[md]["input_ids"])
            full_text.append(tokenizer.decode(full_tok))
            tok = dataset[md]["input_ids"][:s_ind+1]
            full_activations.append(all_activations[md])
            partial_activations.append(all_activations[md][:s_ind+1])
            text = tokenizer.decode(tok)
            text_list.append(text)
            token_list.append(tok)
            full_token_list.append(full_tok)
        return text_list, full_text, token_list, full_token_list, partial_activations, full_activations

    uniform_indices = get_feature_indices(feature, dictionary_activations, model.tokenizer, cfg.max_seq_length, dataset, k=10, setting="uniform")
    text_list, full_text, token_list, full_token_list, partial_activations, full_activations = get_feature_datapoints(uniform_indices, dictionary_activations[:, feature], model.tokenizer, cfg.max_seq_length, dataset)
    _, _, _, full_token_list_ablated, _, full_activations_ablated = get_feature_datapoints(uniform_indices, logit_diffs, model.tokenizer, cfg.max_seq_length, dataset)
    # display_tokens(full_token_list_ablated, full_activations, model.tokenizer, logit_diffs = full_activations_ablated)
    save_token_display(token_list, partial_activations, model.tokenizer, path =f"{cfg.save_path}uniform_{feature}.png", logit_diffs = full_activations_ablated)

    def get_token_statistics(feature, feature_activation, dataset, cfg.max_seq_length, save_location="", num_unique_tokens=10, setting="input", negative_threshold=-0.01):
        if(setting=="input"):
            nonzero_indices = feature_activation.nonzero()[:, 0]  # Get the nonzero indices
        else:
            nonzero_indices = (feature_activation < negative_threshold).nonzero()[:, 0]
        nonzero_values = feature_activation[nonzero_indices].abs()  # Get the nonzero values

        # Unravel the indices to get the token IDs
        datapoint_indices = [np.unravel_index(i, (dataset.num_rows, cfg.max_seq_length)) for i in nonzero_indices]
        all_tokens = [dataset[int(md)]["input_ids"][int(s_ind)] for md, s_ind in datapoint_indices]

        # Find the max value for each unique token
        token_value_dict = defaultdict(int)
        for token, value in zip(all_tokens, nonzero_values):
            token_value_dict[token] = max(token_value_dict[token], value)
        # if(setting=="input"):
        sorted_tokens = sorted(token_value_dict.keys(), key=lambda x: -token_value_dict[x])
        # else:
        #     sorted_tokens = sorted(token_value_dict.keys(), key=lambda x: token_value_dict[x])
        # Take the top 10 (or fewer if there aren't 10)
        max_tokens = sorted_tokens[:min(num_unique_tokens, len(sorted_tokens))]
        total_sums = nonzero_values.abs().sum()
        max_token_sums = []
        token_activations = []
        for max_token in max_tokens:
            # Find ind of max token
            max_token_indices = tokens_for_each_datapoint[nonzero_indices] == max_token
            # Grab the values for those indices
            max_token_values = nonzero_values[max_token_indices]
            max_token_sum = max_token_values.abs().sum()
            max_token_sums.append(max_token_sum)
            token_activations.append(max_token_values)


        if(setting=="input"):
            title_text = "Input Token Activations"
            save_name = "input"
            y_label = "Feature Activation"
        else:
            title_text = "Output Logit Difference"
            save_name = "logit_diff"
            y_label = "Logit Difference"

        # Plot a boxplot for each tensor in the list
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(f'{title_text}: feature={feature}')
        max_text = [model.tokenizer.decode([t]).replace("\n", "\\n").replace(" ", "_") for t in max_tokens]
        # Set x-axis label
        ax.set_xlabel('Token')
        #rotate x labels
        plt.xticks(rotation=30)
        # Set y-axis label
        ax.set_ylabel(y_label)
        ax.boxplot(token_activations[::-1], labels=max_text[::-1])
        #Save it
        plt.savefig(f'{save_location}feature_{feature}_{save_name}_boxplot.png')

        #Bar graph of the percentage of total activations
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(f'Weighted Percentage of {title_text}: feature={feature}')
        max_text = [model.tokenizer.decode([t]).replace("\n", "\\n").replace(" ", "_") for t in max_tokens]
        # Set x-axis label
        ax.set_xlabel('Token')
        plt.xticks(rotation=30)

        # Set y-axis label
        ax.set_ylabel(f'Weighted Percentage of Total {y_label}')
        ax.bar(max_text[::-1], [t/total_sums*100 for t in max_token_sums[::-1]])
        plt.savefig(f'{save_location}feature_{feature}_{save_name}_bar.png')
    # get_token_statistics(feature, dictionary_activations[:, feature], dataset, cfg.max_seq_length, save_location = "features/", num_unique_tokens=10)
    get_token_statistics(feature, logit_diffs, dataset, cfg.max_seq_length, save_location = "features/", setting="output", num_unique_tokens=10)