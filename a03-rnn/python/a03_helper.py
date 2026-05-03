# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import Dataset
import os.path as osp
from lightning import seed_everything

# ===
# Begin: Macros
# ===

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN = 200
BATCH_SIZE = 32
SEED = 0
seed_everything(SEED)

# ===
# End: Macros
# ===

# setup plotting
from IPython import get_ipython
import psutil

inTerminal = not "IPKernelApp" in get_ipython().config
inJupyterNb = any(
    filter(
        lambda x: x.endswith("jupyter-notebook"), psutil.Process().parent().cmdline()
    )
)
get_ipython().run_line_magic(
    "matplotlib", "" if inTerminal else "notebook" if inJupyterNb else "widget"
)


def nextplot():
    if inTerminal:
        plt.clf()  # this clears the current plot
    else:
        plt.figure()  # this creates a new plot


def tsne_vocab(embeddings: nn.Module, tokens=None, vocab=None, colors=None):
    """
    Visualize embeddings from an embedding module using t-SNE.

    Args:
        embeddings: embedding layer of the model
        tokens: limit to the provided embeddings (a tensor of indexes)
        vocab: a vocabulary to label dots in scatter plot with tokens
    """
    embeddings = embeddings.weight.data.cpu().numpy()
    if tokens is not None:
        embeddings = embeddings[tokens]
    else:
        tokens = torch.arange(len(embeddings))

    if len(embeddings) > 100 and vocab is not None:
        print(
            "Using token labels with more than 100 tokens might produce an "
            "overcrowded plot. Consider using this function without the "
            "vocabulary."
        )

    # Compute 2d representation of first no_token embeddings
    tsne = TSNE(n_components=2)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot embeddings
    # plt.figure(figsize=(10, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors)

    # Add token to points
    if vocab:
        for i in range(len(embeddings)):
            plt.annotate(
                vocab.get_itos()[tokens[i]], (embeddings_2d[i, 0], embeddings_2d[i, 1])
            )

    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title(
        f"t-SNE visualization of vocabulary embeddings ({len(embeddings)} tokens)"
    )
    plt.show()

    return embeddings_2d


@torch.no_grad()
def tsne_thought(model: nn.Module, dataloader, device):
    """
    Visualize hidden representations of the provided reviews using t-SNE.

    Args:
        model: model that produces hidden states in forward pass (second output)
        dataset: review dataset
        device: accelerator (cuda or cpu)
    """
    model.to(device)
    model.eval()

    # Compute embeddings
    review_embeddings = []
    review_labels = []
    for reviews, labels in dataloader:
        _, rnn_out = model(reviews.to(device))
        review_embeddings.extend(rnn_out.cpu().tolist())
        review_labels.extend(labels)

    # Convert to numpy
    review_embeddings = np.array(review_embeddings)
    review_labels = np.array(review_labels)

    # Compute 2d representation of first no_token embeddings
    tsne = TSNE(n_components=2)
    embeddings_2d = tsne.fit_transform(review_embeddings)

    # Plot embeddings
    # plt.figure(figsize=(10, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=review_labels)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title(f"t-SNE visualization of thought vectors")
    plt.show()

    return embeddings_2d


@torch.no_grad()
def reviews_load_embeddings(
    embedding_layer: nn.Embedding,
    token_dict: dict,
    pretrained_embeddings_file=osp.join("data", "word-embeddings.txt"),
):
    """
    Load pretrained embeddings into an embedding layer.

    Updates the weights of the embedding layer with the embeddings given in the
    provided word embeddings file.

    Args:
        embedding_layer: torch.nn.Embedding used in the model
        token_dict: dictionary mapping each token to its unique identifier
        pretrained_embeddings_file: path to the file containing pretrained embeddings

    """
    print("Initializing embedding layer with pretrained word embeddings...")
    words_initialized = 0
    with open(pretrained_embeddings_file, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            encoded_word = token_dict.get(word)
            if encoded_word is not None:
                words_initialized += 1
                float_list = [float(entry) for entry in values[1:]]
                embedding_layer.weight[encoded_word, :] = torch.tensor(
                    float_list, dtype=torch.float
                )
    print(
        "Initialized {}/{} word embeddings".format(
            words_initialized, embedding_layer.num_embeddings
        )
    )
