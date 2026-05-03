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
# %load_ext autoreload
# %autoreload 2

# %%
import torch
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from a03_helper import (
    nextplot,
    tsne_vocab,
    tsne_thought,
    DEVICE,
    reviews_load_embeddings,
)
from a03_functions import SimpleLSTM, ReviewsDataset, ReviewsDataModule, LitSimpleLSTM
import matplotlib.pyplot as plt

# %% [markdown]
# ## Task 5: Pre-trained Embeddings & Visualization

# %% [markdown]
# ### Task 5b

# %%
# Load Glove embeddings into a plain embedding layer.
dataset = ReviewsDataset(use_vocab=True)
vocab = dataset.vocab
glove_embeddings = nn.Embedding(len(vocab), 100, device=DEVICE)
reviews_load_embeddings(glove_embeddings, vocab.get_stoi())

# %%
# Print one embedding
glove_embeddings(torch.tensor(vocab["movie"], device=DEVICE))

# %%
# Plot embeddings of first 100 words using t-SNE
nextplot()
_ = tsne_vocab(glove_embeddings, torch.arange(100), vocab)

# %%
# You can also specify colors and/or drop the item labels
nextplot()
_ = tsne_vocab(glove_embeddings, torch.arange(100), colors=[0] * 50 + [1] * 50)

# %%
# YOUR CODE HERE
# Note: you can obtain the embeddings tensor using glove_embeddings.weight.data

# %% [markdown]
# ### Task 5c

# %%
# hyperparameter settings for rest of task 5
vocab_size = len(dataset.vocab)
embedding_dim = 100
hidden_dim = 100
num_layers = 2
n_epochs = 10
cell_dropout = 0.0

# %%
model = LitSimpleLSTM(vocab_size, embedding_dim, hidden_dim, num_layers, cell_dropout)
dataset = ReviewsDataset(use_vocab=True)
dm = ReviewsDataModule(dataset)
# TODO: Your code here
# Train a plain model so that it reaches a train accuracy of >0.9.

# %%
# Plot t-SNE embeddings of the thought vectors for training data
# point color = label
dm.setup("fit")
train_loader = dm.train_dataloader()
nextplot()
_ = tsne_thought(model, train_loader, DEVICE)

# %%
# Plot t-SNE embeddings of of the thought vectors for validation data
dm.setup("fit")
val_loader = dm.val_dataloader()
nextplot()
_ = tsne_thought(model, val_loader, DEVICE)

# %% [markdown]
# ### Task 5d

# %%
# Initialize the model with *p*re-trained embeddings with *f*inetuning, then
# train.
model_pf = LitSimpleLSTM(
    vocab_size, embedding_dim, hidden_dim, num_layers, cell_dropout
)
reviews_load_embeddings(model_pf.model.embedding, vocab.get_stoi())
# TODO: Your code here

# %% [markdown]
# ### Task 5e

# %%
# Initialize the model with *p*re-trained embeddings without finetuning, then
# train.
model_p = LitSimpleLSTM(vocab_size, embedding_dim, hidden_dim, num_layers, cell_dropout)
reviews_load_embeddings(model_p.model.embedding, vocab.get_stoi())
model_p.model.embedding.weight.requires_grad = False
# TODO: Your code here
