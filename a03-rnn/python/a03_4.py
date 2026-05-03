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
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from a03_functions import (
    ReviewsDataset,
    ReviewsDataModule,
    LitSimpleLSTM,
)

from a03_helper import DEVICE


# %%
# Initialize the data module.
dataset = ReviewsDataset(use_vocab=True)
dm = ReviewsDataModule(dataset)

# Initialize the model.
vocab_size = len(dm.vocab)
embedding_dim = 128
hidden_dim = 128
num_layers = 2
n_epochs = 10
cell_dropout = 0.5
model = LitSimpleLSTM(vocab_size, embedding_dim, hidden_dim, num_layers, cell_dropout)

# %%
# Test the model's forward method.
dm.setup("fit")
train_loader = dm.train_dataloader()
reviews, labels = next(iter(train_loader))
with torch.no_grad():
    out = model(reviews)

print(out)

# Should give you something such as (conrete numbers may differ):
# (
# tensor([[0.4875], [0.4942], [0.4918], ..., device='cuda:0'),
# tensor([[-0.0833, -0.0235, 0.0280, ..., device='cuda:0')
# )
# Shape:
#
# tuple[torch.Size([32, 1]), torch.Size([32, 100])]

# %%
# Initialize the trainer.
# TODO: YOUR CODE HERE

# %%
# Train the model.
trainer.fit(model, datamodule=dm)

# Result after 10 epochs (roughly):
# val_loss=0.793, val_acc=0.726, train_acc=0.956

# %%
# Evaluate.
trainer.test(model, datamodule=dm)

# Should give you something similar to:
# [{'test_loss': 0.695..., 'test_acc': 0.737...}]
