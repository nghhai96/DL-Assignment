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
plt.gcf().savefig("task5b_glove_tsne_first100_labeled.png", dpi=300, bbox_inches="tight")

# %%
# You can also specify colors and/or drop the item labels
nextplot()
_ = tsne_vocab(glove_embeddings, torch.arange(100), colors=[0] * 50 + [1] * 50)
plt.gcf().savefig("task5b_glove_tsne_first100_colored.png", dpi=300, bbox_inches="tight")

# %%
# Explore the same first 100 embeddings with cosine similarities.
import torch.nn.functional as F

token_ids = torch.arange(100, device=DEVICE)
embeddings = glove_embeddings.weight.data[token_ids]
embeddings = F.normalize(embeddings, dim=1)
similarities = embeddings @ embeddings.T
token_labels = [vocab.get_itos()[idx] for idx in token_ids.cpu().tolist()]

plt.figure(figsize=(14, 12))
plt.imshow(similarities.cpu(), cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Cosine similarity")
plt.xticks(range(len(token_labels)), token_labels, rotation=90, fontsize=6)
plt.yticks(range(len(token_labels)), token_labels, fontsize=6)
plt.title("Cosine similarity matrix for first 100 vocabulary embeddings")
plt.tight_layout()
plt.savefig("task5b_glove_cosine_first100.png", dpi=300, bbox_inches="tight")
plt.show()

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

# Train a plain model so that it reaches a train accuracy of >0.9.
logger = TensorBoardLogger("logs", name="task5c_plain_lstm")
trainer = Trainer(
    max_epochs=n_epochs,
    gradient_clip_val=3,
    check_val_every_n_epoch=1,
    logger=logger,
)

trainer.fit(model, datamodule=dm)
print("Task 5c train/validation metrics:")
print(trainer.callback_metrics)
print("Task 5c test metrics:")
print(trainer.test(model, datamodule=dm))

# %%
# Plot t-SNE embeddings of the thought vectors for training data
# point color = label
dm.setup("fit")
train_loader = dm.train_dataloader()
nextplot()
_ = tsne_thought(model, train_loader, DEVICE)
plt.gcf().savefig("task5c_plain_train_thought_tsne.png", dpi=300, bbox_inches="tight")

# %%
# Plot t-SNE embeddings of of the thought vectors for validation data
dm.setup("fit")
val_loader = dm.val_dataloader()
nextplot()
_ = tsne_thought(model, val_loader, DEVICE)
plt.gcf().savefig("task5c_plain_val_thought_tsne.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# ### Task 5d

# %%
# Initialize the model with *p*re-trained embeddings with *f*inetuning, then
# train.
model_pf = LitSimpleLSTM(
    vocab_size, embedding_dim, hidden_dim, num_layers, cell_dropout
)
reviews_load_embeddings(model_pf.model.embedding, vocab.get_stoi())

logger_pf = TensorBoardLogger("logs", name="task5d_glove_finetune")
trainer_pf = Trainer(
    max_epochs=n_epochs,
    gradient_clip_val=3,
    check_val_every_n_epoch=1,
    logger=logger_pf,
)

trainer_pf.fit(model_pf, datamodule=dm)
print("Task 5d train/validation metrics:")
print(trainer_pf.callback_metrics)
print("Task 5d test metrics:")
print(trainer_pf.test(model_pf, datamodule=dm))

# Plot t-SNE embeddings of the thought vectors for training data.
dm.setup("fit")
train_loader = dm.train_dataloader()
nextplot()
_ = tsne_thought(model_pf, train_loader, DEVICE)
plt.gcf().savefig("task5d_glove_finetune_train_thought_tsne.png", dpi=300, bbox_inches="tight")

# Plot t-SNE embeddings of the thought vectors for validation data.
val_loader = dm.val_dataloader()
nextplot()
_ = tsne_thought(model_pf, val_loader, DEVICE)
plt.gcf().savefig("task5d_glove_finetune_val_thought_tsne.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# ### Task 5e

# %%
# Initialize the model with *p*re-trained embeddings without finetuning, then
# train.
model_p = LitSimpleLSTM(vocab_size, embedding_dim, hidden_dim, num_layers, cell_dropout)
reviews_load_embeddings(model_p.model.embedding, vocab.get_stoi())
model_p.model.embedding.weight.requires_grad = False

logger_p = TensorBoardLogger("logs", name="task5e_glove_frozen")
trainer_p = Trainer(
    max_epochs=n_epochs,
    gradient_clip_val=3,
    check_val_every_n_epoch=1,
    logger=logger_p,
)

trainer_p.fit(model_p, datamodule=dm)
print("Task 5e train/validation metrics:")
print(trainer_p.callback_metrics)
print("Task 5e test metrics:")
print(trainer_p.test(model_p, datamodule=dm))

# Plot t-SNE embeddings of the thought vectors for training data.
dm.setup("fit")
train_loader = dm.train_dataloader()
nextplot()
_ = tsne_thought(model_p, train_loader, DEVICE)
plt.gcf().savefig("task5e_glove_frozen_train_thought_tsne.png", dpi=300, bbox_inches="tight")

# Plot t-SNE embeddings of the thought vectors for validation data.
val_loader = dm.val_dataloader()
nextplot()
_ = tsne_thought(model_p, val_loader, DEVICE)
plt.gcf().savefig("task5e_glove_frozen_val_thought_tsne.png", dpi=300, bbox_inches="tight")
