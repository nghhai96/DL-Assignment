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
from torch.utils.data import DataLoader, random_split
from a03_functions import ReviewsDataset, review_collate_fn
from a03_helper import MAX_SEQ_LEN, BATCH_SIZE, SEED

# %% [markdown]
# ## Task 2: Data Loaders

# %%
# Split dataset into training, validation, and test subsets
dataset = ReviewsDataset(use_vocab=True)
train_set, val_set, test_set = random_split(
    dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(SEED)
)

# %% [markdown]
# ### Task 2a

# %%
# Example usage of a data loader
dataloader = DataLoader(
    val_set,  # a dataset
    1,  # desired batch size
    False,  # whether to randomly shuffle the dataset
)

# %%
# Let's print the first batch
batch = next(iter(dataloader))
print(batch)

# [[tensor([11]), tensor([6]), tensor([140]), ... , tensor([8])], tensor([1])]


# %% [markdown]
# ### Task 2b

# %%
# Test your function
review_collate_fn([([1, 2, 3], 1), (torch.arange(MAX_SEQ_LEN * 2) + 1, 0)])

# Should yield:
# (tensor([[  1,   2,   3,   0,   0,  ..., 0 ],
#          [  1,   2,   3,   4,   5, ..., 200 ]]),
#  tensor([1, 0]))

# %%
# Create the data loaders (with shuffling for training data -> randomization)
train_loader = DataLoader(train_set, BATCH_SIZE, True, collate_fn=review_collate_fn)
val_loader = DataLoader(val_set, BATCH_SIZE, False, collate_fn=review_collate_fn)
test_loader = DataLoader(test_set, BATCH_SIZE, False, collate_fn=review_collate_fn)

# %%
# Let's print the first batch
batch = next(iter(val_loader))
print(batch)

# (
#   tensor([[  11,    6,  140,  ...,    0,    0,    0],
#         [  10,  123,  345,  ...,    0,    0,    0],
#         [   1,  822,  331,  ...,    0,    0,    0],
#         ...,
#         [   3,   50,  798,  ...,    0,    0,    0],
#         [ 417,   96,   35,  ...,    0,    0,    0],
#         [  25,   23,    3,  ..., 7774,  110,   73]]),
#   tensor([1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1,
#         1, 0, 0, 0, 0, 0, 1, 1])
# )
