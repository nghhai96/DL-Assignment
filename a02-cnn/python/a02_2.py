# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# make sure you installed the mnist1d package

# %load_ext autoreload
# %autoreload 2
import torch
from a02_functions import SimpleCNN, SimpleMLP, train_model
from a02_helper import (
    plot_templates,
    get_raw_data,
    count_model_params,
    shuffle_pixels,
    plot_example,
    nextplot,
)

# %% [markdown]
# # Task 2: MNIST-1D
# ### Dataset


# %%
# Those are the templates that the examples in the dataset are based on.
nextplot()
plot_templates()

# %%
data = get_raw_data()

# %%
idx = 8
x, y, t = data["x"][idx], data["y"][idx], data["t"]
nextplot()
plot_example(x, y, t)

# %% [markdown]
# ### Model
#
# Verify your model's implementation by running the following test cases.

# %%
torch.manual_seed(0)
cnn = SimpleCNN()
x_unbatched = torch.ones(40)
x_batched = x_unbatched.view(1, -1)
with torch.no_grad():  # these should give no error
    y_batched = cnn(x_batched)
    y_unbatched = cnn(x_unbatched)

x_batched, x_unbatched  # these are the inputs being used

# %% [markdown]
# Initialize your model and verify the total number of parameters computed by
# hand.

# %%
torch.manual_seed(0)
cnn = SimpleCNN()
print(cnn)
print(f"No. of parameters: {count_model_params(cnn)}")

# %% [markdown]
# ### Training
#
# Train your model.

# %%
results = train_model(data, cnn)

# %% [markdown]
# ### Simple Feedforward Neural Network
#
# Compare the results of the CNN and FNN models (after your conjecture!).

# %%
torch.manual_seed(0)
fnn = SimpleMLP()
print(f"No. of parameters: {count_model_params(fnn)}")


# %%
results = train_model(data, fnn)

# %% [markdown]
# ### Shuffled Dataset
#
# Shuffle the dataset along the spatial dimension.

# %%
shuffled_dataset = shuffle_pixels(data)

# %%
idx = 8
x, y, t = shuffled_dataset["x"][idx], shuffled_dataset["y"][idx], shuffled_dataset["t"]
nextplot()
plot_example(x, y, t)

# %%
torch.manual_seed(0)
cnn = SimpleCNN()
results = train_model(shuffled_dataset, cnn)

# %%
torch.manual_seed(0)
fnn = SimpleMLP()
results = train_model(shuffled_dataset, fnn)
