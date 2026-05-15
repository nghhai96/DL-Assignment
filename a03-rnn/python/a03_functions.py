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
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from lightning import LightningModule, LightningDataModule
import torchtext
from torchtext.data import get_tokenizer
import string
if hasattr(torchtext, "disable_torchtext_deprecation_warning"):
    torchtext.disable_torchtext_deprecation_warning()


# %% [markdown]
# NOTE: If importing `torchtext` fails, you may have incompatible versions between
#  `torch` and `torchtext`. In this case, you can try to install compatible versions by
#  running in your (UNIX) terminal:
#
#  pip install torch==2.3 torchtext==0.18
#
# or (in a notebook cell):
#
# !pip install torch==2.3 torchtext==0.18


# %%
# you may (and should) use these values whenever needed in this file
from a03_helper import MAX_SEQ_LEN, SEED, BATCH_SIZE


# %% [markdown]
# ## Task 1: Datasets


# %%
class ReviewsDataset(Dataset):
    def __init__(
        self,
        reviews_file="data/reviews_small.txt",
        labels_file="data/labels_small.txt",
        use_vocab=False,
    ):
        """
        A dataset of movie reviews and their labels.

        Args:
            reviews_file: the reviews file
            labels_file:  the labels file
            use_vocab: if True, yield reviews in a numerical representation
        """
        # Load data from filesystem
        with open(reviews_file) as f:
            raw_reviews = f.readlines()

        with open(labels_file) as f:
            raw_labels = f.readlines()

        # Preprocessing and store (in memory)
        self._reviews = self._preprocess_reviews(raw_reviews)
        self._labels = self._preprocess_labels(raw_labels)

        # Build vocabulary
        self.vocab = None
        if use_vocab:
            from torchtext.vocab import build_vocab_from_iterator

            self.vocab = build_vocab_from_iterator(
                self._reviews,
                specials=["<pad>"],  # will get token id 0
            )

    def __len__(self):
        """Returns the length of the dataset."""
        # YOUR CODE HERE
        return len(self._reviews)

    def __getitem__(self, idx):
        """
        Returns a tuple of a preprocessed review and the corresponding label. If the
        vocabulary is enabled, returns a numerical representation of the review.

        Args:
            idx: a single index
        Returns: a (review, label) tuple
        """
        # YOUR CODE HERE
        if self.vocab is None:
            return (self._reviews[idx], self._labels[idx])
        return ([self.vocab[token] for token in self._reviews[idx]], self._labels[idx])

    def _preprocess_reviews(self, raw_reviews):
        """
        Applies two kinds of preprocessing:

        (i) Apply the "basic_english" tokenizer from the torchtext library to
        transform every review into a list of normalized tokens (cf.
        https://pytorch.org/text/stable/data_utils.html#get-tokenizer).

        (ii) Remove punctuation (cf.
        https://docs.python.org/3/library/string.html#string.punctuation).

        Returns: list of tokenized reviews
        """
        # YOUR CODE HERE
        tokens = []
        tokenizer = get_tokenizer("basic_english")
        for line in raw_reviews:
            line_tokens = tokenizer(line)
            clean_line_tokens  = [
                token for token in line_tokens
                if token not in string.punctuation]
            tokens.append(clean_line_tokens)
        return tokens

    def _preprocess_labels(self, raw_labels):
        """
        Transform raw labels into integers, where 1="positive" and 0 otherwise.

        Returns: list of labels
        """
        # YOUR CODE HERE
        # Hint: You can remove leading and trailing whitespace from the raw
        # labels using the strip() method.
        labels=[1 if line.strip() == 'positive' else 0 for line in raw_labels]
        return labels


# %% [markdown]
# ## Task 2
#
# ### Task 2b


# %%
def review_collate_fn(raw_batch):
    """Prepare batches of reviews from a review dataset.

    Args:
        raw_batch: collection of (review, label)-tuples from a ReviewDataset

    Returns: a tuple (review x token id tensor, label tensor) of sizes
    batch_size*MAX_SEQ_LEN and batch_size, respectively.

    """
    # YOUR CODE HERE
    max_batch_length = max([len(tokens) for tokens, label in raw_batch])
    padded_batch = []
    labels = []
    for tokens, label in raw_batch:
        tokens = torch.as_tensor(tokens)
        tokens = tokens[:MAX_SEQ_LEN]
        pad_length = MAX_SEQ_LEN - len(tokens)
        padded_tokens = F.pad(tokens, (0, pad_length), value=0)
        padded_batch.append(padded_tokens)
        labels.append(label)
    return (torch.stack(padded_batch), torch.tensor(labels))

# %% [markdown]
# ## Task 3: Recurrent Neural Networks


# %%
class SimpleLSTM(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, num_layers=1, cell_dropout=0.0
    ):
        """
        Initializes the model by setting up the layers.

        Args:
            vocab_size: number of unique words in the reviews
            embedding_dim: size of the embeddings
            hidden_dim: dimension of the LSTM output
            num_layers: number of LSTM layers
            cell_dropout: dropout applied between the LSTM layers
                          (provide to LSTM constructor as dropout argument)
        """
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # YOUR CODE HERE
        # Note: Use the following attributes to store the required layers.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first= True,
            dropout=cell_dropout,
        )
        self.fc = nn.Linear(
            in_features=hidden_dim,
            out_features=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Performs a forward pass of the model on some input and hidden state.

        Parameters
        ----------
        x: batch as a (batch_size, sequence_length) tensor

        Returns
        -------
        Tuple of form (probability of positive class, last LSTM output)
        """
        # init hidden layer, which is needed for the LSTM
        hidden = self.init_hidden(len(x))

        # YOUR CODE HERE
        embedded = self.embedding(x)
        _, (final_hiddens, final_memory) = self.lstm(embedded, hidden)

        thought_vector = final_hiddens[-1] # final hidden of final layer
        probs = self.sigmoid(self.fc(thought_vector))

        return (probs, thought_vector)

    def init_hidden(self, batch_size):
        """Initialize hidden states.

        Returns a tuple of two num_layers x batch_size x hidden_dim tensors (one for
        initial cell states, one for initial hidden states) consisting of all zeros.
        """
        # YOUR CODE HERE
        device = next(self.parameters()).device
        initial_hidden_state = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim, device=device
        )
        initial_cell_state = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim, device=device
        )
        return (initial_hidden_state, initial_cell_state)

        # Note: ensure that the returned tensors are located on the same device
        # as the model's parameters.


# %% [markdown]
# ## Task 4: Lightning


# %%
class LitSimpleLSTM(LightningModule):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers=1,
        cell_dropout=0.0,
        lr=0.01,
    ) -> None:
        super().__init__()
        self.model = SimpleLSTM(
            vocab_size, embedding_dim, hidden_dim, num_layers, cell_dropout
        )
        self.lr = lr
        self.loss_fn = F.binary_cross_entropy

    def configure_optimizers(self):
        # YOUR CODE HERE
        # Return an optimizer for the parameters which require gradients (cf.
        # pre-training task).
        # Note: Use self.lr as the learning rate.
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        return torch.optim.Adam(parameters, lr=self.lr)

    def forward(self, inputs):
        # YOUR CODE HERE
        # Compute and return the model's outputs given the inputs.
        return self.model(inputs)

    def training_step(self, batch, _):
        reviews, labels = batch

        # YOUR CODE HERE
        # Forward pass: Compute the model's output, reshape it to a vector, and
        # then compute the (binary) cross-entropy.
        # Use self.loss_fn to compute the loss.
        output, _ = self(reviews)
        labels = labels.float().view(-1, 1)
        loss = self.loss_fn(output, labels)

        # YOUR CODE HERE
        # Logging: Log the loss per step and the accuracy per epoch.
        # Use a single call to self.log(...) per metric and use descriptive
        # names (such as "train_loss" and "train_acc").
        #
        # Hint: It is often helpful to put one (or both) of these
        # metrics into the progress bar.
        predictions = (output >= 0.5).float()
        accuracy = (predictions == labels).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        # YOUR CODE HERE
        # Return the loss and let lightning handle backprop automatically.
        return loss

    def validation_step(self, batch, _):
        return self._eval(batch, "val")

    def test_step(self, batch, _):
        return self._eval(batch, "test")

    def _eval(self, batch, eval_type):
        reviews, labels = batch

        # YOUR CODE HERE
        # Implement the forward pass (similar to the training step).
        output, _ = self(reviews)
        labels = labels.float().view(-1, 1)
        loss = self.loss_fn(output, labels)

        # YOUR CODE HERE
        # Logging: Log the loss and the accuracy per epoch. It is often
        # helpful to put one (or both) of these metrics into the progress bar.
        # Use `eval_type` to distinguish between validation and test metrics.
        predictions = (output >= 0.5).float()
        accuracy = (predictions == labels).float().mean()
        self.log(f"{eval_type}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{eval_type}_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss


# %%
class ReviewsDataModule(LightningDataModule):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.vocab = self.dataset.vocab
        self.train_set, self.val_set, self.test_set = None, None, None

    def setup(self, stage):
        # YOUR CODE HERE
        # Randomly split the dataset into train, validation, and test sets with
        # a ratio of 8:1:1 using the global SEED. Store them
        # in the variables defined in the constructor.
        #
        # Note: You can ignore the `stage` parameter in this task (but read
        # about its purpose in the documentation).
        #
        n = len(self.dataset)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        n_test = n - n_train - n_val

        #split
        self.train_set, self.val_set, self.test_set = random_split(
            self.dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(SEED),
        )


    def train_dataloader(self):
        # YOUR CODE HERE
        # Return a data loader for the train set with your reviews_collate_fn of task 2.
        # Make sure that the data is shuffled before every epoch.
        train_set = DataLoader(
            self.train_set,
            batch_size = BATCH_SIZE,
            shuffle = True,
            collate_fn=review_collate_fn,
        )
        return train_set

    def val_dataloader(self):
        # YOUR CODE HERE
        # Return a data loader for the validation set with  your reviews_collate_fn of task 2.
        # No need to shuffle.
        val_set = DataLoader(
            self.val_set,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=review_collate_fn,
        )
        return val_set

    def test_dataloader(self):
        # YOUR CODE HERE
        # Return a data loader for the test set with your reviews_collate_fn of task 2.
        # No need to shuffle.
        test_set = DataLoader(
            self.test_set,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=review_collate_fn,
        )
        return test_set
