# -*- coding: utf-8 -*-
"""Image Classification on CIFAR-10

The specific task we are trying to solve in this problem is image classification. We're using a common dataset called CIFAR-10 which has 60,000 images separated into 10 classes:
* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck
"""

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)  # this should print out CUDA

# 1. Import dependencies

# Commented out IPython magic to ensure Python compatibility.
import torch
from torch import nn
import numpy as np

from typing import Tuple, Union, List, Callable
from torch.optim import SGD
import torchvision
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# %matplotlib inline

# 2. Check if using GPU

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)  # this should print out CUDA

#3. Load CIFAR data

train_dataset = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())

#4. Wrap dataset 

SAMPLE_DATA = False # set this to True if you want to speed up training when searching for hyperparameters!

batch_size = 128

if SAMPLE_DATA:
  train_dataset, _ = random_split(train_dataset, [int(0.1 * len(train_dataset)), int(0.9 * len(train_dataset))]) # get 10% of train dataset and "throw away" the other 90%

train_dataset, val_dataset = random_split(train_dataset, [int(0.9 * len(train_dataset)), int( 0.1 * len(train_dataset))])

# Create separate dataloaders for the train, test, and validation set
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
)


imgs, labels = next(iter(train_loader))
print(f"A single batch of images has shape: {imgs.size()}")
example_image, example_label = imgs[0], labels[0]
c, w, h = example_image.size()
print(f"A single RGB image has {c} channels, width {w}, and height {h}.")

# This is one way to flatten our images
batch_flat_view = imgs.view(-1, c * w * h)
print(f"Size of a batch of images flattened with view: {batch_flat_view.size()}")

# This is another equivalent way
batch_flat_flatten = imgs.flatten(1)
print(f"Size of a batch of images flattened with flatten: {batch_flat_flatten.size()}")

# The new dimension is just the product of the ones we flattened
d = example_image.flatten().size()[0]
print(c * w * h == d)

# View the image
t =  torchvision.transforms.ToPILImage()
plt.imshow(t(example_image))

# These are what the class labels in CIFAR-10 represent. For more information,
# visit https://www.cs.toronto.edu/~kriz/cifar.html
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
           "horse", "ship", "truck"]
print(f"This image is labeled as class {classes[example_label]}")

#1. Create model

def linear_model() -> nn.Module:
    """Instantiate a linear model and send it to device."""
    model =  nn.Sequential(
            nn.Flatten(),
            nn.Linear(d, 10)
         )
    return model.to(DEVICE)

#2. Method for training (using SGD)

def train(
    model: nn.Module, optimizer: SGD,
    train_loader: DataLoader, val_loader: DataLoader,
    epochs: int = 20
    )-> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Trains a model for the specified number of epochs using the loaders.

    Returns:
    Lists of training loss, training accuracy, validation loss, validation accuracy for each epoch.
    """

    loss = nn.CrossEntropyLoss()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for e in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        # Main training loop; iterate over train_loader. The loop
        # terminates when the train loader finishes iterating, which is one epoch.
        for (x_batch, labels) in train_loader:
            x_batch, labels = x_batch.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            labels_pred = model(x_batch)
            batch_loss = loss(labels_pred, labels)
            train_loss = train_loss + batch_loss.item()

            labels_pred_max = torch.argmax(labels_pred, 1)
            batch_acc = torch.sum(labels_pred_max == labels)
            train_acc = train_acc + batch_acc.item()

            batch_loss.backward()
            optimizer.step()
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc / (batch_size * len(train_loader)))

        # Validation loop; use .no_grad() context manager to save memory.
        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for (v_batch, labels) in val_loader:
                v_batch, labels = v_batch.to(DEVICE), labels.to(DEVICE)
                labels_pred = model(v_batch)
                v_batch_loss = loss(labels_pred, labels)
                val_loss = val_loss + v_batch_loss.item()

                v_pred_max = torch.argmax(labels_pred, 1)
                batch_acc = torch.sum(v_pred_max == labels)
                val_acc = val_acc + batch_acc.item()
            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(val_acc / (batch_size * len(val_loader)))

    return train_losses, train_accuracies, val_losses, val_accuracies

#3. Hyperparameter search

def parameter_search(train_loader: DataLoader,
                     val_loader: DataLoader,
                     model_fn:Callable[[], nn.Module]) -> float:
    """
    Parameter search for our linear model using SGD.

    Args:
    train_loader: the train dataloader.
    val_loader: the validation dataloader.
    model_fn: a function that, when called, returns a torch.nn.Module.

    Returns:
    The learning rate with the least validation loss.
    NOTE: you may need to modify this function to search over and return
     other parameters beyond learning rate.
    """
    num_iter = 10
    best_loss = torch.tensor(np.inf)
    best_lr = 0.0

    lrs = torch.linspace(10 ** (-6), 10 ** (-1), num_iter)

    for lr in lrs:
        print(f"trying learning rate {lr}")
        model = model_fn()
        optim = SGD(model.parameters(), lr)
        train_loss, train_acc, val_loss, val_acc = train(
            model,
            optim,
            train_loader,
            val_loader,
            epochs=20
            )

        if min(val_loss) < best_loss:
            best_loss = min(val_loss)
            best_lr = lr

    return best_lr

#4. Train and evaluate model

best_lr = parameter_search(train_loader, val_loader, linear_model)

model = linear_model()
optimizer = SGD(model.parameters(), best_lr)

# We are using 20 epochs
train_loss, train_accuracy, val_loss, val_accuracy = train(
    model, optimizer, train_loader, val_loader, 20)

#5. Plot training and accuracy for each epoch

epochs = range(1, 21)
plt.plot(epochs, train_accuracy, label="Train Accuracy")
plt.plot(epochs, val_accuracy, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Logistic Regression Accuracy for CIFAR-10 vs Epoch")
plt.show()

# Evaluate model on test data

def evaluate(
    model: nn.Module, loader: DataLoader
) -> Tuple[float, float]:
    """Computes test loss and accuracy of model on loader."""
    loss = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for (batch, labels) in loader:
            batch, labels = batch.to(DEVICE), labels.to(DEVICE)
            y_batch_pred = model(batch)
            batch_loss = loss(y_batch_pred, labels)
            test_loss = test_loss + batch_loss.item()

            pred_max = torch.argmax(y_batch_pred, 1)
            batch_acc = torch.sum(pred_max == labels)
            test_acc = test_acc + batch_acc.item()
        test_loss = test_loss / len(loader)
        test_acc = test_acc / (batch_size * len(loader))
        return test_loss, test_acc

test_loss, test_acc = evaluate(model, test_loader)
print(f"Test Accuracy: {test_acc}")

def fully_connected_model(M, input_dim=3072, output_dim=10):
    """Instantiate a fully connected model with one hidden layer and send it to device."""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, M),
        nn.ReLU(),
        nn.Linear(M, output_dim)
    )
    return model.to(DEVICE)

def conv_model(M, k, N, output_dim=10, input_size=(3, 32, 32)):
    """Instantiate a convolutional model with one conv layer followed by max-pooling and a fully connected layer, and send it to device."""
    model = nn.Sequential(
        nn.Conv2d(input_size[0], M, k, padding=k//2),
        nn.ReLU(),
        nn.MaxPool2d(N),
        nn.Flatten(),
        nn.Linear(M * ((input_size[1] // N) ** 2), output_dim)
    )
    return model.to(DEVICE)

def parameter_search_parta(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_fn:Callable[[], nn.Module]
) -> float:
    """
    Parameter search for our linear model using SGD.

    Args:
    train_loader: the train dataloader.
    val_loader: the validation dataloader.
    model_fn: a function that, when called, returns a torch.nn.Module.

    Returns:
    The learning rate with the least validation loss.
    """
    num_iter = 10
    best_3_results = []
    num_epoch = 15

    lrs = torch.linspace(10 ** (-6), 10 ** (-1), num_iter)
    Ms = [256, 512, 800, 900]

    for lr in lrs:
        for M in Ms:
          model = model_fn(M)
          optim = SGD(model.parameters(), lr, momentum=0.9)
          train_loss, train_acc, val_loss, val_acc = train(
              model,
              optim,
              train_loader,
              val_loader,
              epochs=num_epoch
              )

          curr_loss = min(val_loss)
          best_3_results.append((curr_loss, lr, M))
          best_3_results.sort(key=lambda x: x[0])
          best_3_results = best_3_results[:3]

    return [(lr, M) for _, lr, M in best_3_results]

results = parameter_search_parta(train_loader, val_loader, fully_connected_model)

num_epochs = 60
for i, (lr, M) in enumerate(results):
  model = fully_connected_model(M)
  optimizer = SGD(model.parameters(), lr, momentum=0.9)
  train_loss, train_accuracy, val_loss, val_accuracy = train(
      model, optimizer, train_loader, val_loader, num_epochs
  )
  epochs = range(1, num_epochs+1)
  plt.plot(epochs, train_accuracy, label=f"Train Accuracy, lr: {lr}, M: {M}, momentum: 0.9")
  plt.plot(epochs, val_accuracy, label=f"Validation Accuracy, lr: {lr}, M: {M}, momentum: 0.9", linestyle='dotted')

  test_loss, test_acc = evaluate(model, test_loader)
  print(f"Test Accuracy: {test_acc}, lr: {lr}, M: {M}, momentum: 0.9")

plt.axhline(y=0.5, linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

def parameter_search_partb(train_loader: DataLoader,
                                     val_loader: DataLoader,
                                     model_fn:Callable[[], nn.Module]) -> List[Tuple[float, int, int, int]]:
    """
    Parameter search for our linear model using SGD.

    Args:
    train_loader: the train dataloader.
    val_loader: the validation dataloader.
    model_fn: a function that, when called, returns a torch.nn.Module.

    Returns:
    The learning rate with the least validation loss.
    """
    num_iter = 10
    best_3_results = []
    num_epoch = 15

    lrs = torch.linspace(10 ** (-6), 10 ** (-1), num_iter)
    Ms = [64, 128, 256, 512, 800, 900]
    Ks = [3, 5, 7]
    Ns = [2, 4, 7]

    for lr in lrs:
        for M in Ms:
          for K in Ks:
            for N in Ns:
              model = model_fn(M, K, N)
              optim = SGD(model.parameters(), lr, momentum=0.9)
              train_loss, train_acc, val_loss, val_acc = train(
                  model,
                  optim,
                  train_loader,
                  val_loader,
                  epochs=num_epoch
                  )

              curr_loss = min(val_loss)
              best_3_results.append((curr_loss, lr, M, K, N))
              best_3_results.sort(key=lambda x: x[0])
              best_3_results = best_3_results[:3]

    return [(lr, M, K, N) for _, lr, M, K, N in best_3_results]


results = parameter_search_partb(train_loader, val_loader, conv_model)

num_epochs = 20
for i, (lr, M, K, N) in enumerate(results):
  model = conv_model(M, K, N)
  optimizer = SGD(model.parameters(), lr, momentum=0.9)
  train_loss, train_accuracy, val_loss, val_accuracy = train(
      model, optimizer, train_loader, val_loader, num_epochs
  )
  epochs = range(1, num_epochs+1)
  plt.plot(epochs, train_accuracy, label=f"Train Accuracy, lr: {lr}, M: {M}, K: {K}, N: {N}, momentum: 0.9")
  plt.plot(epochs, val_accuracy, label=f"Validation Accuracy, lr: {lr}, M: {M}, K: {K}, N: {N}, momentum: 0.9", linestyle='dotted')

  test_loss, test_acc = evaluate(model, test_loader)
  print(f"Test Accuracy: {test_acc}, lr: {lr}, M: {M}, K: {K}, N: {N}, momentum: 0.9")

plt.axhline(y=0.65, linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
