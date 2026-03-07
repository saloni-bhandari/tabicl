import os

import model
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from model.tabicl import TabICL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class TabularDataset(Dataset):
    def __init__(self, filepath, num_rows_per_datapoint=4):
        self.df = pd.read_csv(filepath)
        self.features = self.df.iloc[:, :-1]
        self.labels = self.df.iloc[:, -1]
        self.num_rows_per_datapoint = num_rows_per_datapoint

    def get_num_labels(self):
        return self.labels.unique().shape[0]

    def __len__(self):
        return self.df.shape[0]//self.num_rows_per_datapoint

    def __getitem__(self, idx):
        start_idx = idx * self.num_rows_per_datapoint
        end_idx = start_idx + self.num_rows_per_datapoint

        features = self.features.iloc[start_idx:end_idx, :].T
        labels = self.labels.iloc[start_idx:end_idx]
        
        return np.array(features), np.array(labels)
    
def train(num_epochs=100, learning_rate=1e-3, test_size=1, batch_size=16):

    embedding_dim = 32
    num_rows_per_datapoint = 4

    training_data = TabularDataset(
        filepath="synthetic_tabicl_data/dataset_0.csv",
        num_rows_per_datapoint=num_rows_per_datapoint
    )

    dataloader = DataLoader(training_data, batch_size=batch_size, pin_memory=True)

    vocab_size = training_data.get_num_labels()

    model = TabICL(vocab_size=vocab_size, embedding_dim=embedding_dim).to(device)
    print(next(model.parameters()).device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)

    # store epoch losses for later analysis
    epoch_losses = []

    for t in range(num_epochs):

        print(f"\nTraining epoch number: {t}/{num_epochs}")

        running_loss = 0.0
        num_batches = 0

        for i, (features, labels) in enumerate(dataloader):
            
            print(f"Mini-table {i} | Shape of data: {features.shape}, {labels.shape}")

            features = features.float()

            # forward pass
            outputs = model(features, labels, test_size=test_size)

            # reshape outputs to (test_size * batch, vocab_size)
            outputs = outputs.view(-1, vocab_size)

            # ground truth labels for test rows
            ground_truth = labels[:, -test_size:].squeeze(-1).long()

            loss = criterion(outputs, ground_truth)

            # backward pass
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            # accumulate statistics
            running_loss += loss.item()
            num_batches += 1

            print("batch_loss:", loss.item())

        # compute average epoch loss
        epoch_loss = running_loss / num_batches
        epoch_losses.append(epoch_loss)

        print(f"\nEpoch {t} average loss: {epoch_loss:.6f}")

    print("\nTraining finished.")
    print("Epoch losses:", epoch_losses)

    return model, epoch_losses


def overfit_single_batch(training_data, model, vocab_size, test_size=1, batch_size=16):

    print("\nRunning overfit-single-batch test")

    dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    # get batch_size number of mini-tables
    features, labels = next(iter(dataloader))

    features = features.float().to(device)
    labels = labels.to(device)

    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=1e-3)

    num_steps = 1000

    for step in range(num_steps):
        # print(f"\nStep {step}/{num_steps}")

        outputs = model(features, labels, test_size=test_size).to(device)
        outputs = outputs.view(-1, vocab_size)

        ground_truth = labels[:, -test_size:].squeeze(-1).long().to(device)

        loss = criterion(outputs, ground_truth)

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        if step % 50 == 0:
            print(f"Step {step} | Loss: {loss.item():.6f}")

    print("\nFinal loss:", loss.item())

    return model


def test(model, dataset, vocab_size, test_size=1, batch_size=16):

    model.eval()  # disable dropout etc

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total = 0
    correct = 0

    with torch.no_grad():

            features, labels = next(iter(dataloader))
            print(f"Class counts in test batch: {torch.bincount(labels[:, -test_size:].squeeze(-1).long())}")
            print(f"Fraction of each class in test batch: {torch.bincount(labels[:, -test_size:].squeeze(-1).long()) / labels[:, -test_size:].squeeze(-1).shape[0]}")

            features = features.float().to(device)
            labels = labels.to(device)
            outputs = model(features, labels, test_size=test_size).to(device)
            outputs = outputs.view(-1, vocab_size)

            ground_truth = labels[:, -test_size:].squeeze(-1).long()

            predictions = outputs.argmax(dim=-1)

            # print("\nMini-table", i)
            # print("Prediction :", predictions)
            # print("GroundTruth:", ground_truth)

            correct += (predictions == ground_truth).sum().item()
            total += ground_truth.numel()

    accuracy = correct / total

    # print("\nTest Accuracy:", accuracy)

    model.train()  # switch back to training mode

    return accuracy

if __name__ == "__main__":

    training_data = TabularDataset(
        filepath="synthetic_tabicl_data/dataset_0.csv",
        num_rows_per_datapoint=4
    )

    vocab_size = training_data.get_num_labels()
    batch_size = 16

    model = TabICL(vocab_size=vocab_size, embedding_dim=32).to(device)
    print(f"Before training model: test accuracy: {test(model, training_data, vocab_size, batch_size=batch_size)}")
    model = overfit_single_batch(training_data, model, vocab_size, batch_size=batch_size)
    
    print(f"After training model: test accuracy: {test(model, training_data, vocab_size, batch_size=batch_size)}")