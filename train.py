import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from prior.dataset import PriorDataset
from prior.genload import LoadPriorDataset
from model.tabicl import TabICL


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def configure_prior(prior_dir, batch_size=16, max_seq_len=128, max_features=10):
    """Set up a tabular dataset generator for synthetic data during training."""

    if prior_dir is None:
        print("No prior directory provided. Generating synthetic data on the fly.")
        dataset = PriorDataset(
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            max_features=max_features,
        )
    else:
        print(f"Loading prior dataset from directory: {prior_dir}")
        dataset = LoadPriorDataset(data_dir=prior_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=1,
        prefetch_factor=4,
    )

    return dataset, dataloader

def pad_nested_batch(batch):
    out = []
    for t in batch:
        if hasattr(t, "is_nested") and t.is_nested:
            out.append(t.to_padded_tensor(padding=0.0))
        else:
            out.append(t)
    return out


def split_batch_by_shape(batch):
    """
    Split one padded batch into groups where all samples share
    the same seq_len and train_size.
    """
    X, y, d, seq_lens, train_sizes = pad_nested_batch(batch)

    groups = {}

    for i in range(X.size(0)):
        seq_len_i = int(seq_lens[i].item()) if torch.is_tensor(seq_lens[i]) else int(seq_lens[i])
        train_size_i = int(train_sizes[i].item()) if torch.is_tensor(train_sizes[i]) else int(train_sizes[i])

        key = (seq_len_i, train_size_i)
        groups.setdefault(key, []).append(i)

    grouped_batches = []

    for (seq_len, train_size), indices in groups.items():
        idx = torch.tensor(indices, dtype=torch.long)

        sub_X = X[idx, :seq_len, :]
        sub_y = y[idx, :seq_len]
        sub_d = d[idx]

        max_features = int(sub_d.max().item()) if torch.is_tensor(sub_d.max()) else int(sub_d.max())
        sub_X = sub_X[..., :max_features]

        y_train = sub_y[:, :train_size]
        y_test = sub_y[:, train_size:seq_len]

        grouped_batches.append((sub_X, y_train, y_test, sub_d, (seq_len, train_size)))

    return grouped_batches

def train(model, dataloader, num_epochs=10, learning_rate=1e-3, max_steps_per_epoch=100):
    """Train normally across batches."""
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    model.train()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        running_loss = 0.0
        running_acc = 0.0
        steps = 0
        stop_epoch = False

        for i, batch in enumerate(dataloader):
            grouped_batches = split_batch_by_shape(batch)

            if len(grouped_batches) == 0:
                raise ValueError("split_batch_by_shape(batch) returned no groups.")

            for sub_batch in grouped_batches:
                X, y_train, y_test, d, shape_info = sub_batch

                X = X.float().to(device)
                y_train = y_train.long().to(device)
                y_test = y_test.long().to(device)
                d = d.to(device)

                optimiser.zero_grad()

                # Use this if your model API still accepts d
                logits = model(X, y_train, d)  # (B, test_size, C)

                # If your simplified model no longer uses d, use this instead:
                # logits = model(X, y_train)

                pred = logits.reshape(-1, logits.size(-1))
                true = y_test.reshape(-1)

                loss = criterion(pred, true)
                loss.backward()
                optimiser.step()

                acc = (pred.argmax(dim=1) == true).float().mean().item()

                running_loss += loss.item()
                running_acc += acc
                steps += 1

                if steps % 10 == 0:
                    print(f"Step {steps} | Loss {loss.item():.6f} | Acc {acc:.4f}")

                    preds = pred.argmax(dim=1)
                    unique_preds = torch.unique(preds)
                    pred_counts = torch.bincount(preds, minlength=logits.size(-1))
                    print(f"  Unique classes: {unique_preds.tolist()}")
                    print(f"  Distribution:   {pred_counts.tolist()}")
                    print(f"  Logit std:      {logits.std().item():.4f}")
                    true_counts = torch.bincount(true, minlength=logits.size(-1))
                    print("  Ground Truth Distribution:", true_counts.tolist())
                if steps >= max_steps_per_epoch:
                    stop_epoch = True
                    break

            if stop_epoch:
                break

        if steps == 0:
            raise ValueError("No training steps were run in this epoch.")

        print(f"Epoch avg loss: {running_loss / steps:.6f}")
        print(f"Epoch avg acc : {running_acc / steps:.4f}")

    return model


def overfit_single_batch(model, dataloader, num_steps=500, learning_rate=1e-3):
    print("\nRunning overfit-single-batch test")

    model.train()

    batch = next(iter(dataloader))
    grouped_batches = split_batch_by_shape(batch)

    if len(grouped_batches) == 0:
        raise ValueError("split_batch_by_shape(batch) returned no groups.")

    X, y_train, y_test, d, shape_info = max(grouped_batches, key=lambda g: g[0].shape[0])
    seq_len, train_size = shape_info

    X = X.float().to(device)
    y_train = y_train.long().to(device)
    y_test = y_test.long().to(device)
    d = d.to(device)

    print(
        f"Using overfit batch: "
        f"X={tuple(X.shape)}, y_train={tuple(y_train.shape)}, y_test={tuple(y_test.shape)}, "
        f"seq_len={seq_len}, train_size={train_size}"
    )

    if y_test.shape[1] != seq_len - train_size:
        raise ValueError(
            f"Bad split: y_test.shape[1]={y_test.shape[1]}, expected {seq_len - train_size}"
        )

    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)

    for step in range(num_steps):
        optimiser.zero_grad(set_to_none=True)

        logits = model(X, y_train, d)

        if logits.shape[1] != y_test.shape[1]:
            raise ValueError(
                f"logit/test mismatch: logits.shape={tuple(logits.shape)}, y_test.shape={tuple(y_test.shape)}"
            )

        pred = logits.reshape(-1, logits.size(-1))
        true = y_test.reshape(-1)

        loss = criterion(pred, true)
        loss.backward()
        optimiser.step()

        if step % 50 == 0 or step == num_steps - 1:
            with torch.no_grad():
                acc = (pred.argmax(dim=1) == true).float().mean().item()
            print(f"Step {step:4d}/{num_steps} | Loss {loss.item():.6f} | Acc {acc:.4f}")

    print("Final overfit loss:", loss.item())
    return model

def test(model, dataloader, num_batches=2):
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_steps = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            grouped_batches = split_batch_by_shape(batch)

            for sub_batch in grouped_batches:
                X, y_train, y_test, d, shape_info = sub_batch
                seq_len, train_size = shape_info

                X = X.float().to(device)
                y_train = y_train.long().to(device)
                y_test = y_test.long().to(device)
                d = d.to(device)

                # sanity checks
                if X.shape[1] != seq_len:
                    raise ValueError(f"X seq len mismatch: X.shape[1]={X.shape[1]}, seq_len={seq_len}")

                if y_train.shape[1] != train_size:
                    raise ValueError(
                        f"y_train mismatch: y_train.shape[1]={y_train.shape[1]}, train_size={train_size}"
                    )

                if y_test.shape[1] != seq_len - train_size:
                    raise ValueError(
                        f"y_test mismatch: y_test.shape[1]={y_test.shape[1]}, seq_len-train_size={seq_len-train_size}"
                    )

                logits = model(X, y_train, d)

                if logits.shape[1] != y_test.shape[1]:
                    raise ValueError(
                        f"logit/test mismatch: logits.shape={tuple(logits.shape)}, y_test.shape={tuple(y_test.shape)}"
                    )

                pred = logits.reshape(-1, logits.size(-1))
                true = y_test.reshape(-1)

                loss = criterion(pred, true)
                acc = (pred.argmax(dim=1) == true).float().mean().item()

                total_loss += loss.item()
                total_acc += acc
                total_steps += 1

            if i + 1 >= num_batches:
                break

    model.train()
    return total_loss / total_steps, total_acc / total_steps


if __name__ == "__main__":
    batch_size = 8
    max_seq_len = 256
    max_features = 20
    prior_dir = None  # set None to generate on the fly

    dataset, dataloader = configure_prior(
        prior_dir=prior_dir,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        max_features=max_features,
    )

    if prior_dir is not None:
        vocab_size = dataset.metadata.get("max_classes")
    else:
        vocab_size = dataset.max_classes

    print("Vocab size:", vocab_size)

    # IMPORTANT:
    # These argument names should match your current TabICL implementation.
    model = TabICL(
        max_classes=vocab_size,
        embed_dim=256,       # was 16
        col_num_blocks=4,
        col_nhead=4,
        col_num_inds=32,
        row_num_blocks=4,
        row_nhead=4,
        row_num_cls=4,
        icl_num_blocks=4,
        icl_nhead=4, 
        ff_factor=2,
        dropout=0.0,
        activation="gelu",
        norm_first=True,
    ).to(device)
    before_loss, before_acc = test(model, dataloader, num_batches=5)
    print(f"\nBefore training | Loss {before_loss:.6f} | Acc {before_acc:.4f}")

    # model = overfit_single_batch(model, dataloader, num_steps=500)

    # after_overfit_loss, after_overfit_acc = test(model, dataloader, num_batches=2)
    # print(f"\nAfter overfit     | Loss {after_overfit_loss:.6f} | Acc {after_overfit_acc:.4f}")

    model = train(model, dataloader, num_epochs=5, learning_rate=1e-4, max_steps_per_epoch=50)

    final_loss, final_acc = test(model, dataloader, num_batches=5)
    print(f"\nAfter training    | Loss {final_loss:.6f} | Acc {final_acc:.4f}")

    model.eval()
    
    test_batch = next(iter(dataloader))
    grouped = split_batch_by_shape(test_batch)
    
    X, y_train, y_test, d, _ = grouped[0]
    
    X = X.float().to(device)
    y_train = y_train.long().to(device)
    y_test = y_test.long().to(device)
    d = d.to(device)

    with torch.no_grad():
        logits = model(X, y_train, d=d)
        preds = logits.argmax(dim=-1)

    print("Test Predictions: ", preds[0, :20].tolist())
    print("Test Ground Truth:", y_test[0, :20].tolist())
    
    accuracy = (preds == y_test).float().mean().item()
    print(f"Batch Accuracy:    {accuracy:.4f}")

    unique_in_batch = torch.unique(preds)
    print(f"Unique classes predicted in this batch: {unique_in_batch.tolist()}")
    pred_counts = torch.bincount(preds.flatten(), minlength=vocab_size)
    print("Prediction distribution:", pred_counts.tolist())
    gt_counts = torch.bincount(y_test.flatten(), minlength=10)
    print(f"Ground Truth Distribution: {gt_counts.tolist()}")