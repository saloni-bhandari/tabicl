import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from prior.dataset import PriorDataset
from prior.genload import LoadPriorDataset
from model.tabicl import TabICL


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


ANALYSIS_DIR = Path("analysis")
ANALYSIS_DIR.mkdir(exist_ok=True)
STEP_LOG_INTERVAL = 10
DIST_LOG_INTERVAL = 50


def configure_prior(prior_dir, batch_size=16, max_seq_len=128, max_features=10, max_classes=10):
    """Set up a tabular dataset generator for synthetic data during training."""

    if prior_dir is None:
        print("No prior directory provided. Generating synthetic data on the fly.")
        dataset = PriorDataset(
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            max_features=max_features,
            prior_type="mix_scm",
            max_classes=max_classes,
        )
    else:
        print(f"Loading prior dataset from directory: {prior_dir}")
        dataset = LoadPriorDataset(data_dir=prior_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=0,
        prefetch_factor=None,
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

def make_scheduler(optimiser, schedule, total_steps, lr_init, lr_end=0.0):
    if schedule == "cosine_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimiser, T_0=max(1, total_steps // 5), eta_min=1e-6
        )
    elif schedule == "polynomial":
        def poly_lambda(step):
            frac = max(0.0, 1.0 - step / total_steps)
            lr = (lr_init - lr_end) * (frac ** 2) + lr_end
            return lr / lr_init
        return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=poly_lambda)
    else:
        return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lambda _: 1.0)


def append_jsonl(path, record):
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def save_summary_artifacts(run_summary, base_name="train_metrics"):
    json_path = ANALYSIS_DIR / f"{base_name}_summary.json"
    pt_path = ANALYSIS_DIR / f"{base_name}_summary.pt"
    json_path.write_text(json.dumps(run_summary, indent=2))
    torch.save(run_summary, pt_path)
    print(f"Saved analysis summary to {json_path} and {pt_path}")


def train(model, dataloader, num_epochs=10, learning_rate=1e-3, max_steps_per_epoch=100,
            lr_schedule="constant", lr_end=0.0, frozen_modules=None, history=None, stage_name="train",
            step_log_path=None, dist_log_path=None):
    """Train normally across batches."""
    max_steps = num_epochs * max_steps_per_epoch

    frozen_params = set()
    if frozen_modules:
        for m in frozen_modules:
            for p in m.parameters():
                frozen_params.add(id(p))

    trainable_params = [p for p in model.parameters()
                        if id(p) not in frozen_params and p.requires_grad]
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)
    scheduler = make_scheduler(optimiser, lr_schedule, max_steps, learning_rate, lr_end)
    model.train()
    
    if history is None:
        history = {"epochs": []}

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
                if not torch.isfinite(loss):
                    print(f"  WARNING: non-finite loss {loss.item()}, skipping step")
                    optimiser.zero_grad()
                    continue
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
                optimiser.step()
                scheduler.step()

                acc = (pred.argmax(dim=1) == true).float().mean().item()
                preds = pred.argmax(dim=1)
                pred_counts = torch.bincount(preds, minlength=logits.size(-1)).cpu().tolist()
                true_counts = torch.bincount(true, minlength=logits.size(-1)).cpu().tolist()
                step_record = {
                    "stage": stage_name,
                    "epoch": epoch + 1,
                    "step": steps + 1,
                    "loss": loss.item(),
                    "accuracy": acc,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "seq_len": int(shape_info[0]),
                    "train_size": int(shape_info[1]),
                    "batch_size": int(X.shape[0]),
                    "max_features": int(X.shape[-1]),
                    "active_features_max": int(d.max().item()),
                    "logit_std": logits.std().item(),
                }
                if step_log_path is not None:
                    append_jsonl(step_log_path, step_record)
                if dist_log_path is not None and steps % DIST_LOG_INTERVAL == 0:
                    append_jsonl(
                        dist_log_path,
                        {
                            "stage": stage_name,
                            "epoch": epoch + 1,
                            "step": steps + 1,
                            "pred_distribution": pred_counts,
                            "true_distribution": true_counts,
                        },
                    )

                running_loss += loss.item()
                running_acc += acc
                steps += 1

                if steps % STEP_LOG_INTERVAL == 0:
                    print(f"Step {steps} | Loss {loss.item():.6f} | Acc {acc:.4f}")

                    unique_preds = torch.unique(preds)
                    print(f"  Unique classes: {unique_preds.tolist()}")
                    print(f"  Distribution:   {pred_counts}")
                    print(f"  Logit std:      {logits.std().item():.4f}")
                    print("  Ground Truth Distribution:", true_counts)
                if steps >= max_steps_per_epoch:
                    stop_epoch = True
                    break

            if stop_epoch:
                break

        if steps == 0:
            raise ValueError("No training steps were run in this epoch.")

        epoch_record = {
            "stage": stage_name,
            "epoch": epoch + 1,
            "avg_loss": running_loss / steps,
            "avg_accuracy": running_acc / steps,
            "num_steps": steps,
        }
        history["epochs"].append(epoch_record)
        print(f"Epoch avg loss: {epoch_record['avg_loss']:.6f}")
        print(f"Epoch avg acc : {epoch_record['avg_accuracy']:.4f}")

    return model, history


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
    return {
        "loss": total_loss / total_steps,
        "accuracy": total_acc / total_steps,
        "num_batches": num_batches,
        "num_steps": total_steps,
    }

def inspect_predictions(model, dataloader, vocab_size, device):
    model.eval()
    batch = next(iter(dataloader))
    grouped = split_batch_by_shape(batch)
    X, y_train, y_test, d, _ = grouped[0]

    X       = X.float().to(device)
    y_train = y_train.long().to(device)
    y_test  = y_test.long().to(device)
    d       = d.to(device)

    with torch.no_grad():
        logits = model(X, y_train, d=d)
        preds  = logits.argmax(dim=-1)

    accuracy = (preds == y_test).float().mean().item()
    pred_counts = torch.bincount(preds.flatten(), minlength=vocab_size)
    gt_counts = torch.bincount(y_test.flatten(), minlength=vocab_size)
    summary = {
        "accuracy": accuracy,
        "unique_classes_predicted": unique_in_batch.tolist() if (unique_in_batch := torch.unique(preds)).numel() > 0 else [],
        "pred_distribution": pred_counts.cpu().tolist(),
        "ground_truth_distribution": gt_counts.cpu().tolist(),
        "logit_std": logits.std().item(),
        "logit_mean": logits.mean().item(),
    }

    print(f"Batch Accuracy:    {accuracy:.4f}")
    print(f"Unique classes predicted: {summary['unique_classes_predicted']}")
    print(f"Pred distribution: {summary['pred_distribution']}")
    print(f"Ground truth distribution:   {summary['ground_truth_distribution']}")
    model.train()
    return summary


if __name__ == "__main__":
    batch_size = 1
    max_seq_len = 100
    max_features = 10
    prior_dir = None  # set None to generate on the fly
    max_classes = 2

    dataset, dataloader = configure_prior(
        prior_dir=prior_dir,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        max_features=max_features,
        max_classes=max_classes,
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
        embed_dim=16,       # was 16
        col_num_blocks=3,
        col_nhead=2,
        col_num_inds=4,
        row_num_blocks=3,
        row_nhead=2,
        row_num_cls=4,
        icl_num_blocks=4,
        icl_nhead=4, 
        ff_factor=2,
        dropout=0.0,
        activation="gelu",
        norm_first=True,
    ).to(device)
    run_history = {
        "device": str(device),
        "max_classes": max_classes,
        "max_features": max_features,
        "initial_eval": None,
        "stages": {},
    }
    step_log_path = ANALYSIS_DIR / "train_metrics_steps.jsonl"
    dist_log_path = ANALYSIS_DIR / "train_metrics_distributions.jsonl"
    for log_path in (step_log_path, dist_log_path):
        log_path.write_text("")

    initial_eval = test(model, dataloader, num_batches=5)
    run_history["initial_eval"] = initial_eval
    print(f"\nBefore training | Loss {initial_eval['loss']:.6f} | Acc {initial_eval['accuracy']:.4f}")
    save_summary_artifacts(run_history)

    # model = overfit_single_batch(model, dataloader, num_steps=500)

    # after_overfit_loss, after_overfit_acc = test(model, dataloader, num_batches=2)
    # print(f"\nAfter overfit     | Loss {after_overfit_loss:.6f} | Acc {after_overfit_acc:.4f}")

    # model = train(model, dataloader, num_epochs=5, learning_rate=1e-4, max_steps_per_epoch=50)

    # final_loss, final_acc = test(model, dataloader, num_batches=5)
    # print(f"\nAfter training    | Loss {final_loss:.6f} | Acc {final_acc:.4f}")
    print("\n=== STAGE 1 ===")
    stage1_start = time.time()
    _, dataloader1 = configure_prior(None, batch_size=8, max_seq_len=1024, max_features=max_features, max_classes=max_classes)
    stage1_history = {"epochs": []}
    model, stage1_history = train(
        model, dataloader1, num_epochs=1, learning_rate=1e-4, max_steps_per_epoch=50,
        lr_schedule="cosine_restarts", history=stage1_history, stage_name="stage1",
        step_log_path=step_log_path, dist_log_path=dist_log_path
    )
    stage1_eval = test(model, dataloader1)
    stage1_end = time.time()
    stage1_duration = stage1_end - stage1_start
    
    print(f"Stage 1 test | Loss {stage1_eval['loss']:.4f} | Acc {stage1_eval['accuracy']:.4f}")
    print(f"--> Stage 1 Total Time: {stage1_duration:.2f} seconds ({stage1_duration / 60:.2f} minutes)")
    stage1_predictions = inspect_predictions(model, dataloader1, vocab_size, device)
    run_history["stages"]["stage1"] = {
        "duration_seconds": stage1_duration,
        "train": stage1_history,
        "eval": stage1_eval,
        "prediction_summary": stage1_predictions,
        "checkpoint": "tabicl_stage1.pt",
    }
    torch.save(model.state_dict(), "tabicl_stage1.pt")
    print("Saved to tabicl_stage1.pt")
    save_summary_artifacts(run_history)
    del dataloader1  # Explicitly delete the old loader
    torch.cuda.empty_cache()
    # model.load_state_dict(torch.load("tabicl_stage1.pt", map_location=device, weights_only=True))

    print("\n=== STAGE 2 ===")
    stage2_start = time.time()
    _, dataloader2 = configure_prior(None, batch_size=4, max_seq_len=2048, max_features=max_features, max_classes=max_classes)
    stage2_history = {"epochs": []}
    model, stage2_history = train(
        model, dataloader2, num_epochs=1, learning_rate=2e-5, max_steps_per_epoch=50,
        lr_schedule="polynomial", lr_end=5e-6, history=stage2_history, stage_name="stage2",
        step_log_path=step_log_path, dist_log_path=dist_log_path
    )
    stage2_eval = test(model, dataloader2)
    stage2_end = time.time()
    stage2_duration = stage2_end - stage2_start
    
    print(f"Stage 2 test | Loss {stage2_eval['loss']:.4f} | Acc {stage2_eval['accuracy']:.4f}")
    print(f"--> Stage 2 Total Time: {stage2_duration:.2f} seconds ({stage2_duration / 60:.2f} minutes)")
    stage2_predictions = inspect_predictions(model, dataloader2, vocab_size, device)
    run_history["stages"]["stage2"] = {
        "duration_seconds": stage2_duration,
        "train": stage2_history,
        "eval": stage2_eval,
        "prediction_summary": stage2_predictions,
        "checkpoint": "tabicl_stage2.pt",
    }
    torch.save(model.state_dict(), "tabicl_stage2.pt")
    print("Saved to tabicl_stage2.pt")
    save_summary_artifacts(run_history)
    del dataloader2
    torch.cuda.empty_cache()

    print("\n=== STAGE 3 ===")
    stage3_start = time.time()
    _, dataloader3 = configure_prior(None, batch_size=1, max_seq_len=16384, max_features=max_features, max_classes=max_classes)
    stage3_history = {"epochs": []}
    model, stage3_history = train(
        model, dataloader3, num_epochs=1, learning_rate=2e-6, max_steps_per_epoch=50,
        lr_schedule="constant", frozen_modules=[model.col_embedder, model.row_interactor],
        history=stage3_history, stage_name="stage3",
        step_log_path=step_log_path, dist_log_path=dist_log_path
    )
    stage3_eval = test(model, dataloader3)
    stage3_end = time.time()
    stage3_duration = stage3_end - stage3_start
    
    print(f"Stage 3 test | Loss {stage3_eval['loss']:.4f} | Acc {stage3_eval['accuracy']:.4f}")
    print(f"--> Stage 3 Total Time: {stage3_duration:.2f} seconds ({stage3_duration / 60:.2f} minutes)")
    stage3_predictions = inspect_predictions(model, dataloader3, vocab_size, device)
    run_history["stages"]["stage3"] = {
        "duration_seconds": stage3_duration,
        "train": stage3_history,
        "eval": stage3_eval,
        "prediction_summary": stage3_predictions,
        "checkpoint": "tabicl_test_run.pt",
    }
    save_summary_artifacts(run_history)
    inspect_predictions(model, dataloader3, vocab_size, device)

    print("\nTest run complete.")
    torch.save(model.state_dict(), "tabicl_test_run.pt")
    total_time = stage1_duration + stage2_duration + stage3_duration
    run_history["total_duration_seconds"] = total_time
    run_history["step_log_path"] = str(step_log_path)
    run_history["distribution_log_path"] = str(dist_log_path)
    save_summary_artifacts(run_history)
    print(f"Total Script Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    print("Saved to tabicl_test_run.pt")
