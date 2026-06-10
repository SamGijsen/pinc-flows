import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinySubjectCNN(nn.Module):
    def __init__(self, num_rois, num_subjects):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(num_rois, 3))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=1)
        self.head = nn.Linear(32, num_subjects)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        return self.head(x)


def _window_batch(sequences, labels, starts):
    batch = []
    for seq, start in zip(sequences, starts):
        window = seq[start:start + 3].transpose(1, 0)
        batch.append(window)
    x = np.stack(batch, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return torch.from_numpy(x[:, None]), torch.from_numpy(y)


def run_subject_id_eval(
    train_sequences,
    train_labels,
    eval_real_sequences,
    eval_synth_sequences_by_name,
    eval_labels,
    *,
    num_rois,
    num_subjects,
    train_steps,
    batch_size,
    lr,
    seeds,
    device,
):
    train_sequences = np.asarray(train_sequences, dtype=np.float32)
    eval_real_sequences = np.asarray(eval_real_sequences, dtype=np.float32)
    eval_synth_sequences_by_name = {
        str(name): np.asarray(value, dtype=np.float32)
        for name, value in eval_synth_sequences_by_name.items()
    }
    train_labels = np.asarray(train_labels, dtype=np.int64)
    eval_labels = np.asarray(eval_labels, dtype=np.int64)

    seq_len = int(train_sequences.shape[1])
    assert seq_len >= 3
    n_positions = seq_len - 2

    real_all = []
    synth_all = {name: [] for name in eval_synth_sequences_by_name}
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        model = TinySubjectCNN(num_rois=num_rois, num_subjects=num_subjects).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

        model.train()
        for _ in range(int(train_steps)):
            idx = rng.integers(0, train_sequences.shape[0], size=int(batch_size))
            starts = rng.integers(0, n_positions, size=int(batch_size))
            x, y = _window_batch(train_sequences[idx], train_labels[idx], starts)
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        real_acc = []
        synth_acc = {name: [] for name in eval_synth_sequences_by_name}
        with torch.no_grad():
            for start in range(n_positions):
                starts = np.full(eval_real_sequences.shape[0], start, dtype=np.int64)
                x_real, y = _window_batch(eval_real_sequences, eval_labels, starts)
                pred_real = model(x_real.to(device)).argmax(dim=1).cpu().numpy()
                y_np = y.numpy()
                real_acc.append(float(np.mean(pred_real == y_np)))
                for name, sequences in eval_synth_sequences_by_name.items():
                    x_synth, _ = _window_batch(sequences, eval_labels, starts)
                    pred_synth = model(x_synth.to(device)).argmax(dim=1).cpu().numpy()
                    synth_acc[name].append(float(np.mean(pred_synth == y_np)))

        real_all.append(np.asarray(real_acc, dtype=np.float32))
        for name in synth_all:
            synth_all[name].append(np.asarray(synth_acc[name], dtype=np.float32))

    real_all = np.stack(real_all, axis=0)
    synth_stats = {}
    for name in eval_synth_sequences_by_name:
        values = np.stack(synth_all[name], axis=0)
        synth_stats[name] = {
            "mean": values.mean(axis=0),
            "std": values.std(axis=0),
            "all": values,
        }
    return {
        "positions": np.arange(n_positions, dtype=np.int64),
        "real_mean": real_all.mean(axis=0),
        "real_std": real_all.std(axis=0),
        "real_all": real_all,
        "synth": synth_stats,
    }
