import os, glob, cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.insert(0, "/content/anomaly-detection/src")
from config import Config
cfg = Config()

def load_frames_from_folder(folder, limit=None):
    paths = sorted(glob.glob(folder + "/*.png"))
    if limit:
        paths = paths[:limit]
    frames = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (cfg.FRAME_W, cfg.FRAME_H))
        frames.append(img.astype(np.float32) / 255.0)
    return np.array(frames)

def make_sequences(frames, seq_len, stride):
    seqs = []
    for start in range(0, len(frames) - seq_len + 1, stride):
        seqs.append(frames[start : start + seq_len])
    return np.array(seqs)

class NormalSequenceDataset(Dataset):
    def __init__(self, normal_dir, train=True):
        frames = load_frames_from_folder(normal_dir)
        split  = int(len(frames) * cfg.TRAIN_SPLIT)
        frames = frames[:split] if train else frames[split:]
        self.seqs = make_sequences(frames, cfg.SEQ_LEN, cfg.STRIDE)
        print("  Sequences (train={}): {}".format(train, len(self.seqs)))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return torch.from_numpy(self.seqs[idx]).unsqueeze(1)

class AnomalyTestDataset(Dataset):
    def __init__(self, normal_dir, anomaly_dirs, limit_per_class=500):
        all_seqs, all_labels = [], []
        normal_frames = load_frames_from_folder(normal_dir)
        split         = int(len(normal_frames) * cfg.TRAIN_SPLIT)
        val_frames    = normal_frames[split:]
        n_seqs        = make_sequences(val_frames, cfg.SEQ_LEN, cfg.STRIDE)[:limit_per_class]
        all_seqs.extend(n_seqs)
        all_labels.extend([0] * len(n_seqs))
        for adir in anomaly_dirs:
            if not os.path.exists(adir):
                continue
            a_frames = load_frames_from_folder(adir)
            a_seqs   = make_sequences(a_frames, cfg.SEQ_LEN, cfg.STRIDE)[:limit_per_class]
            all_seqs.extend(a_seqs)
            all_labels.extend([1] * len(a_seqs))
            print("  Anomaly seqs from {}: {}".format(
                os.path.basename(adir), len(a_seqs)))
        self.seqs   = np.array(all_seqs)
        self.labels = np.array(all_labels)
        print("  Test set: {} normal | {} anomaly seqs".format(
            (self.labels==0).sum(), (self.labels==1).sum()))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.seqs[idx]).unsqueeze(1)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def get_train_loader():
    ds = NormalSequenceDataset(cfg.NORMAL_DIR, train=True)
    return DataLoader(ds, batch_size=cfg.BATCH_SIZE,
                      shuffle=True, num_workers=2, pin_memory=True)

def get_val_loader():
    ds = NormalSequenceDataset(cfg.NORMAL_DIR, train=False)
    return DataLoader(ds, batch_size=cfg.BATCH_SIZE,
                      shuffle=False, num_workers=2, pin_memory=True)

def get_test_loader():
    anomaly_dirs = [cfg.FIGHT_DIR, cfg.ACCIDENT_DIR, cfg.THEFT_DIR]
    ds = AnomalyTestDataset(cfg.NORMAL_DIR, anomaly_dirs)
    return DataLoader(ds, batch_size=cfg.BATCH_SIZE,
                      shuffle=False, num_workers=2, pin_memory=True)
