import torch
import numpy as np
import os, sys
sys.path.insert(0, "/content/anomaly-detection/src")
from config import Config
from model import ConvLSTMAE
from data_loader import load_frames_from_folder, make_sequences
cfg = Config()

def load_model(device):
    model = ConvLSTMAE().to(device)
    state = torch.load(cfg.CHECKPOINT, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    print("Model loaded from epoch {}, loss={:.5f}".format(
        state["epoch"] + 1, state["loss"]))
    return model

def compute_scores_on_folder(model, folder, device):
    frames = load_frames_from_folder(folder)
    seqs   = make_sequences(frames, cfg.SEQ_LEN, cfg.STRIDE)
    scores = []
    model.eval()
    with torch.no_grad():
        for seq in seqs:
            t = torch.from_numpy(seq).unsqueeze(1).unsqueeze(0).to(device)
            r = model(t)
            s = model.anomaly_score(t, r).item()
            scores.append(s)
    return np.array(scores)

def get_adaptive_threshold(scores, k=None):
    if k is None:
        k = cfg.THRESHOLD_K
    mu        = scores.mean()
    sigma     = scores.std()
    threshold = mu + k * sigma
    print("  mean={:.5f}  std={:.5f}  threshold={:.5f}".format(
        mu, sigma, threshold))
    return threshold, mu, sigma
