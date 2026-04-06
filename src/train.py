
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_msssim import ssim
from tqdm import tqdm
import os, sys
sys.path.insert(0, "/content/anomaly-detection/src")
from config import Config
from model import ConvLSTMAE
from data_loader import get_train_loader, get_val_loader

cfg = Config()


def loss_fn(x, recon):
    B, T, C, H, W = x.shape
    xf = x.view(B*T, C, H, W)
    rf = recon.view(B*T, C, H, W)
    mse  = F.mse_loss(rf, xf)
    sval = ssim(rf, xf, data_range=1.0, size_average=True)
    return mse + cfg.SSIM_WEIGHT * (1 - sval), mse.item()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model     = ConvLSTMAE().to(device)
    optimizer = Adam(model.parameters(), lr=cfg.LR, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

    train_loader = get_train_loader()
    val_loader   = get_val_loader()

    best_loss   = float("inf")
    start_epoch = 0
    history     = {"train": [], "val": []}

    if os.path.exists(cfg.CHECKPOINT):
        state = torch.load(cfg.CHECKPOINT, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_loss   = state["loss"]
        print("Resumed from epoch {}, best loss {:.6f}".format(
            start_epoch, best_loss))

    for epoch in range(start_epoch, cfg.EPOCHS):
        # Train
        model.train()
        t_loss = 0
        for batch in tqdm(train_loader,desc="Ep {}/{} train".format(epoch+1, cfg.EPOCHS),leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss, _ = loss_fn(batch, recon)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)

        # Validate
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss, _ = loss_fn(batch, recon)
                v_loss += loss.item()
        v_loss /= len(val_loader)

        scheduler.step(v_loss)
        history["train"].append(t_loss)
        history["val"].append(v_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print("Ep {:03d} | Train: {:.5f} | Val: {:.5f} | LR: {:.1e}".format(epoch+1, t_loss, v_loss, lr_now))

        if v_loss < best_loss:
            best_loss = v_loss
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss":      best_loss
            }, cfg.CHECKPOINT)
            print("  Saved (val_loss={:.5f})".format(best_loss))

    print("Training done. Best val loss: {:.5f}".format(best_loss))
    return model, history
