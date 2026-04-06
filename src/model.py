import torch
import torch.nn as nn
import sys
sys.path.insert(0, "/content/anomaly-detection/src")
from config import Config
cfg = Config()

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_ch = hidden_ch
        self.gates = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch,
                               kernel_size, padding=pad, bias=True)
        self.bn = nn.BatchNorm2d(4 * hidden_ch)

    def forward(self, x, h, c):
        gates      = self.bn(self.gates(torch.cat([x, h], dim=1)))
        i, f, g, o = gates.chunk(4, dim=1)
        c_ = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_ = torch.sigmoid(o) * torch.tanh(c_)
        return h_, c_

    def init_hidden(self, B, spatial, device):
        h = torch.zeros(B, self.hidden_ch, *spatial, device=device)
        return h, h.clone()

class SpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, cfg.LSTM_CH * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(cfg.LSTM_CH * 2), nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class SpatialDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(cfg.LSTM_CH, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class ConvLSTMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.sp_enc     = SpatialEncoder()
        self.sp_dec     = SpatialDecoder()
        self.enc_lstm   = ConvLSTMCell(cfg.LSTM_CH * 2, cfg.LSTM_CH)
        self.dec_lstm   = ConvLSTMCell(cfg.LSTM_CH,     cfg.LSTM_CH)
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cfg.LSTM_CH * 16 * 16, cfg.LATENT_DIM),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(cfg.LATENT_DIM, cfg.LSTM_CH * 16 * 16),
            nn.ReLU()
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        dev = x.device
        h_e, c_e = self.enc_lstm.init_hidden(B, (16, 16), dev)
        for t in range(T):
            feat     = self.sp_enc(x[:, t])
            h_e, c_e = self.enc_lstm(feat, h_e, c_e)
        z = self.bottleneck(h_e).view(B, cfg.LSTM_CH, 16, 16)
        h_d, c_d = self.dec_lstm.init_hidden(B, (16, 16), dev)
        recons, inp = [], z
        for _ in range(T):
            h_d, c_d = self.dec_lstm(inp, h_d, c_d)
            recons.append(self.sp_dec(h_d))
            inp = h_d
        return torch.stack(recons, dim=1)

    def anomaly_score(self, x, recon):
        return torch.mean((x - recon) ** 2, dim=[1, 2, 3, 4])

def count_params(model):
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters: {:,}".format(n))
    return n
