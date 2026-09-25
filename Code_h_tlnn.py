#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
"""  
H-TLNN: Hierarchical Transformer–Liquid Neural Network for Automatic Modulation  
Recognition in PLC / Smart-Grid channels.  

Standalone PyTorch re-implementation of  
  "A Coarse-to-Fine Modulation Classification Scheme Utilizing Transformers and  
   Liquid Neural Networks for Smart Grids"  (Hu, Huang, Lin, Huang)  

Pipeline (paper section in brackets):  
  1. Synthetic PLC dataset            [4.1.1 / 4.1.2]  
       8 modulations, RRC pulse shaping, Zimmermann–Dostert multipath,  
       CFO / SCO / phase offset, Middleton Class-A impulsive noise.  
  2. Stage-1 features: 12 higher-order cumulants, C21-normalised      [3.1.1]  
  3. Stage-1 classifier: HOC-Transformer (3 families)                 [3.1.2]  
  4. Stage-2 features: 32x32x3 "3D constellation matrix", read row-by-row [3.2.1]  
  5. Stage-2 classifiers: bank of Liquid-Time-Constant (LTC) experts   [3.2.2 / 3.3.3]  
  6. Decoupled two-phase training                                      [3.3]  
  7. Evaluation: Acc-vs-SNR, routing-error decomposition, confusion matrix,  
     per-class precision/recall/F1                                     [4.2, 4.3.10, 4.3.11, 4.3.15]  

Dependencies: numpy, torch (>=1.13), matplotlib (optional, for plots)  

Quick demo (small dataset, few epochs):  
    python h_tlnn.py --quick  
Paper-scale run (153,600 frames, 100/50 epochs; needs a GPU and time):  
    python h_tlnn.py --full  
"""  

import argparse  
import math  
import os  
import time  
from dataclasses import dataclass  
from functools import lru_cache  

import numpy as np  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from torch.utils.data import DataLoader, TensorDataset  

# ----------------------------------------------------------------------------- #  
#                               Global definitions                              #  
# ----------------------------------------------------------------------------- #  
MODS = ["BPSK", "QPSK", "8PSK", "16QAM", "32QAM", "64QAM", "128QAM", "256QAM"]  
FAMILY_NAMES = ["PSK", "Square-QAM", "Cross-QAM"]  
FAMILIES = {0: [0, 1, 2],      # PSK        : BPSK, QPSK, 8PSK  
            1: [3, 5, 7],      # Square-QAM : 16QAM, 64QAM, 256QAM  
            2: [4, 6]}         # Cross-QAM  : 32QAM, 128QAM  
FAMILY_OF = {m: k for k, ms in FAMILIES.items() for m in ms}  
LOCAL_IDX = {m: i for k, ms in FAMILIES.items() for i, m in enumerate(ms)}  

# (order p, #conjugates q) for the 12 cumulants of Eq. (10)  
HOC_ORDERS = [(2, 0), (2, 1), (4, 0), (4, 1), (4, 2),  
              (6, 0), (6, 1), (6, 2), (6, 3), (8, 0), (8, 1), (8, 4)]  


@dataclass  
class Config:  
    # ---- signal framing (4.1.1) ----  
    n_samples: int = 1024  
    sps: int = 4                      # oversampling factor  (fs = 400 kHz)  
    rs: float = 100e3                 # symbol rate  
    rrc_beta: float = 0.25  
    rrc_span: int = 6                 # symbols  
    rx_matched_filter: bool = True    # receiver RRC matched filter (after noise)  
    # ---- multipath channel (2.1 / 4.1.2) ----  
    n_paths: int = 4  
    tau_mean_us: float = 0.5  
    gain_range: tuple = (0.2, 0.8)  
    a0: float = 0.0  
    a1: float = 7.8e-10  
    k_exp: float = 1.0  
    f_carrier: float = 10e6           # centre frequency used in A(f,d)  
    v_prop: float = 2e8               # propagation speed (m/s) -> d_i = tau_i * v  
    crosstalk_db: float = -25.0       # inter-modal coupling (2x2 MIMO); None disables  
    # ---- transceiver imperfections ----  
    cfo_max: float = 0.01             # fraction of symbol rate  
    sco_ppm: float = 10.0  
    # ---- Middleton Class-A noise (2.2 / 4.1.2) ----  
    class_a_A: float = 0.1  
    class_a_gamma: float = 0.01  
    class_a_mmax: int = 20  
    # ---- dataset ----  
    snr_list: tuple = tuple(range(-10, 21, 2))  
    frames_per_mod_snr: int = 1200  
    split: tuple = (0.70, 0.15, 0.15)  
    # ---- stage-2 features ----  
    cm_size: int = 32  
    cm_range: float = 2.5             # I/Q clipping range after unit-power normalisation  
    # ---- stage-1 Transformer (4.1.4) ----  
    d_model: int = 128  
    n_heads: int = 4  
    n_layers: int = 4  
    d_ff: int = 256  
    dropout: float = 0.2  
    # ---- stage-2 LTC experts (4.1.4) ----  
    lnn_layers: int = 3  
    lnn_hidden: int = 64  
    w_sys: float = 1.0  
    ode_dt: float = 0.1               # bounded in [0.01, 0.1] in the paper  
    ode_steps: int = 5                # RK4 sub-steps between consecutive rows  
    # ---- training (4.1.4) ----  
    batch_size: int = 256  
    epochs_stage1: int = 100  
    epochs_stage2: int = 50  
    lr: float = 1e-3  
    lr_min: float = 1e-6  
    weight_decay: float = 1e-5  
    patience: int = 15  
    grad_clip: float = 1.0  

    @property  
    def fs(self):  
        return self.rs * self.sps  


# ----------------------------------------------------------------------------- #  
#                         1.  Signal / channel simulation                       #  
# ----------------------------------------------------------------------------- #  
def make_constellation(name: str) -> np.ndarray:  
    """Unit-average-power constellation points."""  
    if name == "BPSK":  
        pts = np.array([1.0, -1.0], dtype=complex)  
    elif name.endswith("PSK"):  
        M = int(name[:-3])  
        k = np.arange(M)  
        pts = np.exp(1j * (2 * np.pi * k / M + np.pi / M))  
    elif name in ("16QAM", "64QAM", "256QAM"):  
        M = int(name[:-3])  
        m = int(round(math.sqrt(M)))  
        lv = np.arange(-(m - 1), m, 2)  
        I, Q = np.meshgrid(lv, lv)  
        pts = (I + 1j * Q).ravel()  
    elif name == "32QAM":                       # 6x6 grid minus 4 corners  
        lv = np.arange(-5, 6, 2)  
        I, Q = np.meshgrid(lv, lv)  
        keep = ~((np.abs(I) == 5) & (np.abs(Q) == 5))  
        pts = (I + 1j * Q)[keep]  
    elif name == "128QAM":                      # 12x12 grid minus 2x2 corners  
        lv = np.arange(-11, 12, 2)  
        I, Q = np.meshgrid(lv, lv)  
        keep = ~((np.abs(I) >= 9) & (np.abs(Q) >= 9))  
        pts = (I + 1j * Q)[keep]  
    else:  
        raise ValueError(name)  
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))  


CONSTELLATIONS = [make_constellation(m) for m in MODS]  
assert [len(c) for c in CONSTELLATIONS] == [2, 4, 8, 16, 32, 64, 128, 256]  


def rrc_taps(beta: float, span: int, sps: int) -> np.ndarray:  
    """Root-raised-cosine filter, unit energy."""  
    N = span * sps  
    t = np.arange(-N / 2, N / 2 + 1) / sps  
    h = np.zeros_like(t)  
    for i, ti in enumerate(t):  
        if abs(ti) < 1e-10:  
            h[i] = 1 - beta + 4 * beta / np.pi  
        elif beta > 0 and abs(abs(ti) - 1 / (4 * beta)) < 1e-10:  
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))  
                                       + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))  
        else:  
            h[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) / \
                   (np.pi * ti * (1 - (4 * beta * ti) ** 2))  
    return h / np.sqrt(np.sum(h ** 2))  


def batch_conv_full(x: np.ndarray, h: np.ndarray) -> np.ndarray:  
    """FFT based 'full' convolution of every row of x (B,N) with h (T,)."""  
    n = x.shape[1] + len(h) - 1  
    nfft = 1 << (n - 1).bit_length()  
    return np.fft.ifft(np.fft.fft(x, nfft, axis=1) * np.fft.fft(h, nfft), axis=1)[:, :n]  


def shape_symbols(syms: np.ndarray, taps: np.ndarray, sps: int) -> np.ndarray:  
    B, nsym = syms.shape  
    up = np.zeros((B, nsym * sps), dtype=complex)  
    up[:, ::sps] = syms  
    full = batch_conv_full(up, taps)  
    T = len(taps)  
    return full[:, T - 1: nsym * sps]                   # fully-overlapped region  


def multipath_response(B: int, nfft: int, cfg: Config, rng) -> np.ndarray:  
    """Eq. (3): H(f) = sum_i g_i A(f,d_i) exp(-j2pi f tau_i), energy-normalised."""  
    f = np.fft.fftfreq(nfft, d=1.0 / cfg.fs)  
    tau = rng.exponential(cfg.tau_mean_us * 1e-6, size=(B, cfg.n_paths))  
    tau[:, 0] = 0.0                                      # direct path  
    g = rng.uniform(*cfg.gain_range, size=(B, cfg.n_paths))  
    g[:, 0] = 1.0  
    d = tau * cfg.v_prop  
    fabs = np.abs(cfg.f_carrier + f)  
    att = np.exp(-(cfg.a0 + cfg.a1 * fabs ** cfg.k_exp)[None, None, :] * d[:, :, None])  
    H = (g[:, :, None] * att * np.exp(-2j * np.pi * f[None, None, :] * tau[:, :, None])).sum(1)  
    H /= np.sqrt(np.mean(np.abs(H) ** 2, axis=1, keepdims=True))  
    return H  


def apply_channel(x: np.ndarray, H: np.ndarray) -> np.ndarray:  
    nfft = H.shape[1]  
    return np.fft.ifft(np.fft.fft(x, nfft, axis=1) * H, axis=1)[:, :x.shape[1]]  


def middleton_class_a(shape, sigma2, cfg: Config, rng) -> np.ndarray:  
    """Eq. (4): Poisson mixture of Gaussians; total power = sigma2."""  
    m = np.minimum(rng.poisson(cfg.class_a_A, size=shape), cfg.class_a_mmax)  
    var_m = sigma2 * (m / cfg.class_a_A + cfg.class_a_gamma) / (1.0 + cfg.class_a_gamma)  
    return np.sqrt(var_m / 2.0) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))  


def generate_batch(mod_idx: int, snr_db: float, B: int, cfg: Config, rng, taps) -> np.ndarray:  
    """Returns B received frames (B, n_samples) complex64 for one modulation / SNR."""  
    N, sps = cfg.n_samples, cfg.sps  
    nsym = N // sps + cfg.rrc_span + 4  

    def tx_stream(mods):  
        syms = np.empty((B, nsym), dtype=complex)  
        for mi in np.unique(mods):  
            sel = mods == mi  
            pts = CONSTELLATIONS[mi]  
            syms[sel] = pts[rng.integers(0, len(pts), size=(sel.sum(), nsym))]  
        return shape_symbols(syms, taps, sps)  

    x = tx_stream(np.full(B, mod_idx))  
    Ltot = x.shape[1]  
    nfft = 1 << (Ltot + 64 - 1).bit_length()  

    # ---- multipath (+ optional MIMO crosstalk from an independent stream) ----  
    y = apply_channel(x, multipath_response(B, nfft, cfg, rng))  
    if cfg.crosstalk_db is not None:  
        x2 = tx_stream(rng.integers(0, len(MODS), size=B))  
        y += 10 ** (cfg.crosstalk_db / 20) * apply_channel(x2, multipath_response(B, nfft, cfg, rng))  
    y /= np.sqrt(np.mean(np.abs(y) ** 2, axis=1, keepdims=True))  

    # ---- CFO + initial phase ----  
    n = np.arange(Ltot)[None, :]  
    eps = rng.uniform(-cfg.cfo_max, cfg.cfo_max, size=(B, 1))  
    phi0 = rng.uniform(0, 2 * np.pi, size=(B, 1))  
    y = y * np.exp(1j * (2 * np.pi * eps * n / sps + phi0))  

    # ---- sampling clock offset (linear interpolation resampling) ----  
    delta = rng.uniform(-cfg.sco_ppm, cfg.sco_ppm, size=(B, 1)) * 1e-6  
    t = np.arange(N)[None, :] * (1.0 + delta)  
    i0 = np.floor(t).astype(int)  
    frac = t - i0  
    i1 = np.minimum(i0 + 1, Ltot - 1)  
    y = np.take_along_axis(y, i0, 1) * (1 - frac) + np.take_along_axis(y, i1, 1) * frac  

    # ---- Class-A impulsive noise, power set by SNR (signal power = 1) ----  
    sigma2 = 10 ** (-snr_db / 10.0)  
    y = y + middleton_class_a(y.shape, sigma2, cfg, rng)  

    # ---- receiver matched filter ----  
    if cfg.rx_matched_filter:  
        T = len(taps)  
        y = batch_conv_full(y, taps)[:, (T - 1) // 2:(T - 1) // 2 + N]  
    return y.astype(np.complex64)  


# ----------------------------------------------------------------------------- #  
#                    2.  Higher-order cumulants (Stage-1 features)              #  
# ----------------------------------------------------------------------------- #  
def set_partitions(elements):  
    if not elements:  
        yield []  
        return  
    first, rest = elements[0], elements[1:]  
    for smaller in set_partitions(rest):  
        for i in range(len(smaller)):  
            yield smaller[:i] + [[first] + smaller[i]] + smaller[i + 1:]  
        yield [[first]] + smaller  


@lru_cache(maxsize=None)  
def cumulant_table(n_r: int, n_c: int):  
    """  
    Exact moment-to-cumulant expansion for Cum(r,...,r, r*,...,r*) using the  
    set-partition formula: sum_pi (-1)^(|pi|-1)(|pi|-1)! prod_B E[prod_{i in B} z_i].  
    Returns {multiset of block types (a,b): coefficient}. Terms containing  
    first-order moments are dropped (zero-mean process).  
    """  
    labels = [0] * n_r + [1] * n_c  
    table = {}  
    for part in set_partitions(list(range(n_r + n_c))):  
        if any(len(blk) == 1 for blk in part):  
            continue  
        k = len(part)  
        coef = (-1) ** (k - 1) * math.factorial(k - 1)  
        key = tuple(sorted((sum(1 for e in blk if labels[e] == 0),  
                            sum(1 for e in blk if labels[e] == 1)) for blk in part))  
        table[key] = table.get(key, 0) + coef  
    return {k: v for k, v in table.items() if v != 0}  


def compute_hoc(iq: np.ndarray, chunk: int = 512) -> np.ndarray:  
    """  
    12-dim HOC vector (Eq. 10), normalised by C21^((p+q)/2) (Eq. 11).  
    We use log1p(|C~pq|): magnitude makes the feature phase-rotation invariant,  
    log compresses the dynamic range (|C~80| of BPSK = 272).  
    """  
    out = np.zeros((len(iq), len(HOC_ORDERS)), dtype=np.float32)  
    for s in range(0, len(iq), chunk):  
        r = iq[s:s + chunk].astype(np.complex128)  
        r = r - r.mean(1, keepdims=True)  
        rc = np.conj(r)  
        rp, cp = [np.ones_like(r)], [np.ones_like(r)]  
        for _ in range(8):  
            rp.append(rp[-1] * r)  
            cp.append(cp[-1] * rc)  
        M = {(a, b): (rp[a] * cp[b]).mean(1) for a in range(9) for b in range(9 - a)}  
        c21 = M[(1, 1)].real  
        for j, (p, q) in enumerate(HOC_ORDERS):  
            c = np.zeros(len(r), dtype=complex)  
            for key, coef in cumulant_table(p - q, q).items():  
                term = coef * np.ones(len(r), dtype=complex)  
                for ab in key:  
                    term = term * M[ab]  
                c += term  
            out[s:s + chunk, j] = np.log1p(np.abs(c / c21 ** (p / 2.0)))  
    return out  


# ----------------------------------------------------------------------------- #  
#                    3.  3D constellation matrix (Stage-2 features)             #  
# ----------------------------------------------------------------------------- #  
def constellation_matrix(iq: torch.Tensor, L: int = 32, R: float = 2.5) -> torch.Tensor:  
    """  
    iq: (B, 2, N) float tensor  ->  (B, L, 3L) sequence of rows.  
    Channel 1: normalised histogram density; channels 2/3: mean dI, dQ per bin  
    (Section 3.2.1). Row i (I-axis bin) is one time-step for the LNN.  
    """  
    B, _, N = iq.shape  
    I, Q = iq[:, 0], iq[:, 1]  
    p = (I ** 2 + Q ** 2).mean(1, keepdim=True).sqrt().clamp_min(1e-9)  
    I, Q = I / p, Q / p  
    bi = ((I + R) / (2 * R) * L).floor().long().clamp(0, L - 1)  
    bj = ((Q + R) / (2 * R) * L).floor().long().clamp(0, L - 1)  
    dev = iq.device  
    flat = (torch.arange(B, device=dev).unsqueeze(1) * (L * L) + bi * L + bj).reshape(-1)  

    dI = torch.zeros_like(I); dI[:, 1:] = I[:, 1:] - I[:, :-1]  
    dQ = torch.zeros_like(Q); dQ[:, 1:] = Q[:, 1:] - Q[:, :-1]  

    cnt = torch.zeros(B * L * L, device=dev).scatter_add_(0, flat, torch.ones(B * N, device=dev))  
    sI = torch.zeros(B * L * L, device=dev).scatter_add_(0, flat, dI.reshape(-1))  
    sQ = torch.zeros(B * L * L, device=dev).scatter_add_(0, flat, dQ.reshape(-1))  
    denom = cnt.clamp_min(1.0)  
    X = torch.stack([cnt / N * L, sI / denom, sQ / denom], dim=-1)   # density scaled by L for range  
    return X.reshape(B, L, L * 3)  


# ----------------------------------------------------------------------------- #  
#                         4.  Stage-1: HOC-Transformer                          #  
# ----------------------------------------------------------------------------- #  
class HOCTransformer(nn.Module):  
    def __init__(self, cfg: Config, n_feat: int = 12, n_classes: int = 3):  
        super().__init__()  
        self.register_buffer("mu", torch.zeros(n_feat))  
        self.register_buffer("sd", torch.ones(n_feat))  
        self.embed = nn.Linear(1, cfg.d_model)                          # Eq. (12): scalar -> e_i  
        self.pos = nn.Parameter(torch.randn(1, n_feat, cfg.d_model) * 0.02)  
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,  
                                           activation="gelu", batch_first=True)  
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers)       # Eq. (13)-(14)  
        self.head = nn.Sequential(nn.LayerNorm(cfg.d_model),  
                                  nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Dropout(cfg.dropout),  
                                  nn.Linear(cfg.d_ff, n_classes))       # Eq. (15)  

    def set_normalizer(self, hoc_train: torch.Tensor):  
        self.mu.copy_(hoc_train.mean(0))  
        sd = hoc_train.std(0)  
        self.sd.copy_(torch.where(sd > 1e-6, sd, torch.ones_like(sd)))  

    def forward(self, c):                      # c: (B, 12)  
        x = (c - self.mu) / self.sd  
        z = self.encoder(self.embed(x.unsqueeze(-1)) + self.pos)  
        return self.head(z.mean(1))            # mean pooling  


# ----------------------------------------------------------------------------- #  
#                  5.  Stage-2: Liquid Time-Constant (LTC) experts              #  
# ----------------------------------------------------------------------------- #  
class LTCLayer(nn.Module):  
    """  
    Liquid time-constant neuron layer, Eq. (22):  
        dh/dt = -[w_sys + f(x,h)] * h + f(x,h) * A,   f = tanh(W_in x + W_rec h + b)  
    Integrated with fixed-step RK4 between consecutive input rows  
    (the paper uses adaptive Dormand-Prince with dt in [0.01, 0.1]).  
    With w_sys >= 1 and |f| < 1 the effective time constant stays positive.  
    """  
    def __init__(self, in_dim: int, hidden: int, w_sys: float = 1.0, dt: float = 0.1, steps: int = 5):  
        super().__init__()  
        self.hidden, self.dt, self.steps = hidden, dt, steps  
        self.W_in = nn.Linear(in_dim, hidden)  
        self.W_rec = nn.Linear(hidden, hidden, bias=False)  
        self.A = nn.Parameter(torch.randn(hidden) * 0.1)               # learnable equilibrium  
        self.w_sys_base = w_sys  
        self.w_sys_extra = nn.Parameter(torch.zeros(hidden))           # w_sys = base + softplus(.)  
        nn.init.orthogonal_(self.W_rec.weight, gain=0.5)  

    def dhdt(self, x, h):  
        f = torch.tanh(self.W_in(x) + self.W_rec(h))  
        w_sys = self.w_sys_base + F.softplus(self.w_sys_extra)  
        return -(w_sys + f) * h + f * self.A  

    def rk4(self, x, h, dt):  
        k1 = self.dhdt(x, h)  
        k2 = self.dhdt(x, h + 0.5 * dt * k1)  
        k3 = self.dhdt(x, h + 0.5 * dt * k2)  
        k4 = self.dhdt(x, h + dt * k3)  
        return h + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)  

    def forward(self, xs):                     # xs: (B, T, D)  
        B, T, _ = xs.shape  
        h = xs.new_zeros(B, self.hidden)       # zero-initialised state  
        outs = []  
        for t in range(T):  
            x = xs[:, t]  
            for _ in range(self.steps):  
                h = self.rk4(x, h, self.dt)  
            outs.append(h)  
        return torch.stack(outs, 1)  


class LNNExpert(nn.Module):  
    def __init__(self, cfg: Config, n_classes: int):  
        super().__init__()  
        in_dim = cfg.cm_size * 3  
        self.norm_in = nn.LayerNorm(in_dim)  
        dims = [in_dim] + [cfg.lnn_hidden] * cfg.lnn_layers  
        self.layers = nn.ModuleList(LTCLayer(dims[i], dims[i + 1], cfg.w_sys, cfg.ode_dt, cfg.ode_steps)  
                                    for i in range(cfg.lnn_layers))  
        self.head = nn.Sequential(nn.LayerNorm(cfg.lnn_hidden), nn.Linear(cfg.lnn_hidden, n_classes))  # Eq. (18)  

    def forward(self, cm):                     # cm: (B, L, 3L)  
        x = self.norm_in(cm)  
        for layer in self.layers:  
            x = layer(x)  
        return self.head(x[:, -1])             # final hidden state h_L  


# ----------------------------------------------------------------------------- #  
#                                6.  Training                                   #  
# ----------------------------------------------------------------------------- #  
def train_model(model, make_xy, train_loader, val_loader, epochs, cfg: Config, device, name):  
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8,  
                           weight_decay=cfg.weight_decay)  
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=cfg.lr_min)  
    best_loss, best_state, bad = float("inf"), None, 0  
    for ep in range(1, epochs + 1):  
        model.train()  
        t0, tr_loss, n_tr = time.time(), 0.0, 0  
        for batch in train_loader:  
            x, y = make_xy([b.to(device) for b in batch])  
            loss = F.cross_entropy(model(x), y)  
            opt.zero_grad(set_to_none=True)  
            loss.backward()  
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)  
            opt.step()  
            tr_loss += loss.item() * len(y); n_tr += len(y)  
        sched.step()  
        val_loss, val_acc = evaluate_loss_acc(model, make_xy, val_loader, device)  
        print(f"[{name}] ep {ep:3d}/{epochs} | train {tr_loss / n_tr:.4f} | val {val_loss:.4f} "  
              f"| val acc {val_acc * 100:5.2f}% | {time.time() - t0:.1f}s")  
        if val_loss < best_loss - 1e-5:  
            best_loss, bad = val_loss, 0  
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}  
        else:  
            bad += 1  
            if bad >= cfg.patience:  
                print(f"[{name}] early stopping (patience {cfg.patience})")  
                break  
    model.load_state_dict(best_state)  
    return model  


@torch.no_grad()  
def evaluate_loss_acc(model, make_xy, loader, device):  
    model.eval()  
    tot_loss, correct, n = 0.0, 0, 0  
    for batch in loader:  
        x, y = make_xy([b.to(device) for b in batch])  
        logits = model(x)  
        tot_loss += F.cross_entropy(logits, y, reduction="sum").item()  
        correct += (logits.argmax(1) == y).sum().item()  
        n += len(y)  
    return tot_loss / n, correct / n  


# ----------------------------------------------------------------------------- #  
#                          7.  Hierarchical inference                           #  
# ----------------------------------------------------------------------------- #  
@torch.no_grad()  
def predict_htlnn(stage1, experts, iq, hoc, fam, cfg: Config, device, bs=512):  
    """Returns (family prediction, end-to-end prediction, oracle-gate prediction)."""  
    stage1.eval(); [e.eval() for e in experts]  
    fam_pred, pred, oracle = [], [], []  
    for s in range(0, len(iq), bs):  
        iq_b, hoc_b, fam_b = iq[s:s + bs].to(device), hoc[s:s + bs].to(device), fam[s:s + bs].to(device)  
        fp = stage1(hoc_b).argmax(1)                                    # hard gate, Sec. 3.2.2 (a)  
        cm = constellation_matrix(iq_b, cfg.cm_size, cfg.cm_range)  
        p, o = torch.zeros_like(fp), torch.zeros_like(fp)  
        for k, members in FAMILIES.items():  
            members_t = torch.tensor(members, device=device)  
            m = fp == k  
            if m.any():  
                p[m] = members_t[experts[k](cm[m]).argmax(1)]  
            m2 = fam_b == k  
            if m2.any():  
                o[m2] = members_t[experts[k](cm[m2]).argmax(1)]  
        fam_pred.append(fp.cpu()); pred.append(p.cpu()); oracle.append(o.cpu())  
    return torch.cat(fam_pred).numpy(), torch.cat(pred).numpy(), torch.cat(oracle).numpy()  


def confusion_matrix(y, p, n):  
    cm = np.zeros((n, n), dtype=int)  
    np.add.at(cm, (y, p), 1)  
    return cm  


def per_class_metrics(cm):  
    tp = np.diag(cm).astype(float)  
    prec = tp / np.maximum(cm.sum(0), 1)  
    rec = tp / np.maximum(cm.sum(1), 1)  
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)  
    return prec, rec, f1  


# ----------------------------------------------------------------------------- #  
#                              8.  Dataset assembly                             #  
# ----------------------------------------------------------------------------- #  
def build_dataset(cfg: Config, seed: int, cache: str):  
    if cache and os.path.exists(cache):  
        print(f"Loading cached dataset: {cache}")  
        d = np.load(cache)  
        return {k: d[k] for k in d.files}  
    rng = np.random.default_rng(seed)  
    taps = rrc_taps(cfg.rrc_beta, cfg.rrc_span, cfg.sps)  
    iq, y, snr = [], [], []  
    t0 = time.time()  
    for s_db in cfg.snr_list:  
        for mi in range(len(MODS)):  
            left = cfg.frames_per_mod_snr  
            while left > 0:  
                B = min(256, left)  
                iq.append(generate_batch(mi, s_db, B, cfg, rng, taps))  
                y.append(np.full(B, mi)); snr.append(np.full(B, s_db, dtype=np.float32))  
                left -= B  
        print(f"  SNR {s_db:+3d} dB generated ({time.time() - t0:.0f}s)")  
    iq = np.concatenate(iq); y = np.concatenate(y); snr = np.concatenate(snr)  
    print("Computing 12-dim HOC features ...")  
    hoc = compute_hoc(iq)  
    data = dict(iq=iq, y=y, snr=snr, hoc=hoc)  
    if cache:  
        np.savez(cache, **data)  
        print(f"Dataset cached to {cache}")  
    return data  


def stratified_split(y, snr, split, seed):  
    rng = np.random.default_rng(seed + 1)  
    tr, va, te = [], [], []  
    for mi in np.unique(y):  
        for s in np.unique(snr):  
            idx = np.where((y == mi) & (snr == s))[0]  
            rng.shuffle(idx)  
            n_tr = int(round(split[0] * len(idx))); n_va = int(round(split[1] * len(idx)))  
            tr.append(idx[:n_tr]); va.append(idx[n_tr:n_tr + n_va]); te.append(idx[n_tr + n_va:])  
    return np.concatenate(tr), np.concatenate(va), np.concatenate(te)  


# ----------------------------------------------------------------------------- #  
#                                    9.  Main                                   #  
# ----------------------------------------------------------------------------- #  
def main():  
    ap = argparse.ArgumentParser(description="H-TLNN standalone implementation")  
    ap.add_argument("--quick", action="store_true", help="small demo: 60 frames/(mod,SNR), 20/15 epochs")  
    ap.add_argument("--full", action="store_true", help="paper scale: 1200 frames/(mod,SNR), 100/50 epochs")  
    ap.add_argument("--frames", type=int, default=None, help="frames per (modulation, SNR)")  
    ap.add_argument("--epochs1", type=int, default=None)  
    ap.add_argument("--epochs2", type=int, default=None)  
    ap.add_argument("--seed", type=int, default=0)  
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")  
    ap.add_argument("--cache", type=str, default="plc_amr_dataset.npz")  
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints")  
    ap.add_argument("--cm_snr", type=float, default=4.0, help="SNR for the confusion matrix")  
    ap.add_argument("--no_plot", action="store_true")  
    args = ap.parse_args()  

    cfg = Config()  
    if args.quick:  
        cfg.frames_per_mod_snr, cfg.epochs_stage1, cfg.epochs_stage2 = 60,     cfg = Config()
    if args.quick:
        cfg.frames_per_mod_snr, cfg.epochs_stage1, cfg.epochs_stage2 = 60, 20, 15
        cfg.patience = 8
        if args.cache == "plc_amr_dataset.npz":
            args.cache = "plc_amr_dataset_quick.npz"
    if args.full:
        cfg.frames_per_mod_snr, cfg.epochs_stage1, cfg.epochs_stage2 = 1200, 100, 50
    if args.frames is not None:
        cfg.frames_per_mod_snr = args.frames
    if args.epochs1 is not None:
        cfg.epochs_stage1 = args.epochs1
    if args.epochs2 is not None:
        cfg.epochs_stage2 = args.epochs2

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"Device: {device} | frames/(mod,SNR): {cfg.frames_per_mod_snr} | "
          f"total frames: {cfg.frames_per_mod_snr * len(MODS) * len(cfg.snr_list)}")

    # ------------------------------------------------------------------ data --
    data = build_dataset(cfg, args.seed, args.cache)
    iq_np, y_np, snr_np, hoc_np = data["iq"], data["y"], data["snr"], data["hoc"]
    tr_idx, va_idx, te_idx = stratified_split(y_np, snr_np, cfg.split, args.seed)
    print(f"Split: train {len(tr_idx)} | val {len(va_idx)} | test {len(te_idx)}")

    # complex -> (B, 2, N) float tensor
    iq_t = torch.from_numpy(np.stack([iq_np.real, iq_np.imag], axis=1).astype(np.float32))
    hoc_t = torch.from_numpy(hoc_np)
    y_t = torch.from_numpy(y_np.astype(np.int64))
    fam_t = torch.tensor([FAMILY_OF[int(m)] for m in y_np], dtype=torch.int64)
    loc_t = torch.tensor([LOCAL_IDX[int(m)] for m in y_np], dtype=torch.int64)
    snr_t = torch.from_numpy(snr_np)

    def subset(idx, *tensors):
        idx = torch.from_numpy(idx)
        return [t[idx] for t in tensors]

    # ======================================================================== #
    #  Phase 1: HOC-Transformer (coarse family classifier)          [Sec. 3.3.1]
    # ======================================================================== #
    print("\n===== Phase 1: training HOC-Transformer (3 families) =====")
    hoc_tr, fam_tr = subset(tr_idx, hoc_t, fam_t)
    hoc_va, fam_va = subset(va_idx, hoc_t, fam_t)
    stage1 = HOCTransformer(cfg).to(device)
    stage1.set_normalizer(hoc_tr.to(device))
    s1_train = DataLoader(TensorDataset(hoc_tr, fam_tr), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    s1_val = DataLoader(TensorDataset(hoc_va, fam_va), batch_size=1024)
    stage1 = train_model(stage1, lambda b: (b[0], b[1]), s1_train, s1_val,
                         cfg.epochs_stage1, cfg, device, "Stage-1")
    torch.save(stage1.state_dict(), os.path.join(args.ckpt_dir, "stage1_hoc_transformer.pt"))

    # ======================================================================== #
    #  Phase 2: three LNN experts, trained on ground-truth families [Sec. 3.3.2]
    # ======================================================================== #
    experts = []
    for k, members in FAMILIES.items():
        print(f"\n===== Phase 2: training LNN expert #{k} ({FAMILY_NAMES[k]}: "
              f"{[MODS[m] for m in members]}) =====")
        m_tr = tr_idx[fam_t[tr_idx].numpy() == k]
        m_va = va_idx[fam_t[va_idx].numpy() == k]
        iq_k_tr, loc_k_tr = subset(m_tr, iq_t, loc_t)
        iq_k_va, loc_k_va = subset(m_va, iq_t, loc_t)
        expert = LNNExpert(cfg, n_classes=len(members)).to(device)
        # constellation matrix is computed on-the-fly on the device (cheap scatter ops)
        make_xy = lambda b: (constellation_matrix(b[0], cfg.cm_size, cfg.cm_range), b[1])
        e_train = DataLoader(TensorDataset(iq_k_tr, loc_k_tr), batch_size=cfg.batch_size, shuffle=True)
        e_val = DataLoader(TensorDataset(iq_k_va, loc_k_va), batch_size=512)
        expert = train_model(expert, make_xy, e_train, e_val, cfg.epochs_stage2, cfg, device,
                             f"Expert-{FAMILY_NAMES[k]}")
        torch.save(expert.state_dict(), os.path.join(args.ckpt_dir, f"stage2_lnn_{FAMILY_NAMES[k]}.pt"))
        experts.append(expert)

    # ======================================================================== #
    #  Evaluation on the held-out test set                       [Sec. 4.2/4.3]
    # ======================================================================== #
    print("\n===== Evaluation (test set) =====")
    iq_te, hoc_te, y_te, fam_te, snr_te = subset(te_idx, iq_t, hoc_t, y_t, fam_t, snr_t)
    fam_pred, pred, oracle = predict_htlnn(stage1, experts, iq_te, hoc_te, fam_te, cfg, device)
    y_te, fam_te, snr_te = y_te.numpy(), fam_te.numpy(), snr_te.numpy()

    # ---- overall & per-SNR accuracy (Fig. Acc-vs-SNR) ----
    snr_axis = np.array(sorted(np.unique(snr_te)))
    acc_stage1, acc_e2e, acc_oracle = [], [], []
    print(f"\n{'SNR':>5} | {'Stage-1 fam.':>12} | {'End-to-end':>10} | {'Oracle gate':>11} | {'Routing loss':>12}")
    for s in snr_axis:
        m = snr_te == s
        a1 = (fam_pred[m] == fam_te[m]).mean()
        a2 = (pred[m] == y_te[m]).mean()
        a3 = (oracle[m] == y_te[m]).mean()
        acc_stage1.append(a1); acc_e2e.append(a2); acc_oracle.append(a3)
        print(f"{s:+5.0f} | {a1 * 100:11.2f}% | {a2 * 100:9.2f}% | {a3 * 100:10.2f}% | {(a3 - a2) * 100:11.2f}%")
    print(f"{'ALL':>5} | {(fam_pred == fam_te).mean() * 100:11.2f}% | {(pred == y_te).mean() * 100:9.2f}% "
          f"| {(oracle == y_te).mean() * 100:10.2f}% |")

    # ---- routing-error decomposition (Table 4/5 style) ----
    print("\n--- Error decomposition (all SNRs) ---")
    wrong = pred != y_te
    route_wrong = fam_pred != fam_te
    n_wrong = wrong.sum()
    print(f"total errors            : {n_wrong}  ({wrong.mean() * 100:.2f}%)")
    print(f"  caused by mis-routing : {(wrong & route_wrong).sum()}  "
          f"({(wrong & route_wrong).sum() / max(n_wrong, 1) * 100:.1f}% of errors)")
    print(f"  intra-family errors   : {(wrong & ~route_wrong).sum()}  "
          f"({(wrong & ~route_wrong).sum() / max(n_wrong, 1) * 100:.1f}% of errors)")
    fam_cm = confusion_matrix(fam_te, fam_pred, len(FAMILIES))
    print("Family confusion matrix (rows = true, cols = pred; PSK / Sq-QAM / Cr-QAM):")
    print(fam_cm)

    # ---- per-class precision / recall / F1 ----
    cm_all = confusion_matrix(y_te, pred, len(MODS))
    prec, rec, f1 = per_class_metrics(cm_all)
    print(f"\n{'Class':>8} | {'Prec':>6} | {'Rec':>6} | {'F1':>6}")
    for i, name in enumerate(MODS):
        print(f"{name:>8} | {prec[i] * 100:5.1f}% | {rec[i] * 100:5.1f}% | {f1[i] * 100:5.1f}%")
    print(f"{'macro':>8} | {prec.mean() * 100:5.1f}% | {rec.mean() * 100:5.1f}% | {f1.mean() * 100:5.1f}%")

    # ---- 8x8 confusion matrix at a chosen SNR ----
    m = snr_te == args.cm_snr
    if m.any():
        cm_snr = confusion_matrix(y_te[m], pred[m], len(MODS))
        print(f"\nConfusion matrix @ {args.cm_snr:+.0f} dB (rows = true, cols = pred):")
        print("        " + " ".join(f"{n:>7}" for n in MODS))
        for i, name in enumerate(MODS):
            print(f"{name:>7} " + " ".join(f"{v:7d}" for v in cm_snr[i]))

    np.savez(os.path.join(args.ckpt_dir, "test_results.npz"),
             snr=snr_axis, acc_stage1=acc_stage1, acc_e2e=acc_e2e, acc_oracle=acc_oracle,
             cm_all=cm_all, fam_cm=fam_cm, y=y_te, pred=pred, fam_pred=fam_pred, snr_te=snr_te)

    # ---- plots ----
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
            ax[0].plot(snr_axis, np.array(acc_e2e) * 100, "o-", label="H-TLNN (end-to-end)")
            ax[0].plot(snr_axis, np.array(acc_oracle) * 100, "s--", label="Stage-2 w/ oracle gate")
            ax[0].plot(snr_axis, np.array(acc_stage1) * 100, "^:", label="Stage-1 family acc.")
            ax[0].set_xlabel("SNR (dB)"); ax[0].set_ylabel("Accuracy (%)")
            ax[0].set_title("Accuracy vs. SNR"); ax[0].grid(alpha=.3); ax[0].legend()

            cm_plot = cm_snr if m.any() else cm_all
            cm_norm = cm_plot / np.maximum(cm_plot.sum(1, keepdims=True), 1)
            im = ax[1].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
            ax[1].set_xticks(range(len(MODS))); ax[1].set_yticks(range(len(MODS)))
            ax[1].set_xticklabels(MODS, rotation=45, ha="right"); ax[1].set_yticklabels(MODS)
            for i in range(len(MODS)):
                for j in range(len(MODS)):
                    ax[1].text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                               fontsize=7, color="white" if cm_norm[i, j] > 0.5 else "black")
            ax[1].set_title(f"Confusion matrix @ {args.cm_snr:+.0f} dB" if m.any() else "Confusion matrix (all)")
            ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("True")
            fig.colorbar(im, ax=ax[1], fraction=0.046)
            fig.tight_layout()
            out = os.path.join(args.ckpt_dir, "results.png")
            fig.savefig(out, dpi=150)
            print(f"\nFigure saved to {out}")
        except ImportError:
            print("matplotlib not available; skipping plots.")


if __name__ == "__main__":
    main()