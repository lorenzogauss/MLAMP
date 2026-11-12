"""
===============================================================================
PIPELINE DI ADDESTRAMENTO LSTM & ESPORTAZIONE JSON PER JUCE C++
===============================================================================
"""

import os
import json
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import numpy as np
from scipy.signal import correlate

# Configurazione Device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"--> Utilizzo Device: {DEVICE}")

# =============================================================================
# 1. ARCHITETTURA MODELLO
# =============================================================================

class NeuralAmpLSTM(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, x, hx=None):
        x = x.transpose(1, 2)
        if hx is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            hx = (h0, c0)

        lstm_out, hx = self.lstm(x, hx)
        out = self.tanh(self.fc(lstm_out))
        out = out.transpose(1, 2)
        return out, hx

# =============================================================================
# 2. LOSS FUNCTIONS (STFT + TIME L1)
# =============================================================================

class SingleSTFTLoss(nn.Module):
    def __init__(self, fft_size, hop_size, win_length, eps=1e-4):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.eps = eps
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, x, y):
        x_sig, y_sig = x.squeeze(1), y.squeeze(1)
        x_stft = torch.stft(x_sig, self.fft_size, self.hop_size, self.win_length, self.window, return_complex=True)
        y_stft = torch.stft(y_sig, self.fft_size, self.hop_size, self.win_length, self.window, return_complex=True)

        x_mag = torch.abs(x_stft) + self.eps
        y_mag = torch.abs(y_stft) + self.eps

        sc_loss = torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + self.eps)
        log_loss = F.l1_loss(torch.log(x_mag), torch.log(y_mag))
        return sc_loss, log_loss

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[128, 256, 64], win_lengths=[1024, 2048, 512]):
        super().__init__()
        self.loss_funcs = nn.ModuleList([
            SingleSTFTLoss(f, h, w) for f, h, w in zip(fft_sizes, hop_sizes, win_lengths)
        ])

    def forward(self, pred, target):
        total_sc, total_log = 0.0, 0.0
        for f in self.loss_funcs:
            sc, log = f(pred, target)
            total_sc += sc
            total_log += log
        return total_sc / len(self.loss_funcs), total_log / len(self.loss_funcs)

class CombinedSTFTTimeLoss(nn.Module):
    def __init__(self, stft_weight=1.0, time_weight=0.2):
        super().__init__()
        self.stft_loss = MultiResolutionSTFTLoss()
        self.time_loss = nn.L1Loss()
        self.stft_weight = stft_weight
        self.time_weight = time_weight

    def forward(self, pred, target):
        sc_loss, log_loss = self.stft_loss(pred, target)
        loss_time = self.time_loss(pred, target)
        return (self.stft_weight * (sc_loss + log_loss)) + (self.time_weight * loss_time)

# =============================================================================
# 3. PREPROCESSING E ALLINEAMENTO AUDIO
# =============================================================================

def align_signals(dry, wet, sr=44100):
    max_check = int(0.5 * sr)
    d = dry.numpy().flatten()[:max_check]
    w = wet.numpy().flatten()[:max_check]

    if len(d) == 0 or len(w) == 0:
        return dry, wet

    corr = correlate(w, d, mode='full')
    lag = corr.argmax() - (len(d) - 1)

    if lag > 0:
        wet = wet[:, lag:]
    elif lag < 0:
        dry = dry[:, -lag:]

    min_len = min(dry.shape[-1], wet.shape[-1])
    return dry[:, :min_len], wet[:, :min_len]

def load_audio_pair(dry_path, wet_path, segment_length=4096):
    dry_np, sr_dry = sf.read(dry_path, dtype='float32')
    wet_np, sr_wet = sf.read(wet_path, dtype='float32')

    dry = torch.from_numpy(dry_np).t().unsqueeze(0) if dry_np.ndim == 1 else torch.from_numpy(dry_np).t()
    wet = torch.from_numpy(wet_np).t().unsqueeze(0) if wet_np.ndim == 1 else torch.from_numpy(wet_np).t()

    if sr_dry != sr_wet:
        wet = torchaudio.transforms.Resample(sr_wet, sr_dry)(wet)

    dry, wet = dry[0:1, :], wet[0:1, :]
    dry, wet = align_signals(dry, wet, sr=sr_dry)

    # Normalizzazione ampiezza
    if torch.max(torch.abs(dry)) > 0: dry = (dry / torch.max(torch.abs(dry))) * 0.95
    if torch.max(torch.abs(wet)) > 0: wet = (wet / torch.max(torch.abs(wet))) * 0.95

    num_segments = min(dry.shape[-1], wet.shape[-1]) // segment_length
    dry_segs, wet_segs = [], []

    for i in range(num_segments):
        d_s = dry[:, i*segment_length : (i+1)*segment_length]
        w_s = wet[:, i*segment_length : (i+1)*segment_length]
        if torch.max(torch.abs(w_s)) > 0.01:
            dry_segs.append(d_s)
            wet_segs.append(w_s)

    return torch.stack(dry_segs), torch.stack(wet_segs)

# =============================================================================
# 4. EXPORT UNICO JSON PER C++ / JUCE
# =============================================================================

def export_json(model, output_path):
    # Salva il file nella Home Directory dell'utente (~/model_weights.json)
    home_dir = os.path.expanduser("~")
    full_path = os.path.join(home_dir, output_path)

    weights_dict = {}
    for name, param in model.named_parameters():
        weights_dict[name] = param.detach().cpu().numpy().tolist()

    with open(full_path, "w") as f:
        json.dump(weights_dict, f, indent=4)
        
    print(f"\n[OK] Pesi salvati con successo in: {full_path}")

# =============================================================================
# 5. PIPELINE PRINCIPALE
# =============================================================================

def main():
    DRY_FILE = "dry_sweep.wav"
    WET_FILE = "wet_sweep.wav"

    EPOCHS = 40
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    SEGMENT_LENGTH = 4096
    HIDDEN_SIZE = 64  # Se la CPU in Logic va in overload, imposta a 32 ed esegui nuovamente

    if not (os.path.exists(DRY_FILE) and os.path.exists(WET_FILE)):
        print(f"Errore: Assicurati che i file '{DRY_FILE}' e '{WET_FILE}' siano nella stessa cartella del file Python.")
        return

    print("--> Caricamento ed elaborazione file audio...")
    train_dry, train_wet = load_audio_pair(DRY_FILE, WET_FILE, segment_length=SEGMENT_LENGTH)

    train_dataset = torch.utils.data.TensorDataset(train_dry, train_wet)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = NeuralAmpLSTM(hidden_size=HIDDEN_SIZE, num_layers=2).to(DEVICE)
    criterion = CombinedSTFTTimeLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--> Inizio Addestramento Modello...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch_dry, batch_wet in train_loader:
            batch_dry, batch_wet = batch_dry.to(DEVICE), batch_wet.to(DEVICE)

            optimizer.zero_grad()
            output, _ = model(batch_dry)
            loss = criterion(output, batch_wet)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoca [{epoch:02d}/{EPOCHS:02d}] - Loss: {avg_loss:.5f}")

    # Salva automaticamente il file nella Home Directory (~/model_weights.json)
    export_json(model, "model_weights.json")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()