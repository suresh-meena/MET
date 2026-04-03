"""
met/tokenizers/audio_tokenizer.py
===================================
Audio front-end: mel/codec latents → shared latent space ℝ^{B×L×D}.

Pipeline:
    raw waveform (B, T_samples) at 16 kHz
      → log-mel spectrogram (B, n_mels, T_frames) via torchaudio
        → temporal resample to L tokens: avg_pool1d or F.interpolate
          → linear projection U_a: n_mels → D
            → add learned positional embedding pos_emb_a: (L, D)
              → x_a_0: (B, L, D)

Frame rate math:
    At 16 kHz with hop=160: 100 fps audio frames → resample to L (e.g., 64)
    At 25 fps video: L = n_video_frames → match to L

The n_mels=128 default and log-mel parameters (n_fft=1024, hop=160)
are standard for audio-visual Foley tasks to capture the 0-8 kHz range.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AudioTokenizer(nn.Module):
    """
    Converts log-mel features to shared-width tokens x_a ∈ ℝ^{B×L×D}.

    Handles offline (pre-extracted mel) and online (raw waveform) modes.
    Online mode requires torchaudio.

    Args:
        n_mels: mel filterbank bins (default 128)
        D:      shared state width
        L:      target sequence length
        online: if True, accept raw waveform and compute mel on the fly
        sample_rate: audio sample rate (online mode only)
        hop_length:  STFT hop (online mode only)
    """

    def __init__(
        self,
        n_mels: int = 128,
        D: int = 256,
        L: int = 64,
        online: bool = False,
        sample_rate: int = 16_000,
        hop_length: int = 160,
        n_fft: int = 1024,
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.D = D
        self.L = L
        self.online = online

        if online:
            try:
                import torchaudio
                self.mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                    power=2.0,
                )
            except ImportError:
                raise ImportError(
                    "torchaudio is required for online AudioTokenizer. "
                    "Install with: pip install torchaudio"
                )
        else:
            self.mel_transform = None

        # Linear projection: n_mels → D
        self.U_a = nn.Linear(n_mels, D, bias=False)
        nn.init.normal_(self.U_a.weight, std=0.02)

        # Learned positional embedding: (1, L, D)
        self.pos_emb = nn.Parameter(torch.zeros(1, L, D))
        nn.init.normal_(self.pos_emb, std=0.02)

    def _extract_mel(self, waveform: Tensor) -> Tensor:
        """
        Args:
            waveform: (B, T_samples) raw audio
        Returns:
            log_mel: (B, T_frames, n_mels)
        """
        mel = self.mel_transform(waveform)           # (B, n_mels, T_frames)
        log_mel = torch.log(mel.clamp(min=1e-5))     # log for stability
        return log_mel.permute(0, 2, 1)              # (B, T_frames, n_mels)

    def forward(self, z_a: Tensor) -> Tensor:
        """
        Args:
            z_a: (B, T_raw, n_mels)  pre-extracted mel features  [offline]
                 (B, T_samples)      raw waveform                 [online]

        Returns:
            x_a: (B, L, D)  projected + position-encoded tokens
        """
        if self.online and z_a.dim() == 2:
            # Raw waveform → mel
            z_a = self._extract_mel(z_a)             # (B, T_frames, n_mels)

        B, T_raw, _ = z_a.shape

        # Resample to L tokens along time axis
        if T_raw != self.L:
            z_a = F.interpolate(
                z_a.permute(0, 2, 1),   # (B, n_mels, T_raw)
                size=self.L,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)           # (B, L, n_mels)

        # Project + positional embedding
        x_a = self.U_a(z_a) + self.pos_emb          # (B, L, D)
        return x_a

    def extra_repr(self) -> str:
        return f"n_mels={self.n_mels}, D={self.D}, L={self.L}, online={self.online}"
