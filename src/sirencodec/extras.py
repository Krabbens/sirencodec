# --- mel_refiner (conditional refiner net) ---
"""
Conditional mel refiner: maps coarse VQ mels toward ground-truth mels.

Residual dilated CNN (~0.5–1.5M params): delta = net(coarse), refined = coarse + delta.
"""
import torch
import torch.nn as nn


class MelRefinerNet(nn.Module):
    """Dilated residual stack on coarse log-mel [B, n_mels, T]."""

    def __init__(self, n_mels: int = 100, hidden: int = 128, n_layers: int = 10):
        super().__init__()
        self.n_mels = n_mels
        self.in_proj = nn.Conv1d(n_mels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            d = 2 ** (i % 4)
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden, hidden, 5, padding=2 * d, dilation=d),
                    nn.GroupNorm(8, hidden),
                    nn.SiLU(),
                    nn.Conv1d(hidden, hidden, 1),
                    nn.GroupNorm(8, hidden),
                )
            )
        self.out = nn.Conv1d(hidden, n_mels, kernel_size=1)

    def forward(self, coarse: torch.Tensor) -> torch.Tensor:
        """Returns residual delta; refined = coarse + delta."""
        h = self.in_proj(coarse)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)

    def refine(self, coarse: torch.Tensor) -> torch.Tensor:
        return coarse + self.forward(coarse)


# Alias for thesis docs
MelRefinerUNet = MelRefinerNet

# --- hifigan_vocoder (HF pretrained wrapper) ---
"""
Pre-trained HiFi-GAN vocoder for neural codec training.

Downloads and wraps the Microsoft SpeechT5 HiFi-GAN vocoder from HuggingFace.
Used as a frozen decoder: mel spectrogram → 16kHz audio.

Architecture (from microsoft/speecht5_hifigan):
- Input: 80-dim log-mel spectrogram (normalized)
- 4× upsampling (4^4 = 256x): 62.5 fps → 16kHz audio
- 12 dilated residual blocks (3 per stage, kernel sizes 3/7/11)
- Pre-trained on 685 hours of speech data
- Params: 7.2M (frozen)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from huggingface_hub import hf_hub_download


class ResBlock1(nn.Module):
    """Dilated residual block with weight normalization."""
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(
                channels, channels, kernel_size, stride=1,
                padding=(kernel_size * d - d) // 2, dilation=d))
            for d in dilations
        ])

    def forward(self, x):
        for conv in self.convs:
            xt = F.leaky_relu(x, 0.1)
            xt = conv(xt)
            x = xt + x
        return x


class SpeechT5HifiGAN(nn.Module):
    """Pre-trained HiFi-GAN vocoder matching Microsoft SpeechT5 architecture."""
    
    def __init__(self, weights_path=None):
        super().__init__()
        
        # Architecture config (from microsoft/speecht5_hifigan config.json)
        self.n_mels = 80
        self.upsample_rates = [4, 4, 4, 4]  # 4^4 = 256x upsampling
        self.upsample_kernel_sizes = [8, 8, 8, 8]
        self.resblock_kernel_sizes = [3, 7, 11]
        self.resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        self.upsample_initial_channel = 512
        
        # Normalization parameters (loaded from checkpoint)
        self.register_buffer('mean', torch.zeros(self.n_mels))
        self.register_buffer('scale', torch.ones(self.n_mels))
        
        # Input projection: 80 → 512
        self.conv_pre = nn.utils.weight_norm(
            nn.Conv1d(self.n_mels, self.upsample_initial_channel, 7, padding=3))
        
        # Upsampling layers: 512→256→128→64→32
        self.num_upsamples = len(self.upsample_rates)
        self.ups = nn.ModuleList()
        for i, (rate, kernel) in enumerate(zip(
                self.upsample_rates, self.upsample_kernel_sizes)):
            self.ups.append(nn.utils.weight_norm(
                nn.ConvTranspose1d(
                    self.upsample_initial_channel // (2**i),
                    self.upsample_initial_channel // (2**(i+1)),
                    kernel, stride=rate, padding=(kernel - rate) // 2)))
        
        # Residual blocks: 12 total (3 per upsampling stage)
        # Indices: stage 0 → blocks 0-2, stage 1 → blocks 3-5, etc.
        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            channels = self.upsample_initial_channel // (2**(i+1))
            for k, d in zip(self.resblock_kernel_sizes, self.resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(channels, k, d))
        
        # Output projection: 32 → 1
        self.conv_post = nn.utils.weight_norm(
            nn.Conv1d(
                self.upsample_initial_channel // (2**self.num_upsamples),
                1, 7, padding=3))
        
        # Load pre-trained weights if available
        if weights_path and Path(weights_path).exists():
            self._load_weights(weights_path)
        
        # Freeze all parameters (pre-trained vocoder)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
    
    def _load_weights(self, path):
        """Load pre-trained weights from HuggingFace checkpoint."""
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        
        # Load normalization parameters
        if 'mean' in state_dict:
            self.mean.copy_(state_dict['mean'])
        if 'scale' in state_dict:
            self.scale.copy_(state_dict['scale'])
        
        # Helper to load weight_norm parameter
        def load_wn(module, tensor):
            """Load weight into weight_norm module."""
            # weight_norm creates 'weight_orig' parameter
            if hasattr(module, 'weight_orig'):
                module.weight_orig.data.copy_(tensor)
            else:
                module.weight.data.copy_(tensor)
        
        # Load conv_pre
        load_wn(self.conv_pre, state_dict['conv_pre.weight'])
        self.conv_pre.bias.data.copy_(state_dict['conv_pre.bias'])
        
        # Load upsampling layers
        for i in range(self.num_upsamples):
            load_wn(self.ups[i], state_dict[f'upsampler.{i}.weight'])
            self.ups[i].bias.data.copy_(state_dict[f'upsampler.{i}.bias'])
        
        # Load residual blocks
        # Checkpoint uses resblocks.{0-11}.convs1.{0-2} naming
        block_idx = 0
        for stage in range(self.num_upsamples):
            for resblock_i in range(3):  # 3 resblocks per stage
                for conv_i in range(3):  # 3 convs per resblock
                    key_w = f'resblocks.{block_idx}.convs1.{conv_i}.weight'
                    key_b = f'resblocks.{block_idx}.convs1.{conv_i}.bias'
                    if key_w in state_dict:
                        load_wn(self.resblocks[block_idx].convs[conv_i], state_dict[key_w])
                        self.resblocks[block_idx].convs[conv_i].bias.data.copy_(state_dict[key_b])
                    block_idx += 1
        
        # Load conv_post
        load_wn(self.conv_post, state_dict['conv_post.weight'])
        self.conv_post.bias.data.copy_(state_dict['conv_post.bias'])
        
        print(f"  HiFi-GAN: loaded {len(state_dict)} weight tensors")
    
    def forward(self, mel):
        """Generate audio from mel spectrogram.
        
        Args:
            mel: [batch, n_mels, time] log-mel spectrogram
        
        Returns:
            audio: [batch, 1, time * 256] waveform at 16kHz
        """
        # Normalize input
        x = (mel - self.mean[:, None]) / self.scale[:, None]
        
        # Initial projection
        x = self.conv_pre(x)
        
        # Upsampling + residual blocks
        resblock_idx = 0
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            
            # Sum outputs from all residual blocks in this stage
            xs = None
            for j in range(3):  # 3 resblocks per stage
                out = self.resblocks[resblock_idx](x)
                if xs is None:
                    xs = out
                else:
                    xs = xs + out
                resblock_idx += 1
            x = xs / 3  # Average across resblocks
        
        # Output projection
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x  # [batch, 1, T_audio]
    
    @property
    def sample_rate(self):
        return 16000
    
    @property
    def hop_length(self):
        """Effective hop length: 256 samples (4^4 upsampling)."""
        return 256
    
    @property
    def fps(self):
        """Frame rate: 62.5 fps."""
        return self.sample_rate / self.hop_length
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters()) / 1e6
    
    def count_params(self):
        return self.num_params


# ──────────────────────────────────────────────
# Module-level loader (singleton)
# ──────────────────────────────────────────────
_HIFIGAN_INSTANCE = None

def get_hifigan_vocoder(device='cpu'):
    """Get or create pre-trained HiFi-GAN vocoder.
    
    Downloads weights from HuggingFace if not cached.
    """
    global _HIFIGAN_INSTANCE
    
    if _HIFIGAN_INSTANCE is not None:
        return _HIFIGAN_INSTANCE
    
    # Find weights in HuggingFace cache
    cache_path = Path.home() / '.cache/huggingface/hub'
    models = list(cache_path.glob('models--microsoft--speecht5_hifigan/snapshots/*/pytorch_model.bin'))
    
    if models:
        weights_path = str(models[0])
    else:
        print("Downloading HiFi-GAN vocoder from HuggingFace...")
        weights_path = hf_hub_download('microsoft/speecht5_hifigan', 'pytorch_model.bin')
    
    print(f"Loading HiFi-GAN vocoder...")
    vocoder = SpeechT5HifiGAN(weights_path=weights_path)
    vocoder = vocoder.to(device)
    
    print(f"  Params: {vocoder.num_params:.1f}M")
    print(f"  Input: {vocoder.n_mels}-dim mel spectrogram")
    print(f"  Output: 16kHz waveform (256× upsampling)")
    print(f"  Status: frozen (pre-trained)")
    
    _HIFIGAN_INSTANCE = vocoder
    return vocoder


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocoder = get_hifigan_vocoder(device)
    
    # Test forward pass
    mel = torch.randn(2, 80, 100, device=device)
    with torch.no_grad():
        audio = vocoder(mel)
    
    print(f"\nTest:")
    print(f"  Input:  {mel.shape}")
    print(f"  Output: {audio.shape}")
    print(f"  Duration: {audio.shape[-1] / 16000:.2f}s")
    print(f"  Range: [{audio.min():.3f}, {audio.max():.3f}]")
