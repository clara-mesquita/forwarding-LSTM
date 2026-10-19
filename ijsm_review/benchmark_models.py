"""Adapters for pinned upstream forecasting architectures (see benchmark_sources.json)."""

import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace


ROOT = Path(__file__).resolve().parent
SOURCES = json.loads((ROOT / "benchmark_sources.json").read_text(encoding="utf-8"))
MODELS = ("iTransformer", "TimeBridge", "TimeMixerPP")


def fft_periods_without_dc(x, k=2):
    """Exclude DC structurally, including flat windows where all FFT magnitudes tie."""
    import torch

    spectrum = torch.fft.rfft(x, dim=1)
    magnitude = spectrum.abs().mean(0).mean(-1)
    count = min(k, len(magnitude) - 1)
    if count < 1:
        raise ValueError("At least two time steps are needed for period extraction.")
    indices = torch.topk(magnitude[1:], count).indices + 1
    frequencies = indices.detach().cpu().numpy()
    return x.shape[1] // frequencies, spectrum.abs().mean(-1)[:, indices], frequencies


def build_model(name, seq_len, pred_len, channels, d_model=32):
    """Run different model families in separate processes: upstream names collide."""
    import torch

    source_name = "PyPOTS" if name == "TimeMixerPP" else name
    source = SOURCES[source_name]
    path = ROOT / "vendor" / source_name
    if not (path / ".git").exists():
        raise FileNotFoundError("Run python ijsm_review/setup_benchmarks.py first.")
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True)
    if commit != source["commit"] or changes.strip():
        raise RuntimeError(f"{name}: upstream checkout differs from the pinned, clean revision.")
    if seq_len < 16 or seq_len % 4:
        raise ValueError("seq_len must be at least 16 and divisible by 4 for these patch/multiscale settings.")
    if d_model % 4:
        raise ValueError("d_model must be divisible by 4 attention heads.")

    config = SimpleNamespace(
        seq_len=seq_len, pred_len=pred_len, enc_in=channels,
        d_model=d_model, d_ff=d_model * 2, n_heads=4, e_layers=2,
        dropout=0.1, activation="gelu", output_attention=False,
        use_norm=True, embed="timeF", freq="h", factor=5, class_strategy="projection",
        revin=True, period=4, num_p=4, ia_layers=1, pd_layers=1,
        ca_layers=1, stable_len=3, attn_dropout=0.1,
    )
    if name != "TimeMixerPP":
        previous = getattr(build_model, "upstream", name)
        if previous != name:
            raise RuntimeError("Use separate processes for iTransformer and TimeBridge imports.")
        build_model.upstream = name
        sys.path.insert(0, str(path))
        model = importlib.import_module(f"model.{name}").Model(config)
        return model, vars(config), source

    # Import just the numerical modules, avoiding PyPOTS' unrelated application dependencies.
    namespace = ModuleType("benchmark_pypots")
    namespace.__path__ = [str(path / "pypots" / "nn" / "modules")]
    sys.modules[namespace.__name__] = namespace
    Backbone = importlib.import_module("benchmark_pypots.timemixerpp").BackboneTimeMixerPP
    # Merely setting DC's magnitude to zero still selects it on tied, flat spectra.
    importlib.import_module("benchmark_pypots.timemixerpp.layers").FFT_for_Period = fft_periods_without_dc

    class TimeMixerPP(Backbone):
        """PyPOTS BSD-3-Clause blocks with all-scale ensemble from paper Eq. 3.

        The pinned PyPOTS forecast returns only the last head. This adapter
        averages all heads explicitly; it is not an exact author-code reproduction.
        """

        def forward(self, x, *_):
            batch, _, features = x.shape
            scales = [x]
            for _ in range(self.downsampling_layers):
                channels = scales[-1].transpose(1, 2).reshape(batch * features, 1, -1)
                scales.append(self.down_pool(channels).reshape(batch, features, -1).transpose(1, 2))
            inputs = []
            for i, scale in enumerate(scales):
                scale = self.revin_layers[i](scale, mode="norm")
                inputs.append(scale.transpose(1, 2).reshape(batch * features, -1, 1))
            coarse = inputs[-1].reshape(batch, features, -1)
            mixed, _ = self.channel_mixing_attention(coarse, coarse, coarse, None)
            inputs[-1] = (coarse + mixed).reshape(batch * features, -1, 1)
            encoded = [self.enc_embedding(scale, None) for scale in inputs]
            for block in self.encoder_model:
                encoded = block(encoded)
            heads = []
            for predictor, scale in zip(self.predict_layers, encoded):
                output = predictor(scale.transpose(1, 2)).transpose(1, 2)
                output = self.projection_layer(output)
                heads.append(output.reshape(batch, features, self.n_pred_steps).transpose(1, 2))
            return self.revin_layers[0](torch.stack(heads).mean(0), mode="denorm")

    mixer_config = dict(
        task_name="long_term_forecast", n_steps=seq_len, n_features=channels,
        n_pred_steps=pred_len, n_pred_features=channels, n_layers=2,
        d_model=d_model, d_ffn=d_model * 2, n_heads=4, dropout=0.1,
        top_k=2, n_kernels=2, channel_mixing=True, channel_independence=True,
        downsampling_layers=2, downsampling_window=2, downsampling_method="conv",
        use_future_temporal_feature=False, use_norm=True,
    )
    return TimeMixerPP(**mixer_config), mixer_config, source
