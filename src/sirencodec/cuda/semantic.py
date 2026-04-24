from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchaudio


@dataclass(frozen=True)
class SemanticLossInfo:
    feature_cos: float
    subset_items: int


def _resolve_bundle(name: str):
    bundle = getattr(torchaudio.pipelines, str(name), None)
    if bundle is None:
        raise ValueError(f"unknown semantic model bundle {name!r}")
    return bundle


def _select_subset(x: torch.Tensor, max_items: int) -> torch.Tensor:
    batch = int(x.shape[0])
    n = min(batch, max(1, int(max_items)))
    if n >= batch:
        return x
    idx = torch.linspace(0, batch - 1, steps=n, device=x.device).round().long()
    return x.index_select(0, idx)


def _match_time(ref: torch.Tensor, est: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    t = min(int(ref.shape[1]), int(est.shape[1]))
    return ref[:, :t, :], est[:, :t, :]


class FrozenSemanticTeacher:
    """Frozen SSL feature extractor used only while training."""

    def __init__(self, *, bundle_name: str, layers: tuple[int, ...], sample_rate: int, device: torch.device):
        bundle = _resolve_bundle(bundle_name)
        self.bundle_name = str(bundle_name)
        self.sample_rate = int(sample_rate)
        self.teacher_sample_rate = int(bundle.sample_rate)
        self.layers = tuple(sorted(set(int(x) for x in layers)))
        if not self.layers:
            raise ValueError("semantic layers cannot be empty")
        if min(self.layers) < 1:
            raise ValueError("semantic layers must be 1-based positive integers")
        try:
            model = bundle.get_model(dl_kwargs={"weights_only": True})
        except TypeError:
            model = bundle.get_model()
        self.model = model.to(device=device, dtype=torch.float32).eval()
        self.model.requires_grad_(False)

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        wav = x[..., 0].float()
        if self.sample_rate != self.teacher_sample_rate:
            wav = torchaudio.functional.resample(wav, self.sample_rate, self.teacher_sample_rate)
        return wav

    def _extract(self, x: torch.Tensor, *, grad_inputs: bool) -> list[torch.Tensor]:
        wav = self._prepare(x)
        num_layers = max(self.layers)
        if grad_inputs:
            feats, _ = self.model.extract_features(wav, num_layers=num_layers)
        else:
            with torch.no_grad():
                feats, _ = self.model.extract_features(wav, num_layers=num_layers)
        return [feats[i - 1] for i in self.layers]

    def loss(self, reference: torch.Tensor, estimate: torch.Tensor, *, max_items: int) -> tuple[torch.Tensor, SemanticLossInfo]:
        ref_subset = _select_subset(reference, max_items)
        est_subset = _select_subset(estimate, max_items)
        ref_feats = self._extract(ref_subset.detach(), grad_inputs=False)
        est_feats = self._extract(est_subset, grad_inputs=True)
        total = est_subset.new_zeros((), dtype=torch.float32)
        total_cos = 0.0
        for ref_feat, est_feat in zip(ref_feats, est_feats):
            ref_feat, est_feat = _match_time(ref_feat, est_feat)
            cos = F.cosine_similarity(ref_feat, est_feat, dim=-1, eps=1e-8).mean()
            total = total + (1.0 - cos)
            total_cos += float(cos.detach().float().item())
        n_layers = max(1, len(ref_feats))
        return total / float(n_layers), SemanticLossInfo(feature_cos=total_cos / float(n_layers), subset_items=int(ref_subset.shape[0]))
