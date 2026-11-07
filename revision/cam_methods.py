"""Explicit video token layouts and CAMs; no temporal averaging or smoothing.

LayerCAM and Grad-CAM++ follow the pytorch-grad-cam reference formulas, applied
to a [B,C,T,H,W] activation volume. Raw rollout is a token-mixing diagnostic;
it ignores value/output projections and MLPs and is not class-specific.
"""
import torch


def activation_volume(value, model, frames, height=14, width=14, attention_frames=None):
    """Last pre-spatial-attention LayerNorm outputs -> [B,C,T,H,W]."""
    n, channels = height * width, value.shape[-1]
    if model == 'timesformer':
        # Restore the actual flattened input-token order even when a legacy
        # checkpoint's config.num_frames differs from its supplied input length.
        internal_frames = frames if attention_frames is None else attention_frames
        if frames * n % internal_frames:
            raise ValueError('Input and configured attention grids are incompatible')
        positions = frames * n // internal_frames
        if value.ndim != 3 or value.shape[0] % internal_frames or value.shape[1] != positions + 1:
            raise ValueError(f'Invalid TimeSformer spatial token layout: {tuple(value.shape)}')
        patches = value[:, 1:].reshape(-1, internal_frames, positions, channels).transpose(1, 2)
        patches = patches.reshape(-1, n, frames, channels).transpose(1, 2)
        patches = patches.reshape(-1, frames, height, width, channels)
    else:
        tubelets = frames // 2
        expected = tubelets * n + (model == 'vivit')
        if model not in ('vivit', 'videomae') or value.ndim != 3 or value.shape[1] != expected:
            raise ValueError(f'Invalid joint token layout: {model} {tuple(value.shape)}')
        patches = value[:, 1:] if model == 'vivit' else value
        patches = patches.reshape(value.shape[0], tubelets, height, width, channels)
    return patches.permute(0, 4, 1, 2, 3).contiguous()


def restore_divided_grid(values, input_frames, height=14, width=14):
    """Undo HF's configured-time grouping to the original patch-major input."""
    batch, configured_frames = values.shape[:2]
    if values.numel() != batch * input_frames * height * width:
        raise ValueError('Internal rollout volume does not match actual input tokens')
    flat = values.reshape(batch, configured_frames, -1).transpose(1, 2)
    return flat.reshape(batch, height * width, input_frames).transpose(1, 2).reshape(batch, input_frames, height, width)


def gradient_cams(activations, gradients):
    """Accident-logit target; reference Grad-CAM++ first-gradient approximation.

    References:
    github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/layer_cam.py
    github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/grad_cam_plusplus.py
    """
    if activations.shape != gradients.shape or activations.ndim != 5:
        raise ValueError('Expected matching [B,C,T,H,W] tensors')
    if not torch.isfinite(activations).all() or not torch.isfinite(gradients).all():
        raise ValueError('Nonfinite activation or gradient')
    a, g = activations.float(), gradients.float()
    layercam = (g.clamp_min(0) * a).sum(1).clamp_min(0)
    g2, g3 = g.square(), g.square() * g
    denominator = 2 * g2 + a.sum((2, 3, 4), keepdim=True) * g3 + 1e-6
    alpha = torch.where(g != 0, g2 / denominator, torch.zeros_like(g))
    weights = (alpha * g.clamp_min(0)).sum((2, 3, 4), keepdim=True)
    gradcampp = (weights * a).sum(1).clamp_min(0)
    return {'layercam': layercam, 'gradcampp': gradcampp}


def joint_rollout(attentions, grid, cls):
    """Exact readout of residual-normalized joint rollout, using row vectors."""
    tokens = grid[0] * grid[1] * grid[2]
    expected = tokens + int(cls)
    if not attentions or any(a.shape[-2:] != (expected, expected) for a in attentions):
        raise ValueError('Attention shape does not match explicit token grid')
    relevance = attentions[0].new_zeros((attentions[0].shape[0], 1, expected))
    if cls:
        relevance[:, :, 0] = 1
    else:
        relevance[:] = 1 / expected
    for attention in reversed(attentions):
        matrix = attention.float().mean(1) if attention.ndim == 4 else attention.float()
        matrix = matrix + torch.eye(expected, device=matrix.device)[None]
        matrix = matrix / matrix.sum(-1, keepdim=True)
        relevance = relevance @ matrix
    if cls:
        relevance = relevance[:, :, 1:]
    return relevance.reshape(-1, *grid)


def divided_rollout(spatial_attentions, temporal_attentions, frames, height=14, width=14):
    """TimeSformer residual attention-flow adaptation, with frame-mean CLS.

    Reverse propagation through each block's spatial then temporal token mixing
    is algebraically equivalent to the dense joint transition V @ U. A separate
    identity path at each sublayer receives weight 1/2, as in standard rollout.
    HF 4.46.3 returns spatial attention only; temporal attention must be captured
    from each temporal_attention.attention module. This is an explicit adaptation,
    not a validated localization metric or an exact feature contribution.
    """
    patches = height * width
    if not spatial_attentions or len(spatial_attentions) != len(temporal_attentions):
        raise ValueError('Need spatial AND temporal attention for every block')
    first = spatial_attentions[0]
    batch = first.shape[0] // frames
    cls = first.new_ones(batch)
    values = first.new_zeros(batch, patches, frames)
    for spatial, temporal in reversed(list(zip(spatial_attentions, temporal_attentions))):
        if spatial.shape != (batch * frames, patches + 1, patches + 1):
            raise ValueError(f'Invalid spatial attention shape {tuple(spatial.shape)}')
        if temporal.shape != (batch * patches, frames, frames):
            raise ValueError(f'Invalid temporal attention shape {tuple(temporal.shape)}')
        spatial = spatial.float().reshape(batch, frames, patches + 1, patches + 1)
        temporal = temporal.float().reshape(batch, patches, frames, frames)
        spatial = spatial / spatial.sum(-1, keepdim=True)
        temporal = temporal / temporal.sum(-1, keepdim=True)
        queries = torch.cat((cls[:, None, None].expand(-1, frames, 1) / frames,
                             values.transpose(1, 2)), dim=-1)
        propagated = torch.einsum('bti,btij->btj', queries, spatial)
        cls = .5 * cls + .5 * propagated[:, :, 0].sum(1)
        values = .5 * values + .5 * propagated[:, :, 1:].transpose(1, 2)
        values = .5 * values + .5 * torch.einsum('bnt,bnts->bns', values, temporal)
    if not torch.allclose(cls + values.sum((1, 2)), torch.ones_like(cls), atol=2e-5):
        raise ValueError('Attention-flow mass conservation failed')
    return values.transpose(1, 2).reshape(batch, frames, height, width)
