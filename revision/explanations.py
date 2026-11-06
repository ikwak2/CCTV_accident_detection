"""Explicit token-grid rollout and a common class-specific saliency baseline."""
import torch
import torch.nn.functional as F


def token_rollout(attentions, grid, readout):
    """No arbitrary factorization and no CLS assumption for mean-pooled VideoMAE.

    These attention maps are structural diagnostics, not calibrated probabilities
    and not evidence of faithfulness without perturbation/localization evaluation.
    """
    t,h,w=grid
    n=t*h*w
    expected=n+1 if readout=='cls' else n
    if readout not in ['cls','mean']:
        raise ValueError('Use a model-compatible readout')
    if not attentions or any(a is None for a in attentions):
        raise ValueError('Attention extraction did not return matrices')
    result=None
    for att in attentions:
        if tuple(att.shape[-2:])!=(expected,expected):
            raise ValueError(f'Expected {expected} tokens, received {tuple(att.shape)}')
        a=att.float().mean(1)
        a=a+torch.eye(expected,device=a.device)[None]
        a=a/a.sum(-1,keepdim=True)
        result=a if result is None else a@result
    values=result[:,0,1:] if readout=='cls' else result.mean(1)
    return values.reshape(values.shape[0],t,h,w)


def gradient_times_input(model,pixels):
    """Positive gradient×input for the accident-vs-normal logit margin.

    Common pixel-space baseline for all three backbones; no claim of operator
    utility. Zero baseline is in normalized input space.
    """
    model.eval()
    x=pixels.detach().float().requires_grad_(True)
    with torch.enable_grad():
        logits=model(x).logits
        gradient=torch.autograd.grad((logits[:,1]-logits[:,0]).sum(),x)[0]
    values=(gradient*x).sum(2).clamp_min(0)
    b,t,h,w=values.shape
    patches=F.avg_pool2d(values.reshape(b*t,1,h,w),16).reshape(b,t,h//16,w//16)
    return patches.detach(),logits.detach()
