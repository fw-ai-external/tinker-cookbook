"""Verify the closed-form gradient ∂L/∂logits for (N, K) cross_entropy
against PyTorch autograd. No Tinker needed.
"""
import torch

torch.manual_seed(0)
N, K, V = 7, 5, 32

# Random logits; weights summing to 1 over K at active positions, with the first
# 3 rows zeroed out (the "prompt" mask).
logits = torch.randn(N, V, requires_grad=True)
target_tokens = torch.randint(0, V, (N, K))
raw = torch.rand(N, K)
weights = raw / raw.sum(dim=-1, keepdim=True)
weights[:3] = 0.0

# Forward exactly as the Tinker backend would
logprobs_full = torch.log_softmax(logits, dim=-1)            # (N, V)
logprobs = logprobs_full.gather(-1, target_tokens)           # (N, K)
L = (-logprobs * weights).sum()

# Autograd ground truth
(g_autograd,) = torch.autograd.grad(L, logits)

# Closed form
g_lp = -weights                                              # (N, K) = dL/dlogprobs
scatter = torch.zeros_like(logits).scatter_add_(-1, target_tokens, g_lp)  # (N, V)
softmax_correction = torch.softmax(logits, dim=-1) * g_lp.sum(dim=-1, keepdim=True)
g_closed_form = scatter - softmax_correction                 # (N, V)

print(f"max |Δ|  = {(g_autograd - g_closed_form).abs().max().item():.2e}")
print(f"row sums (autograd):   {g_autograd.sum(dim=-1).tolist()}")
print(f"row sums (closed form): {g_closed_form.sum(dim=-1).tolist()}")
