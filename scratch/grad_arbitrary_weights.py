"""Verify the (N, K) CE gradient formula works for arbitrary real-valued
weights — including negatives, magnitudes far from probabilities.
"""
import torch

torch.manual_seed(0)
N, K, V = 6, 4, 32

logits = torch.randn(N, V, requires_grad=True)
target_tokens = torch.randint(0, V, (N, K))

# Adversarial weights: signed, no sum constraint, big magnitude swings.
# Include a row where the K weights happen to sum to ~0 (cancellation).
weights = torch.tensor([
    [ 1.0, -1.0,  0.7, -0.7],   # row sum 0  → no softmax correction at all
    [ 5.3, -2.1,  0.0,  0.0],   # row sum 3.2
    [-3.0, -3.0, -3.0, -3.0],   # row sum -12  → softmax pushed positive at all v
    [ 0.0,  0.0,  0.0,  0.0],   # full mask
    [ 1e3, -1e3, 1e3, -1e3],    # huge but cancelling
    [ 0.1,  0.2,  0.3,  0.4],   # benign positive
])

logprobs_full = torch.log_softmax(logits, dim=-1)
logprobs = logprobs_full.gather(-1, target_tokens)
L = (-logprobs * weights).sum()
(g_autograd,) = torch.autograd.grad(L, logits)

# Closed-form formula — same expression as before, weights now arbitrary
g_lp = -weights                                                   # (N, K) upstream grad
scatter = torch.zeros_like(logits).scatter_add_(-1, target_tokens, g_lp)
correction = torch.softmax(logits, dim=-1) * g_lp.sum(dim=-1, keepdim=True)
g_closed_form = scatter - correction

print(f"max |Δ| over all rows = {(g_autograd - g_closed_form).abs().max().item():.2e}")
print()

# Print per-row diagnostic to show what's actually happening
softmax = torch.softmax(logits, dim=-1)
for n in range(N):
    rs = g_lp[n].sum().item()
    print(f"row {n}: weights={weights[n].tolist()}  Σ_k(-w)={rs:+.3f}")
    print(f"        max softmax-correction = {(softmax[n] * rs).abs().max().item():.3e}")
    print(f"        scatter slots ({target_tokens[n].tolist()}) get values {g_lp[n].tolist()}")
