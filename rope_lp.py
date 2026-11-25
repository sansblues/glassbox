# Verify the prior claim and LP-optimize prev-token RoPE amplitudes.
import numpy as np
from scipy.optimize import linprog

def freqs(theta, dh):
    return theta ** (-np.arange(0, dh, 2) / dh)

def gap_equal(w, L):
    d = np.arange(0, L + 1) - 1.0  # Delta-1
    d = d[d != 0]
    s = np.cos(np.outer(d, w)).sum(1)
    return len(w) - s.max(), d[s.argmax()] + 1

# claim: 16 high-freq pairs, base 1e4 (dh=128), Delta in [0,32768]
for dh in (128,):
    w = freqs(1e4, dh)[:16]
    g, at = gap_equal(w, 32768)
    print(f"claim check dh={dh} base=1e4 16 pairs equal amp: gap/unit-amp={g:.3f} worst Delta={at:.0f}")
    g, at = gap_equal(w, 8192)
    print(f"   same over [0,8192]: {g:.3f} at {at:.0f}")

def lp(w, L, sumA=1.0):
    # maximize t s.t. sum_j A_j (1 - cos((D-1) w_j)) >= t for all D != 1 in [0,L], A>=0, sum A = 1
    d = np.arange(0, L + 1) - 1.0
    d = d[d != 0]
    M = 1 - np.cos(np.outer(d, w))  # (nD, n)
    n = len(w)
    c = np.zeros(n + 1); c[-1] = -1
    A_ub = np.hstack([-M, np.ones((len(d), 1))])
    b_ub = np.zeros(len(d))
    A_eq = np.zeros((1, n + 1)); A_eq[0, :n] = 1
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[sumA], bounds=[(0, None)] * n + [(None, None)], method="highs")
    return -res.fun, res.x[:n]

if __name__ == "__main__":
    import sys
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    for theta in (1e4, 1e6, 1e8, 1e10, 1e12):
        w = freqs(theta, 128)
        # only pairs with non-negligible freq are useful for layer 1
        use = w > 1e-5
        t, A = lp(w[use], L)
        # equal-amp 16 pairs comparison
        print(f"theta={theta:.0e} dh=128 L={L}: LP gap per unit total amplitude = {t:.4f}; "
              f"n_pairs_used={(A>1e-6).sum()}; unrotated pairs (max angle<0.1 at L): {(w*L<0.1).sum()}")
