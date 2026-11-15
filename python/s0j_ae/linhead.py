"""S0-J: deletion-set selection vs approximate-unlearning error (linear head).

Mechanism (docs/plans/2026-09-16-s0j-approx-error-task.md): exact influence
cancellation destroyed by approximate updates. All formulas are exact for
ridge (summed squared loss + fixed L2):

  theta_RT - theta0 = H_R^{-1} g_F          (exact retrain, quadratic)
  theta_guo - theta0 = H_D^{-1} g_F          (Certified Removal one-step, Guo 2020)
  theta_fish - theta0 = (diag(H_F)+lam I)^{-1} g_F   (damped diagonal-Newton family)
  e_V = v^T (P - H_R^{-1}) g_F               (extra target-direction error)
"""
import numpy as np


def ridge_head(X, y, lam):
    """Solve min (1/2n)||Xt - y||^2 + (lam/2)||t||^2.

    Returns (theta0, Hinv) with H_D = X^T X / n + lam I (used by operators).
    """
    n = X.shape[0]
    H = X.T @ X / n + lam * np.eye(X.shape[1])
    Hinv = np.linalg.inv(H)
    theta0 = Hinv @ (X.T @ y / n)
    return theta0, Hinv


def grad_contributions(X, y, theta0, n):
    """Per-sample gradients of the forget part at theta0: g_i = x_i(x_i^T t0 - y_i)/n.

    n is the ORIGINAL dataset size (same 1/n convention as H_D = X^TX/n + lam I),
    NOT X.shape[0]. Returns (d, len(y)); sum over axis=1 gives g_F.
    """
    return X.T * ((X @ theta0 - y) / float(n))  # (d, m)


def exact_retrain_shift(Hinv, Xf, g, n):
    """theta_RT - theta0 = H_R^{-1} g with H_R = H_D - Xf^T Xf / n (Woodbury).

    H_R^{-1} = Hinv + B (I - U^T Hinv U)^{-1} U^T Hinv,
    U = Xf^T / sqrt(n), B = Hinv U.  Exact for any set size.
    """
    U = Xf.T / np.sqrt(n)              # (d, k)
    B = Hinv @ U                       # (d, k)
    M = np.eye(Xf.shape[0]) - U.T @ B  # (k, k)
    Hg = Hinv @ g
    return Hg + B @ np.linalg.solve(M, U.T @ Hg)


def guo_shift(Hinv, g):
    """Certified-removal one-step: theta_U - theta0 = H_D^{-1} g."""
    return Hinv @ g


def fisher_shift(Xf, g, lam, n):
    """Damped diagonal-Newton: (diag(H_F) + lam I)^{-1} g with
    diag(H_F) = sum_i x_i^2 / n (same 1/n convention as H_D)."""
    df = (Xf ** 2).sum(axis=0) / float(n)
    d = df + lam
    if np.any(d <= 0):
        raise ValueError("non-positive diagonal under damping")
    return g / d


def two_d_example():
    """The advisor's 2D realization check. H_D - H_R = a a^T with a=(1,-2)
    (one real forget sample), g_F = -1 * a.  Returns the three v^T theta values.
    """
    HR = np.array([[1.0, 0.0], [0.0, 2.0]])
    HD = np.array([[2.0, -2.0], [-2.0, 6.0]])
    gF = np.array([-1.0, 2.0])
    a = np.array([1.0, -2.0])
    if not np.allclose(HD - HR, np.outer(a, a)):
        raise AssertionError("H_D - H_R is not the rank-1 sample Hessian")
    if not np.allclose(gF, -1.0 * a):
        raise AssertionError("g_F is not the single-sample gradient x*(r)")
    P = np.diag([1 / 2.0, 1 / 6.0])
    theta0 = np.array([0.05, 0.05])
    v = np.array([1.0, 1.0])
    s_exact = np.linalg.solve(HR, gF)
    s_approx = P @ gF
    return {
        "before": float(v @ theta0),
        "exact_retrain": float(v @ (theta0 + s_exact)),
        "approx_P": float(v @ (theta0 + s_approx)),
    }
