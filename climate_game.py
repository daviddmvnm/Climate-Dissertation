"""
climate_game.py
───────────────
Core mechanics for the Climate Cooperation Markov Game.

Import in any notebook:
    from climate_game import (
        GameParams, PLAYERS, ALL_G,
        solve_model, simulate_modal_path, monte_carlo,
        compute_W, threshold_sigmoid,
    )
"""

import numpy as np
from itertools import product
from dataclasses import dataclass, field
from typing import Dict, Optional

# ─── PLAYERS ──────────────────────────────────────────────────────
PLAYERS = ["US", "EU", "CN", "RoW"]

# ─── PARAMETERS ───────────────────────────────────────────────────

@dataclass
class GameParams:
    T: int = 10
    theta: float = 0.85
    phi: float = 0.5

    lam: Dict[str, float] = field(default_factory=lambda: {
        "US": 1.5, "EU": 1.5, "CN": 1.5, "RoW": 1.5
    })
    b: float = 0.5
    gamma: float = 0.25
    kappa: float = 0.05
    eta: float = 15.0

    weights:  Dict[str, float] = field(default_factory=lambda: {
        "US": 0.28, "EU": 0.22, "CN": 0.30, "RoW": 0.20
    })
    costs:    Dict[str, float] = field(default_factory=lambda: {
        "US": 3.0, "EU": 2.5, "CN": 3.5, "RoW": 8.0
    })
    c_tilde:  Dict[str, float] = field(default_factory=lambda: {
        "US": 3.0, "EU": 2.5, "CN": 3.5, "RoW": 8.0
    })
    damages:  Dict[str, float] = field(default_factory=lambda: {
        "US": 1.0, "EU": 1.2, "CN": 1.5, "RoW": 3.0
    })
    pressure: Dict[str, float] = field(default_factory=lambda: {
        "US": 0.8, "EU": 1.0, "CN": 0.6, "RoW": 1.5
    })
    discount: Dict[str, float] = field(default_factory=lambda: {
        "US": 0.92, "EU": 0.93, "CN": 0.90, "RoW": 0.88
    })


# ─── STATE SPACE ──────────────────────────────────────────────────
ALL_G = list(product([0, 1], repeat=len(PLAYERS)))  # 2^4 = 16 states


# ─── STATE HELPERS ────────────────────────────────────────────────

def compute_W(G: tuple, params: GameParams) -> float:
    """Weighted adoption share: W = Σ w_i * G_i."""
    return sum(params.weights[p] * G[i] for i, p in enumerate(PLAYERS))


def get_active(G: tuple) -> list:
    """Indices of players who haven't adopted yet."""
    return [i for i, g in enumerate(G) if g == 0]


# ─── SIGMOID ──────────────────────────────────────────────────────

def sigmoid(x):
    """Numerically stable logistic sigmoid."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x)),
    )


def threshold_sigmoid(W: float, params: GameParams) -> float:
    """Smooth coordination threshold: 0 when W << θ, 1 when W >> θ."""
    return float(sigmoid(params.eta * (W - params.theta)))


# ─── FLOW PAYOFFS ─────────────────────────────────────────────────

def adoption_cost(player: str, W: float, params: GameParams) -> float:
    return params.costs[player] - params.c_tilde[player] * (1.0 - params.phi) * params.gamma * W


def climate_damage(player: str, t: int, W: float, params: GameParams) -> float:
    S = threshold_sigmoid(W, params)
    return params.damages[player] * (1 + params.kappa * t) * (1 - S)


def political_pressure(player: str, t: int, W: float, params: GameParams) -> float:
    return params.phi * params.pressure[player] * W


def stabilisation_benefit(W: float, params: GameParams) -> float:
    return params.b * threshold_sigmoid(W, params)


def flow_state(player: str, t: int, W: float, params: GameParams) -> float:
    return -climate_damage(player, t, W, params) + stabilisation_benefit(W, params)


def flow_adopt(player: str, t: int, W: float, params: GameParams) -> float:
    return -adoption_cost(player, W, params) + flow_state(player, t, W, params)


def flow_delay(player: str, t: int, W: float, params: GameParams) -> float:
    return -political_pressure(player, t, W, params) + flow_state(player, t, W, params)


# ─── QRE ──────────────────────────────────────────────────────────

def qre_probability(QA: float, QD: float, lam: float) -> float:
    """Adoption probability from Q-values via logistic QRE rule."""
    maxQ = max(QA, QD)
    eA = np.exp(lam * (QA - maxQ))
    eD = np.exp(lam * (QD - maxQ))
    return eA / (eA + eD)


def inclusive_value(QA: float, QD: float, lam: float) -> float:
    """Soft-max value before action is chosen (captures optionality)."""
    maxQ = max(QA, QD)
    eA = np.exp(lam * (QA - maxQ))
    eD = np.exp(lam * (QD - maxQ))
    return maxQ + (1 / lam) * np.log(eA + eD)


# ─── SOLVER ───────────────────────────────────────────────────────

# Damping factor for the QRE fixed-point iteration:
#   σ_{k+1} = α·F(σ_k) + (1-α)·σ_k
# Preserves the fixed points of F but ensures convergence across the
# full GSA parameter range, where the undamped map can be non-contractive
# at isolated nodes. See Appendix A.1.
QRE_DAMPING = 0.5


def solve_model(params: GameParams):
    """
    Backward induction + QRE fixed-point solver.

    Returns
    -------
    V        : V[t][G][i]     — inclusive value
    sigma    : sigma[t][G][i] — equilibrium adoption probability
    QA_table : QA[t][G][i]    — Q-value for adopting
    QD_table : QD[t][G][i]    — Q-value for delaying
    """
    T = params.T
    n = len(PLAYERS)

    V        = {}
    sigma    = {}
    QA_table = {}
    QD_table = {}

    # Terminal condition: V(T+1, .) = 0
    V[T + 1] = {G: {i: 0.0 for i in range(n)} for G in ALL_G}

    for t in range(T, 0, -1):
        V[t]        = {}
        sigma[t]    = {}
        QA_table[t] = {}
        QD_table[t] = {}

        for G in ALL_G:
            W      = compute_W(G, params)
            active = get_active(G)

            # ── All adopted: no decisions ──────────────────────────
            if not active:
                V[t][G]        = {}
                sigma[t][G]    = {}
                QA_table[t][G] = {}
                QD_table[t][G] = {}
                for i in range(n):
                    p = PLAYERS[i]
                    V[t][G][i]        = flow_state(p, t, W, params) + params.discount[p] * V[t + 1][G][i]
                    sigma[t][G][i]    = 1.0
                    QA_table[t][G][i] = 0.0
                    QD_table[t][G][i] = 0.0
                continue

            # ── QRE fixed-point iteration ──────────────────────────
            current_sigmas = {i: 0.5 for i in active}
            new_QA = {}
            new_QD = {}

            for _ in range(2000):
                new_sigmas = {}

                for i in active:
                    p      = PLAYERS[i]
                    others = [j for j in active if j != i]
                    other_combos = list(product([0, 1], repeat=len(others)))

                    EV_adopt = 0.0
                    EV_delay = 0.0

                    for oa in other_combos:
                        prob = 1.0
                        for idx, j in enumerate(others):
                            prob *= current_sigmas[j] if oa[idx] == 1 else (1 - current_sigmas[j])

                        G_adopt = list(G)
                        G_adopt[i] = 1
                        G_delay = list(G)
                        for idx, j in enumerate(others):
                            if oa[idx] == 1:
                                G_adopt[j] = 1
                                G_delay[j] = 1
                        EV_adopt += prob * V[t + 1][tuple(G_adopt)][i]
                        EV_delay += prob * V[t + 1][tuple(G_delay)][i]

                    qa = flow_adopt(p, t, W, params) + params.discount[p] * EV_adopt
                    qd = flow_delay(p, t, W, params) + params.discount[p] * EV_delay
                    new_QA[i]     = qa
                    new_QD[i]     = qd
                    new_sigmas[i] = qre_probability(qa, qd, params.lam[p])

                damped_sigmas = {
                    i: QRE_DAMPING * new_sigmas[i] + (1 - QRE_DAMPING) * current_sigmas[i]
                    for i in active
                }
                max_diff = max(abs(damped_sigmas[i] - current_sigmas[i]) for i in active)
                current_sigmas = damped_sigmas
                if max_diff < 1e-8:
                    break

            # ── Store results ──────────────────────────────────────
            V[t][G]        = {}
            sigma[t][G]    = {}
            QA_table[t][G] = {}
            QD_table[t][G] = {}

            for i in range(n):
                p = PLAYERS[i]
                if G[i] == 1:
                    sigma[t][G][i]    = 1.0
                    QA_table[t][G][i] = 0.0
                    QD_table[t][G][i] = 0.0
                    # Expected future given others' equilibrium play
                    ev = 0.0
                    for oa in product([0, 1], repeat=len(active)):
                        prob = 1.0
                        G_next = list(G)
                        for idx, j in enumerate(active):
                            prob *= current_sigmas[j] if oa[idx] == 1 else (1 - current_sigmas[j])
                            if oa[idx] == 1:
                                G_next[j] = 1
                        ev += prob * V[t + 1][tuple(G_next)][i]
                    V[t][G][i] = flow_state(p, t, W, params) + params.discount[p] * ev
                else:
                    sigma[t][G][i]    = current_sigmas[i]
                    QA_table[t][G][i] = new_QA[i]
                    QD_table[t][G][i] = new_QD[i]
                    V[t][G][i]        = inclusive_value(new_QA[i], new_QD[i], params.lam[p])

    return V, sigma, QA_table, QD_table


# ─── SIMULATION ───────────────────────────────────────────────────

def simulate_modal_path(V, sigma, params: GameParams) -> list:
    """
    Trace the most likely path: adopt if σ > 0.5.
    Returns a list of dicts, one per period + a final state entry.
    """
    G    = (0, 0, 0, 0)
    path = []

    for t in range(1, params.T + 1):
        W      = compute_W(G, params)
        active = get_active(G)
        entry  = {"t": t, "W": W, "G": G, "threshold_met": W >= params.theta}
        for i, p in enumerate(PLAYERS):
            entry[f"sigma_{p}"]   = sigma[t][G][i]
            entry[f"adopted_{p}"] = bool(G[i])
        path.append(entry)

        G_new = list(G)
        for i in active:
            if sigma[t][G][i] > 0.5:
                G_new[i] = 1
        G = tuple(G_new)

    W_final = compute_W(G, params)
    path.append({"t": params.T + 1, "W": W_final, "G": G, "threshold_met": W_final >= params.theta})
    return path


def monte_carlo(V, sigma, params: GameParams, n_runs: int = 1000, seed: int = 42):
    """
    Stochastic forward simulation.

    Returns
    -------
    W_paths    : (n_runs, T+1) array of W values per period
    adopt_time : (n_runs, 4)   array of adoption periods (inf = never)
    """
    rng = np.random.default_rng(seed)
    n   = len(PLAYERS)
    T   = params.T

    W_paths    = np.zeros((n_runs, T + 1))
    adopt_time = np.full((n_runs, n), np.inf)

    for run in range(n_runs):
        G = (0, 0, 0, 0)
        for t in range(1, T + 1):
            W_paths[run, t - 1] = compute_W(G, params)
            active = get_active(G)
            G_new  = list(G)
            for i in active:
                if rng.random() < sigma[t][G][i]:
                    G_new[i] = 1
                    if adopt_time[run, i] == np.inf:
                        adopt_time[run, i] = t
            G = tuple(G_new)
        W_paths[run, T] = compute_W(G, params)

    return W_paths, adopt_time
