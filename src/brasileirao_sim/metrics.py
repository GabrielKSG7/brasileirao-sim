"""Métricas de avaliação probabilística.

Jogo a jogo (3 resultados ordenados: mandante, empate, visitante):
- RPS (ranked probability score): padrão em futebol, premia errar "por pouco" (dar empate
  quando o mandante venceu é menos grave que dar vitória do visitante). Menor é melhor.
- Log loss: -log da probabilidade dada ao que aconteceu. Pune muito a confiança errada.
- Brier: soma dos quadrados dos erros nas três probabilidades.

Temporada (eventos binários por time: título, G4, rebaixamento...):
- Brier e log loss binários; probabilidades 0/1 do Monte Carlo são limitadas a
  [EPS, 1-EPS] no log loss, senão um único erro "impossível" daria infinito.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4
OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}


def _onehot(results: pd.Series) -> np.ndarray:
    idx = results.map(OUTCOME_INDEX).to_numpy()
    out = np.zeros((len(idx), 3))
    out[np.arange(len(idx)), idx] = 1
    return out


def match_scores(probs: np.ndarray, results: pd.Series) -> pd.DataFrame:
    """Uma linha por jogo com rps, log_loss e brier. probs: (n, 3) [H, D, A]."""
    probs = np.asarray(probs, dtype=float)
    o = _onehot(results)
    cum_p, cum_o = probs.cumsum(axis=1), o.cumsum(axis=1)
    rps = 0.5 * ((cum_p[:, :2] - cum_o[:, :2]) ** 2).sum(axis=1)
    p_true = (probs * o).sum(axis=1)
    return pd.DataFrame(
        {
            "rps": rps,
            "log_loss": -np.log(np.clip(p_true, EPS, 1)),
            "brier": ((probs - o) ** 2).sum(axis=1),
        }
    )


def binary_scores(p: np.ndarray, outcome: np.ndarray) -> pd.DataFrame:
    p = np.asarray(p, dtype=float)
    o = np.asarray(outcome, dtype=float)
    pc = np.clip(p, EPS, 1 - EPS)
    return pd.DataFrame(
        {
            "brier": (p - o) ** 2,
            "log_loss": -(o * np.log(pc) + (1 - o) * np.log(1 - pc)),
        }
    )


def calibration_table(p: np.ndarray, outcome: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Agrupa previsões por faixa de probabilidade e compara com a frequência observada."""
    p = np.asarray(p, dtype=float)
    o = np.asarray(outcome, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = which == b
        if m.any():
            rows.append(
                {
                    "faixa": f"{edges[b]:.0%}-{edges[b + 1]:.0%}",
                    "n": int(m.sum()),
                    "prob_media": float(p[m].mean()),
                    "freq_observada": float(o[m].mean()),
                }
            )
    return pd.DataFrame(rows)
