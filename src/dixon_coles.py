"""Dixon-Coles 进球模型(时间衰减加权泊松 + 低比分相关修正)。

每支队有 攻击力 att、防守力 def;主队有主场加成 home;rho 修正 0-0/1-0/0-1/1-1
这类低比分的相关性。拟合后可输出完整比分矩阵,从而得到胜平负、大小球、
双方进球(BTTS)、最可能比分等概率。

  log λ(主队进球) = c0 + home + att[h] - def[a]
  log μ(客队进球) = c0 +        att[a] - def[h]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson


class DixonColes:
    def __init__(self, xi: float = 0.0018, l2: float = 0.01,
                 max_goals: int = 10, fit_years: float = 8.0):
        self.xi = xi            # 时间衰减(每天);0.0018 ≈ 半衰期约 1 年
        self.l2 = l2            # 攻防 L2 正则(破除平移简并、稳健)
        self.max_goals = max_goals
        self.fit_years = fit_years

    def fit(self, results: pd.DataFrame, ref_date):
        ref = pd.Timestamp(ref_date)
        d = results[(results.played) & (results.date < ref)
                    & (results.date >= ref - pd.DateOffset(years=self.fit_years))]
        teams = sorted(set(d.home_team) | set(d.away_team))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = d.home_team.map(idx).to_numpy()
        ai = d.away_team.map(idx).to_numpy()
        hg = d.home_score.to_numpy(float)
        ag = d.away_score.to_numpy(float)
        age = (ref - d.date).dt.days.to_numpy(float)
        w = np.exp(-self.xi * np.maximum(age, 0.0))
        lf = gammaln(hg + 1) + gammaln(ag + 1)

        def negll(p):
            att, dfn = p[:n], p[n:2 * n]
            home, rho, c0 = p[2 * n], p[2 * n + 1], p[2 * n + 2]
            loglam = c0 + home + att[hi] - dfn[ai]
            logmu = c0 + att[ai] - dfn[hi]
            lam, mu = np.exp(loglam), np.exp(logmu)
            ll = hg * loglam - lam + ag * logmu - mu - lf
            tau = np.ones_like(lam)
            m00 = (hg == 0) & (ag == 0); m01 = (hg == 0) & (ag == 1)
            m10 = (hg == 1) & (ag == 0); m11 = (hg == 1) & (ag == 1)
            tau[m00] = 1 - lam[m00] * mu[m00] * rho
            tau[m01] = 1 + lam[m01] * rho
            tau[m10] = 1 + mu[m10] * rho
            tau[m11] = 1 - rho
            ll = ll + np.log(np.clip(tau, 1e-9, None))
            return -(w * ll).sum() + self.l2 * (att @ att + dfn @ dfn)

        p0 = np.zeros(2 * n + 3)
        p0[2 * n] = 0.25                       # home
        p0[2 * n + 2] = np.log(max(hg.mean(), 0.5))  # c0
        res = minimize(negll, p0, method="L-BFGS-B", options={"maxiter": 150})
        x = res.x
        self.teams, self.idx = teams, idx
        self.att = x[:n] - x[:n].mean()
        self.def_ = x[n:2 * n]
        self.home, self.rho, self.c0 = x[2 * n], x[2 * n + 1], x[2 * n + 2]
        return self

    def rates(self, h, a, neutral=False):
        ih, ia = self.idx.get(h), self.idx.get(a)
        if ih is None or ia is None:
            return None
        home = 0.0 if neutral else self.home
        lam = np.exp(self.c0 + home + self.att[ih] - self.def_[ia])
        mu = np.exp(self.c0 + self.att[ia] - self.def_[ih])
        return float(lam), float(mu)

    def score_matrix(self, h, a, neutral=False):
        r = self.rates(h, a, neutral)
        if r is None:
            return None
        lam, mu = r
        k = np.arange(self.max_goals + 1)
        M = np.outer(poisson.pmf(k, lam), poisson.pmf(k, mu))
        M[0, 0] *= 1 - lam * mu * self.rho
        M[0, 1] *= 1 + lam * self.rho
        M[1, 0] *= 1 + mu * self.rho
        M[1, 1] *= 1 - self.rho
        M = np.clip(M, 0, None)
        return M / M.sum()

    def predict_wdl(self, h, a, neutral=False):
        M = self.score_matrix(h, a, neutral)
        if M is None:
            return None
        return np.array([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()])

    def markets(self, h, a, neutral=False) -> dict | None:
        """额外盘口:大小球 2.5、双方进球、最可能比分、预期进球。"""
        M = self.score_matrix(h, a, neutral)
        if M is None:
            return None
        K = M.shape[0]
        x = np.arange(K)
        total = x[:, None] + x[None, :]
        over25 = M[total >= 3].sum()
        btts = M[1:, 1:].sum()
        i, j = np.unravel_index(M.argmax(), M.shape)
        lam, mu = self.rates(h, a, neutral)
        return {"over25": float(over25), "btts": float(btts),
                "score": (int(i), int(j)), "xg_home": lam, "xg_away": mu}
