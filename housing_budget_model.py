#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
studio_model.py — математическая модель анализа экономической активности
и распределения бюджетных средств на жилищное имущество (студия 24 м²).

Модель включает:
  - детерминированную базу: долг, капитал, резерв, условия θ и φ;
  - стохастику: AR(1) доход, инфляция/ГБД цены м², Монте-Карло;
  - сравнение «покупка vs аренда» на общих шоках;
  - точку безубыточности r_inv* (благосостояние) и r_inv** (полезность);
  - чувствительность к ставке ипотеки, цене м² и аренде.

Запуск: см. README.md
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # безопасно для headless-сред
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import brentq


# 1. Параметры

@dataclass
class HousingParams:
    area: float = 24.0
    price_per_m2: float = 120_000
    down_payment_share: float = 0.20
    annual_rate: float = 0.12
    years: int = 20
    inflation_m2: float = 0.04 / 12
    price_mode: str = "deterministic"   # "deterministic" | "stochastic"
    price_mu: float = 0.04 / 12
    price_sigma: float = 0.01

    @property
    def price(self) -> float:
        return self.area * self.price_per_m2

    @property
    def down_payment(self) -> float:
        return self.price * self.down_payment_share

    @property
    def loan_principal(self) -> float:
        return self.price - self.down_payment

    @property
    def monthly_rate(self) -> float:
        return self.annual_rate / 12

    @property
    def n_months(self) -> int:
        return self.years * 12

    @property
    def annuity(self) -> float:
        r, n = self.monthly_rate, self.n_months
        if r == 0:
            return self.loan_principal / n
        return self.loan_principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


@dataclass
class AgentParams:
    monthly_income: float = 80_000
    monthly_consumption_min: float = 25_000
    monthly_maintenance: float = 7_000
    reserve_months: float = 3.0
    reserve_decay: float = 0.02
    theta: float = 0.45
    beta: float = 0.995
    alpha: float = 200.0
    gamma: float = 500.0
    utility_c: float = 1.0
    income_rho: float = 0.95
    income_sigma: float = 0.05
    savings_share: float = 0.10


@dataclass
class RentParams:
    rent_per_m2: float = 700
    rent_inflation: float = 0.06 / 12
    investment_rate: float = 0.08 / 12
    investment_sigma: float = 0.02
    investment_mode: str = "deterministic"


# 2. Стохастическая модель покупки

class StochasticStudioModel:
    def __init__(self, housing: HousingParams, agent: AgentParams, seed: int | None = None):
        self.h = housing
        self.a = agent
        self.rng = np.random.default_rng(seed)
        self.results = None

    def _income_path(self, n: int) -> np.ndarray:
        a = self.a
        log_y = np.zeros(n + 1)
        log_y[0] = np.log(a.monthly_income)
        for t in range(1, n + 1):
            log_y[t] = ((1 - a.income_rho) * np.log(a.monthly_income)
                        + a.income_rho * log_y[t - 1]
                        + a.income_sigma * self.rng.standard_normal())
        return np.exp(log_y)

    def _price_path(self, n: int) -> np.ndarray:
        h = self.h
        p = np.zeros(n + 1)
        p[0] = h.price_per_m2
        if h.price_mode == "deterministic":
            for t in range(1, n + 1):
                p[t] = p[t - 1] * (1 + h.inflation_m2)
        else:
            for t in range(1, n + 1):
                p[t] = p[t - 1] * np.exp(h.price_mu - 0.5 * h.price_sigma ** 2
                                         + h.price_sigma * self.rng.standard_normal())
        return p

    def simulate(self):
        h, a = self.h, self.a
        n = h.n_months
        M = h.annuity
        R = a.monthly_maintenance

        Y = self._income_path(n)
        p = self._price_path(n)

        debt = np.zeros(n + 1); debt[0] = h.loan_principal
        equity = np.zeros(n + 1); equity[0] = h.area * p[0] - debt[0]
        reserve = np.zeros(n + 1); reserve[0] = a.reserve_months * R
        payment_share = np.zeros(n + 1)
        breach_theta = np.zeros(n + 1, dtype=bool)
        breach_reserve = np.zeros(n + 1, dtype=bool)
        default_income = np.zeros(n + 1, dtype=bool)
        utility = np.zeros(n + 1)
        consumption = np.zeros(n + 1)
        consumption[0] = Y[0] - M - R

        for t in range(n):
            interest = debt[t] * h.monthly_rate
            debt[t + 1] = max(debt[t] - (M - interest), 0.0)
            equity[t + 1] = h.area * p[t + 1] - debt[t + 1]

            free = Y[t + 1] - M - R
            if free >= a.monthly_consumption_min:
                C = free * (1 - a.savings_share)
                S = free * a.savings_share
                default_income[t + 1] = False
            else:
                C = max(free, 1.0)
                S = 0.0
                default_income[t + 1] = True

            consumption[t + 1] = C
            reserve[t + 1] = (1 - a.reserve_decay) * reserve[t] + R + S

            payment_share[t + 1] = (M + R) / Y[t + 1]
            breach_theta[t + 1] = payment_share[t + 1] > a.theta
            breach_reserve[t + 1] = reserve[t + 1] < a.reserve_months * R

            u = a.utility_c * np.log(max(C, 1.0))
            penalty = a.gamma * float(breach_theta[t + 1] or breach_reserve[t + 1])
            default_pen = a.gamma * float(default_income[t + 1])
            utility[t + 1] = (a.beta ** (t + 1)) * (u + a.alpha - penalty - default_pen)

        self.results = {
            "months": np.arange(n + 1),
            "income": Y, "price_m2": p,
            "debt": debt, "equity": equity, "reserve": reserve,
            "payment_share": payment_share,
            "breach_theta": breach_theta, "breach_reserve": breach_reserve,
            "default_income": default_income,
            "utility": utility, "total_utility": utility.sum(),
            "consumption": consumption, "annuity": M,
        }
        return self.results

    def plot(self, out_path: str | None = None):
        if self.results is None:
            self.simulate()
        r = self.results
        fig, axes = plt.subplots(3, 2, figsize=(14, 11))

        axes[0, 0].plot(r["months"], r["income"] / 1000)
        axes[0, 0].axhline(self.a.monthly_income / 1000, color="gray", ls="--", label="средний доход")
        axes[0, 0].set_title("Доход, тыс. ₽"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(r["months"], r["price_m2"] / 1000)
        axes[0, 1].set_title("Цена м², тыс. ₽"); axes[0, 1].grid(alpha=0.3)

        axes[1, 0].plot(r["months"], r["debt"] / 1000, label="Долг")
        axes[1, 0].plot(r["months"], r["equity"] / 1000, label="Капитал")
        axes[1, 0].set_title("Долг и капитал, тыс. ₽"); axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

        axes[1, 1].plot(r["months"], r["payment_share"], label="(M+R)/Y")
        axes[1, 1].axhline(self.a.theta, color="red", ls="--", label=f"θ={self.a.theta}")
        axes[1, 1].set_title("Доля платежей в доходе"); axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

        axes[2, 0].plot(r["months"], r["reserve"] / 1000, label="резерв")
        axes[2, 0].axhline(self.a.reserve_months * self.a.monthly_maintenance / 1000,
                           color="red", ls="--", label="минимум")
        axes[2, 0].set_title("Фонд «плотины», тыс. ₽"); axes[2, 0].legend(); axes[2, 0].grid(alpha=0.3)

        axes[2, 1].plot(r["months"], np.cumsum(r["utility"]))
        axes[2, 1].set_title("Накопленная полезность"); axes[2, 1].grid(alpha=0.3)

        plt.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
        else:
            plt.show()


# 3. Монте-Карло для покупки

def monte_carlo(housing: HousingParams, agent: AgentParams,
                n_paths: int = 400, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    stats = {
        "total_utility": [], "p_breach_theta": [], "p_breach_reserve": [],
        "p_default_income": [], "final_equity": [], "payment_share_first": [],
    }
    for _ in range(n_paths):
        m = StochasticStudioModel(housing, agent, seed=int(rng.integers(1e9)))
        r = m.simulate()
        stats["total_utility"].append(r["total_utility"])
        stats["p_breach_theta"].append(r["breach_theta"][1:].mean())
        stats["p_breach_reserve"].append(r["breach_reserve"][1:].mean())
        stats["p_default_income"].append(r["default_income"][1:].mean())
        stats["final_equity"].append(r["equity"][-1])
        stats["payment_share_first"].append(r["payment_share"][1])
    return {k: np.array(v) for k, v in stats.items()}


def plot_mc(stats: dict, out_path: str | None = None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].hist(stats["total_utility"], bins=40)
    axes[0, 0].set_title("Распределение суммарной полезности"); axes[0, 0].grid(alpha=0.3)
    axes[0, 1].hist(stats["final_equity"] / 1000, bins=40)
    axes[0, 1].set_title("Итоговый капитал, тыс. ₽"); axes[0, 1].grid(alpha=0.3)
    axes[1, 0].hist(stats["p_breach_theta"], bins=40)
    axes[1, 0].set_title("Доля месяцев с (M+R)/Y > θ"); axes[1, 0].grid(alpha=0.3)
    axes[1, 1].hist(stats["p_default_income"], bins=40)
    axes[1, 1].set_title("Доля месяцев с дефицитом дохода"); axes[1, 1].grid(alpha=0.3)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120); plt.close(fig)
    else:
        plt.show()


# 4. Покупка vs аренда

def compare_buy_vs_rent(housing: HousingParams, agent: AgentParams,
                        rent: RentParams, n_paths: int = 400, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = housing.n_months
    M = housing.annuity
    R = agent.monthly_maintenance
    I0 = housing.down_payment

    out = {k: [] for k in [
        "wealth_buy", "wealth_rent", "utility_buy", "utility_rent",
        "final_equity_buy", "final_wealth_rent",
        "mean_consumption_buy", "mean_consumption_rent",
    ]}

    for _ in range(n_paths):
        m = StochasticStudioModel(housing, agent, seed=int(rng.integers(1e9)))
        Y = m._income_path(n)
        p = m._price_path(n)

        # покупатель
        debt = np.zeros(n + 1); debt[0] = housing.loan_principal
        equity = np.zeros(n + 1); equity[0] = housing.area * p[0] - debt[0]
        W_buy = np.zeros(n + 1); C_buy = np.zeros(n + 1); U_buy = np.zeros(n + 1)

        for t in range(n):
            interest = debt[t] * housing.monthly_rate
            debt[t + 1] = max(debt[t] - (M - interest), 0.0)
            equity[t + 1] = housing.area * p[t + 1] - debt[t + 1]
            free = Y[t + 1] - M - R
            W_buy[t + 1] = W_buy[t] * (1 + rent.investment_rate) + max(free, 0.0)
            C_buy[t + 1] = max(free, agent.monthly_consumption_min)
            U_buy[t + 1] = (agent.beta ** (t + 1)) * (
                agent.utility_c * np.log(max(C_buy[t + 1], 1.0)) + agent.alpha)

        # арендатор
        rent_t = np.zeros(n + 1); rent_t[0] = housing.area * rent.rent_per_m2
        for t in range(1, n + 1):
            rent_t[t] = rent_t[t - 1] * (1 + rent.rent_inflation)

        W_rent = np.zeros(n + 1); W_rent[0] = I0
        C_rent = np.zeros(n + 1); U_rent = np.zeros(n + 1)

        for t in range(n):
            delta = (M + R) - rent_t[t + 1]
            W_rent[t + 1] = W_rent[t] * (1 + rent.investment_rate) + delta
            free = Y[t + 1] - rent_t[t + 1]
            C_rent[t + 1] = max(free, agent.monthly_consumption_min)
            U_rent[t + 1] = (agent.beta ** (t + 1)) * (
                agent.utility_c * np.log(max(C_rent[t + 1], 1.0)))

        out["wealth_buy"].append(W_buy[-1] + equity[-1])
        out["wealth_rent"].append(W_rent[-1])
        out["final_equity_buy"].append(equity[-1])
        out["final_wealth_rent"].append(W_rent[-1])
        out["utility_buy"].append(U_buy.sum())
        out["utility_rent"].append(U_rent.sum())
        out["mean_consumption_buy"].append(C_buy[1:].mean())
        out["mean_consumption_rent"].append(C_rent[1:].mean())

    return {k: np.array(v) for k, v in out.items()}


def plot_buy_vs_rent(res: dict, out_path: str | None = None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].hist(res["wealth_buy"] / 1000, bins=40, alpha=0.6, label="Покупка")
    axes[0].hist(res["wealth_rent"] / 1000, bins=40, alpha=0.6, label="Аренда")
    axes[0].set_title("Итоговое благосостояние, тыс. ₽"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].hist(res["utility_buy"], bins=40, alpha=0.6, label="Покупка")
    axes[1].hist(res["utility_rent"], bins=40, alpha=0.6, label="Аренда")
    axes[1].set_title("Суммарная полезность"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].hist(res["mean_consumption_buy"], bins=40, alpha=0.6, label="Покупка")
    axes[2].hist(res["mean_consumption_rent"], bins=40, alpha=0.6, label="Аренда")
    axes[2].set_title("Среднее потребление, ₽/мес"); axes[2].legend(); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120); plt.close(fig)
    else:
        plt.show()


def summarize_comparison(res: dict) -> dict:
    def st(x):
        return dict(mean=float(x.mean()), p10=float(np.percentile(x, 10)),
                    p50=float(np.percentile(x, 50)), p90=float(np.percentile(x, 90)))
    summary = {
        "wealth_buy": st(res["wealth_buy"]),
        "wealth_rent": st(res["wealth_rent"]),
        "utility_buy": st(res["utility_buy"]),
        "utility_rent": st(res["utility_rent"]),
        "mean_consumption_buy": st(res["mean_consumption_buy"]),
        "mean_consumption_rent": st(res["mean_consumption_rent"]),
        "share_buy_richer": float((res["wealth_buy"] > res["wealth_rent"]).mean()),
        "share_buy_more_utility": float((res["utility_buy"] > res["utility_rent"]).mean()),
    }
    return summary


# 5. Точка безубыточности

def _mean_wealth_rent(h, a, rent_base, r_inv_annual, n_paths, seed):
    rent = RentParams(rent_per_m2=rent_base.rent_per_m2,
                      rent_inflation=rent_base.rent_inflation,
                      investment_rate=r_inv_annual / 12,
                      investment_mode=rent_base.investment_mode)
    res = compare_buy_vs_rent(h, a, rent, n_paths=n_paths, seed=seed)
    return res["wealth_rent"].mean(), res["utility_rent"].mean()


def _mean_buy(h, a, rent_base, n_paths, seed):
    rent = RentParams(rent_per_m2=rent_base.rent_per_m2,
                      rent_inflation=rent_base.rent_inflation,
                      investment_rate=rent_base.investment_rate,
                      investment_mode=rent_base.investment_mode)
    res = compare_buy_vs_rent(h, a, rent, n_paths=n_paths, seed=seed)
    return res["wealth_buy"].mean(), res["utility_buy"].mean()


def breakeven_investment_rate(h, a, rent_base, n_paths=300, seed=42,
                              r_lo=-0.05, r_hi=0.30) -> dict:
    W_buy, U_buy = _mean_buy(h, a, rent_base, n_paths, seed)

    def f_w(r):
        w, _ = _mean_wealth_rent(h, a, rent_base, r, n_paths, seed)
        return w - W_buy

    def f_u(r):
        _, u = _mean_wealth_rent(h, a, rent_base, r, n_paths, seed)
        return u - U_buy

    r_w = brentq(f_w, r_lo, r_hi, xtol=1e-4) if f_w(r_lo) * f_w(r_hi) <= 0 else None
    r_u = brentq(f_u, r_lo, r_hi, xtol=1e-4) if f_u(r_lo) * f_u(r_hi) <= 0 else None
    return {"W_buy": W_buy, "U_buy": U_buy, "r_star_wealth": r_w, "r_star_utility": r_u}


def breakeven_grid(base_h, base_a, base_rent, rate_grid, price_grid,
                   n_paths=150, seed=42, r_lo=-0.05, r_hi=0.35):
    W_star = np.full((len(rate_grid), len(price_grid)), np.nan)
    U_star = np.full_like(W_star, np.nan)
    for i, rate in enumerate(rate_grid):
        for j, price in enumerate(price_grid):
            h = HousingParams(area=base_h.area, price_per_m2=price,
                              down_payment_share=base_h.down_payment_share,
                              annual_rate=rate, years=base_h.years,
                              inflation_m2=base_h.inflation_m2,
                              price_mode=base_h.price_mode,
                              price_mu=base_h.price_mu, price_sigma=base_h.price_sigma)
            a = AgentParams(**asdict(base_a))
            rent = RentParams(**asdict(base_rent))
            W_buy, U_buy = _mean_buy(h, a, rent, n_paths, seed)

            def f_w(r):
                w, _ = _mean_wealth_rent(h, a, rent, r, n_paths, seed); return w - W_buy
            def f_u(r):
                _, u = _mean_wealth_rent(h, a, rent, r, n_paths, seed); return u - U_buy

            if f_w(r_lo) * f_w(r_hi) <= 0:
                W_star[i, j] = brentq(f_w, r_lo, r_hi, xtol=1e-3)
            if f_u(r_lo) * f_u(r_hi) <= 0:
                U_star[i, j] = brentq(f_u, r_lo, r_hi, xtol=1e-3)
    return W_star, U_star


def plot_breakeven_maps(rate_grid, price_grid, W_star, U_star, out_path=None):
    R, P = np.meshgrid(rate_grid * 100, price_grid / 1000, indexing="ij")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, Z, title in [
        (axes[0], W_star * 100, "r_inv* по благосостоянию, % годовых"),
        (axes[1], U_star * 100, "r_inv** по полезности, % годовых"),
    ]:
        im = ax.imshow(Z, origin="lower", aspect="auto",
                       extent=[P.min(), P.max(), R.min(), R.max()], cmap="viridis")
        cs = ax.contour(P, R, Z, levels=6, colors="white", linewidths=0.8)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")
        ax.set_xlabel("Цена м², тыс. ₽"); ax.set_ylabel("Ставка ипотеки, %")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="% годовых")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120); plt.close(fig)
    else:
        plt.show()


def breakeven_vs_rent(base_h, base_a, base_rent, rent_values, n_paths=200, seed=42):
    r_star = []
    for rp in rent_values:
        rent = RentParams(rent_per_m2=rp,
                          rent_inflation=base_rent.rent_inflation,
                          investment_rate=base_rent.investment_rate,
                          investment_mode=base_rent.investment_mode)
        W_buy, _ = _mean_buy(base_h, base_a, rent, n_paths, seed)
        def f(r):
            w, _ = _mean_wealth_rent(base_h, base_a, rent, r, n_paths, seed)
            return w - W_buy
        r_star.append(brentq(f, -0.05, 0.35, xtol=1e-3)
                      if f(-0.05) * f(0.35) <= 0 else np.nan)
    return np.array(r_star)


def plot_breakeven_vs_rent(rent_values, r_star, out_path=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rent_values, r_star * 100, marker="o")
    ax.set_xlabel("Аренда, ₽/м²/мес"); ax.set_ylabel("r_inv*, % годовых")
    ax.set_title("Точка безубыточности по благосостоянию vs ставка аренды")
    ax.grid(alpha=0.3); plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120); plt.close(fig)
    else:
        plt.show()


# 6. CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Модель жилищного бюджета (студия) + сравнение покупка/аренда.")
    p.add_argument("--out", default="output", help="Папка для графиков и CSV")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-paths", type=int, default=300)
    p.add_argument("--area", type=float, default=24.0)
    p.add_argument("--price-m2", type=float, default=120_000)
    p.add_argument("--down-payment", type=float, default=0.20)
    p.add_argument("--rate", type=float, default=0.12, help="Годовая ставка ипотеки")
    p.add_argument("--years", type=int, default=20)
    p.add_argument("--income", type=float, default=80_000)
    p.add_argument("--maintenance", type=float, default=7_000)
    p.add_argument("--theta", type=float, default=0.45)
    p.add_argument("--rent-m2", type=float, default=700)
    p.add_argument("--rent-inflation", type=float, default=0.06, help="Годовая")
    p.add_argument("--investment-rate", type=float, default=0.08, help="Годовая")
    p.add_argument("--price-mode", choices=["deterministic", "stochastic"],
                   default="deterministic")
    p.add_argument("--mode", choices=["basic", "mc", "compare", "breakeven",
                                      "grid", "rent-sensitivity", "all"],
                   default="all")
    return p


def main():
    args = build_parser().parse_args()
    os.makedirs(args.out, exist_ok=True)

    h = HousingParams(area=args.area, price_per_m2=args.price_m2,
                      down_payment_share=args.down_payment,
                      annual_rate=args.rate, years=args.years,
                      price_mode=args.price_mode)
    a = AgentParams(monthly_income=args.income,
                    monthly_maintenance=args.maintenance, theta=args.theta)
    rent = RentParams(rent_per_m2=args.rent_m2,
                      rent_inflation=args.rent_inflation / 12,
                      investment_rate=args.investment_rate / 12)

    print(f"[i] Цена студии: {h.price:,.0f} ₽")
    print(f"[i] Взнос: {h.down_payment:,.0f} ₽, кредит: {h.loan_principal:,.0f} ₽")
    print(f"[i] Аннуитет: {h.annuity:,.0f} ₽/мес")

    summary_rows = []

    if args.mode in ("basic", "all"):
        m = StochasticStudioModel(h, a, seed=args.seed)
        r = m.simulate()
        m.plot(os.path.join(args.out, "01_basic_path.png"))
        summary_rows.append({
            "section": "basic",
            "metric": "payment_share_first",
            "value": float(r["payment_share"][1]),
        })
        summary_rows.append({
            "section": "basic",
            "metric": "total_utility",
            "value": float(r["total_utility"]),
        })

    if args.mode in ("mc", "all"):
        mc = monte_carlo(h, a, n_paths=args.n_paths, seed=args.seed)
        plot_mc(mc, os.path.join(args.out, "02_mc.png"))
        np.savez(os.path.join(args.out, "02_mc_stats.npz"), **mc)
        summary_rows += [
            {"section": "mc", "metric": "total_utility_mean", "value": float(mc["total_utility"].mean())},
            {"section": "mc", "metric": "p_breach_theta_mean", "value": float(mc["p_breach_theta"].mean())},
            {"section": "mc", "metric": "p_default_income_mean", "value": float(mc["p_default_income"].mean())},
            {"section": "mc", "metric": "final_equity_mean", "value": float(mc["final_equity"].mean())},
        ]

    if args.mode in ("compare", "all"):
        comp = compare_buy_vs_rent(h, a, rent, n_paths=args.n_paths, seed=args.seed)
        plot_buy_vs_rent(comp, os.path.join(args.out, "03_buy_vs_rent.png"))
        s = summarize_comparison(comp)
        np.savez(os.path.join(args.out, "03_compare_stats.npz"), **comp)
        summary_rows += [
            {"section": "compare", "metric": "wealth_buy_mean", "value": s["wealth_buy"]["mean"]},
            {"section": "compare", "metric": "wealth_rent_mean", "value": s["wealth_rent"]["mean"]},
            {"section": "compare", "metric": "share_buy_richer", "value": s["share_buy_richer"]},
            {"section": "compare", "metric": "share_buy_more_utility", "value": s["share_buy_more_utility"]},
        ]

    if args.mode in ("breakeven", "all"):
        be = breakeven_investment_rate(h, a, rent, n_paths=args.n_paths, seed=args.seed)
        print(f"[i] r_inv* (благосостояние): "
              f"{'n/a' if be['r_star_wealth'] is None else f'{be[chr(114)+chr(95)+chr(115)+chr(116)+chr(97)+chr(114)+chr(95)+chr(119)+chr(101)+chr(97)+chr(108)+chr(116)+chr(104)]*100:.2f}%'}")
        summary_rows += [
            {"section": "breakeven", "metric": "W_buy", "value": be["W_buy"]},
            {"section": "breakeven", "metric": "U_buy", "value": be["U_buy"]},
            {"section": "breakeven", "metric": "r_star_wealth",
             "value": None if be["r_star_wealth"] is None else be["r_star_wealth"]},
            {"section": "breakeven", "metric": "r_star_utility",
             "value": None if be["r_star_utility"] is None else be["r_star_utility"]},
        ]
        pd.DataFrame([{
            "W_buy": be["W_buy"], "U_buy": be["U_buy"],
            "r_star_wealth_%": None if be["r_star_wealth"] is None else be["r_star_wealth"] * 100,
            "r_star_utility_%": None if be["r_star_utility"] is None else be["r_star_utility"] * 100,
        }]).to_csv(os.path.join(args.out, "04_breakeven.csv"), index=False)

    if args.mode in ("grid", "all"):
        rate_grid = np.array([0.08, 0.12, 0.16, 0.20, 0.24])
        price_grid = np.array([80_000, 100_000, 120_000, 140_000, 160_000])

        W_star, U_star = breakeven_grid(
            h, a, rent, rate_grid, price_grid,
            n_paths=max(80, args.n_paths // 2), seed=args.seed,
        )
        plot_breakeven_maps(rate_grid, price_grid, W_star, U_star,
                            os.path.join(args.out, "05_breakeven_grid.png"))

        # таблица сетки
        rows = []
        for i, rate in enumerate(rate_grid):
            for j, price in enumerate(price_grid):
                rows.append({
                    "rate_%": round(rate * 100, 1),
                    "price_k_rub_m2": round(price / 1000),
                    "r_star_wealth_%": None if np.isnan(W_star[i, j])
                    else round(W_star[i, j] * 100, 2),
                    "r_star_utility_%": None if np.isnan(U_star[i, j])
                    else round(U_star[i, j] * 100, 2),
                })
        pd.DataFrame(rows).to_csv(
            os.path.join(args.out, "05_breakeven_grid.csv"), index=False
        )
        summary_rows.append({
            "section": "grid",
            "metric": "r_star_wealth_mean_%",
            "value": float(np.nanmean(W_star) * 100),
        })

    if args.mode in ("rent-sensitivity", "all"):
        rent_values = np.array([500, 700, 900, 1100, 1300])
        r_star_rent = breakeven_vs_rent(
            h, a, rent, rent_values,
            n_paths=max(80, args.n_paths // 2), seed=args.seed,
        )
        plot_breakeven_vs_rent(rent_values, r_star_rent,
                                os.path.join(args.out, "06_breakeven_vs_rent.png"))
        pd.DataFrame({
            "rent_rub_m2": rent_values,
            "r_star_wealth_%": r_star_rent * 100,
        }).to_csv(os.path.join(args.out, "06_breakeven_vs_rent.csv"), index=False)
        summary_rows.append({
            "section": "rent-sensitivity",
            "metric": "r_star_at_rent_700_%",
            "value": float(r_star_rent[1] * 100) if not np.isnan(r_star_rent[1]) else None,
        })


# Сводка

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(args.out, "06_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[i] Сводка сохранена: {summary_path}")
        print(summary_df.to_string(index=False))

    print(f"\n[i] Все результаты в папке: {os.path.abspath(args.out)}")


# 7. Точка входа

if __name__ == "__main__":
    main()
