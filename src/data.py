"""数据加载与清洗。

数据来源:martj42/international_results(1872 至今的国家队比赛结果)。
results.csv 字段:date, home_team, away_team, home_score, away_score,
                  tournament, city, country, neutral
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _apply_former_names(df: pd.DataFrame, former_path: Path) -> pd.DataFrame:
    """把历史队名归一到当前队名(如 West Germany -> Germany),保证 Elo 评分连续。

    former_names.csv 字段:current, former, start_date, end_date
    只在该队名处于其历史有效期内时替换。
    """
    fn = pd.read_csv(former_path, parse_dates=["start_date", "end_date"])
    rules = list(fn.itertuples(index=False))
    formers = set(fn["former"])

    def fix(name: str, date) -> str:
        for r in rules:
            if name == r.former and r.start_date <= date <= r.end_date:
                return r.current
        return name

    for col in ("home_team", "away_team"):
        mask = df[col].isin(formers)
        if mask.any():
            df.loc[mask, col] = [
                fix(n, d) for n, d in zip(df.loc[mask, col], df.loc[mask, "date"])
            ]
    return df


def load_results(path: str | Path | None = None,
                 apply_former_names: bool = True) -> pd.DataFrame:
    """加载比赛结果,返回按日期排序的 DataFrame。

    新增列:
      played  : 该场是否已踢完(有比分)
      result  : 'H'(主胜)/ 'D'(平)/ 'A'(客胜),未踢则为 NaN
    """
    path = Path(path) if path else DATA_DIR / "results.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    if apply_former_names:
        fn_path = DATA_DIR / "former_names.csv"
        if fn_path.exists():
            df = _apply_former_names(df, fn_path)

    df = df.sort_values("date").reset_index(drop=True)
    df["played"] = df["home_score"].notna() & df["away_score"].notna()
    df["result"] = np.where(
        df["home_score"] > df["away_score"], "H",
        np.where(df["home_score"] < df["away_score"], "A", "D"),
    )
    df.loc[~df["played"], "result"] = np.nan
    return df


if __name__ == "__main__":
    d = load_results()
    print(f"总场次 {len(d)} | {d.date.min().date()} → {d.date.max().date()}")
    print(d[d.played].result.value_counts(normalize=True).round(3).to_string())
