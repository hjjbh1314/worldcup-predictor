"""The Odds API(免费档)赔率接入 + 去水后转隐含概率。

免费档:500 credits/月。只取 h2h(胜平负)市场 + 单一地区时,每次调用 1 credit;
一次调用即返回该项赛事全部即将开赛的比赛,每天拉一次一个月也才 ~30 credits。

环境变量:ODDS_API_KEY  —— 不要把 key 写进代码或提交到 git。
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

API_BASE = "https://api.the-odds-api.com/v4"

# The Odds API 队名 → 本数据集队名(按需补充)。
# 例:Odds API 常用 "USA",本数据集用 "United States"。
TEAM_ALIASES = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    "South Korea": "South Korea",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Czechia": "Czech Republic",
}


def canon(name: str) -> str:
    """归一化队名以便和数据集匹配。"""
    return TEAM_ALIASES.get(name, name)


def _get(url: str) -> list | dict:
    req = urllib.request.Request(url, headers={"User-Agent": "worldcup-predictor"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_soccer_sports(api_key: str) -> list[dict]:
    """列出当前可用的足球赛事 key(找 soccer_fifa_world_cup 等)。"""
    data = _get(f"{API_BASE}/sports?apiKey={api_key}")
    return [s for s in data if str(s.get("group", "")).lower() == "soccer"]


def fetch_odds(api_key: str, sport_key: str = "soccer_fifa_world_cup",
               regions: str = "eu", markets: str = "h2h",
               odds_format: str = "decimal") -> list[dict]:
    """拉取某项赛事的赔率。返回事件列表(含 bookmakers)。"""
    q = urllib.parse.urlencode({
        "apiKey": api_key, "regions": regions,
        "markets": markets, "oddsFormat": odds_format,
    })
    return _get(f"{API_BASE}/sports/{sport_key}/odds?{q}")


def event_consensus_probs(event: dict) -> dict[str, float] | None:
    """单场比赛:对各家博彩去水(归一化)后取共识隐含概率。

    返回 {队名: 概率, 'Draw': 概率};按 1/赔率 归一化消除返还率(vig)。
    """
    per_book = []
    for bk in event.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") != "h2h":
                continue
            prices = {o["name"]: float(o["price"]) for o in m.get("outcomes", [])
                      if o.get("price")}
            if len(prices) < 2:
                continue
            inv = {k: 1.0 / v for k, v in prices.items()}
            s = sum(inv.values())
            per_book.append({k: inv[k] / s for k in inv})
    if not per_book:
        return None
    keys = set().union(*per_book)
    n = len(per_book)
    return {canon(k) if k != "Draw" else "Draw":
            sum(d.get(k, 0.0) for d in per_book) / n for k in keys}


def build_odds_index(events: list[dict]) -> dict[frozenset, dict]:
    """把事件按 {两支球队} 建索引,便于和赛程匹配(不依赖主客顺序)。"""
    idx = {}
    for ev in events:
        probs = event_consensus_probs(ev)
        if not probs:
            continue
        teams = frozenset(canon(t) for t in (ev.get("home_team"), ev.get("away_team")) if t)
        if len(teams) == 2:
            idx[teams] = probs
    return idx


def get_api_key() -> str | None:
    return os.environ.get("ODDS_API_KEY")
