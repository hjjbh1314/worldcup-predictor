"""下载国家队比赛数据集(martj42/international_results,公开免费)。

用法:
    python -m scripts.download_data
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
FILES = ["results.csv", "goalscorers.csv", "shootouts.csv", "former_names.csv"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    for f in FILES:
        url = f"{BASE}/{f}"
        dst = DATA_DIR / f
        print(f"下载 {url} → {dst}")
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception as e:                       # noqa: BLE001
            print(f"  失败:{e}", file=sys.stderr)
    print("完成。")


if __name__ == "__main__":
    main()
