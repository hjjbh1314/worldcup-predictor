# 接入实时赔率指南(The Odds API 免费档,0 元)

把博彩赔率叠加到预测里——这是唯一能真正超越纯 Elo 的信息源。
本指南用 [The Odds API](https://the-odds-api.com/) 的**免费档**,全程不花钱。

---

## 为什么免费档够用

- 免费档:**500 credits / 月**
- 只取 `h2h`(胜平负)市场 + 单一地区(`eu`)时,**一次调用 = 1 credit**
- 一次调用就返回该项赛事**全部**即将开赛的比赛
- 即使每天拉一次更新:30 天 ≈ **30 credits**,离 500 还差得远

> credit ≠ request:每次调用消耗的 credit = 市场数 × 地区数。
> 所以保持 `markets=h2h`、`regions=eu`(单一)就能把消耗压到最低。

---

## 三步上手

### 1. 注册拿 key(2 分钟)
打开 https://the-odds-api.com/ → 点 **Get API Key** → 填邮箱 → 选 **Free** 档 →
邮箱里会收到一串 key(形如 `a1b2c3d4...`)。
> 注册需要你本人用邮箱完成,代码里我已留好 key 的位置,你只管拿到 key。

### 2. 设置环境变量(不要把 key 写进代码)
```bash
export ODDS_API_KEY=你的key            # 临时,当前终端有效
# 想长期生效,加进 ~/.zshrc:
echo 'export ODDS_API_KEY=你的key' >> ~/.zshrc && source ~/.zshrc
```

### 3. 运行
```bash
python -m scripts.predict_with_odds
```
输出:每场 **Elo 概率 vs 市场概率** 并排对比,标出二者分歧最大的比赛,
完整结果存到 `outputs/wc2026_with_odds.csv`。
未设置 key 时脚本自动退回纯 Elo,不会报错。

---

## 先确认世界杯赛事 key 是否已上线

赔率通常在开赛前几天才挂出。临场前可以先查一下可用的足球赛事 key:
```bash
python -c "from src import odds, os; \
import json; print(json.dumps(odds.list_soccer_sports(os.environ['ODDS_API_KEY']), \
ensure_ascii=False, indent=2))"
```
找名字里带 `world_cup` 的 `key`(预期是 `soccer_fifa_world_cup`)。
若与默认值不同,把 `scripts/predict_with_odds.py` 里 `sport_key=` 改成查到的值即可。

---

## 两个常见坑

1. **队名对不上**:The Odds API 用 `USA`,本数据集用 `United States`。
   已在 `src/odds.py` 的 `TEAM_ALIASES` 里做了映射,遇到没匹配上的队名,
   往这个字典里补一行即可(键=Odds API 的名字,值=数据集的名字)。

2. **省 credit**:别把 `regions` 写成 `us,uk,eu` 这种多地区——会成倍消耗 credit。
   单 `eu` 已能拿到 Pinnacle、Bet365 等主流盘口,足够算共识概率。

---

## 怎么用这个对比(研究价值所在)

脚本里的 `edge_home = Elo主胜概率 − 市场主胜概率`:
- **edge 接近 0**:模型与市场一致,没什么可说的
- **edge 明显为正**:模型比市场更看好主队 → 要么模型抓到了市场忽略的信息,要么模型错了
- **edge 明显为负**:反之

诚实预期:绝大多数比赛 edge 都很小(市场很有效)。真正值得研究的是那几场分歧大的——
但**不构成任何下注建议**,纯属模型与市场的对照实验。
