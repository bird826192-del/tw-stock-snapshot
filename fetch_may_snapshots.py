#!/usr/bin/env python3
"""
本機執行腳本：從 FinMind 抓取 2026年5月 台股資料，
計算各指標後產生每個交易日的 snapshot，commit 進 git repo。

使用方式：
  pip install requests pandas numpy
  python3 fetch_may_snapshots.py

如有 FinMind token 可設環境變數加速（免費也可用）：
  export FINMIND_TOKEN=your_token_here
"""

import os
import json
import time
import subprocess
import requests
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

# ── 設定 ────────────────────────────────────────────────────────────────────
FINMIND_API = "https://api.finmindtrade.com/api/v4/data"
TOKEN = os.environ.get("FINMIND_TOKEN", "")

STOCK_IDS = [
    "2330","2317","2454","2308","2382","2412","2891","1303","1301","2881",
    "2882","6505","3711","2002","2603","2884","1216","2886","2885","2892",
    "2880","5880","2883","2890","2207","2357","3008","2379","4938","2409",
    "3034","2327","2474","1326","2303","2105","1101","2912","2801","0050",
    "1102","1103","1104","1108","1109","1110","1201","1203","1210","1213",
    "1215","1217","1218","1219","1220","1225","1227","1229","1231","1232",
    "1233","1234","1235","1236","1256","1304","1305","1307","1308","1309",
    "1310","1312","1313","1314","1315","1316","1319","1321","1323","1324",
    "1325","1337","1338","1339","1340","1341","1342","1402","1409","1410",
    "1413","1414","1416","1417","1418","1419"
]

# 需要從 2月初開始抓，以便計算 MA60 和 KD
DATA_START = "2026-02-01"
DATA_END   = "2026-05-31"
MAY_DATES  = pd.bdate_range("2026-05-01", "2026-05-31")  # 台股交易日（近似）

# ── FinMind API ──────────────────────────────────────────────────────────────
def finmind_get(dataset, stock_id, start=DATA_START, end=DATA_END):
    params = {
        "dataset": dataset,
        "data_id": stock_id,
        "start_date": start,
        "end_date": end,
        "token": TOKEN,
    }
    for attempt in range(3):
        try:
            r = requests.get(FINMIND_API, params=params, timeout=20)
            if r.status_code == 200:
                data = r.json()
                if data.get("status") == 200:
                    return pd.DataFrame(data["data"])
            print(f"  API error {r.status_code} for {stock_id} {dataset}")
            time.sleep(2)
        except Exception as e:
            print(f"  Request error: {e}")
            time.sleep(2)
    return pd.DataFrame()

# ── 指標計算 ─────────────────────────────────────────────────────────────────
def calc_kd(df, period=9):
    """計算 KD 隨機指標（RSV period=9，K/D smoothing=3）"""
    low_min  = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()
    rsv = (df["close"] - low_min) / (high_max - low_min + 1e-9) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    return k, d

def calc_ma(df, window):
    return df["close"].rolling(window).mean()

def slope_up(series, days=5):
    """最近 days 天的斜率是否為正"""
    if len(series) < days:
        return False
    sub = series.iloc[-days:]
    return float(sub.iloc[-1]) > float(sub.iloc[0])

def drawdown30(df):
    """從近30日最高點的回檔百分比"""
    h = df["high"].rolling(30).max()
    return ((h - df["close"]) / h * 100).clip(lower=0)

# ── 主流程 ────────────────────────────────────────────────────────────────────
def fetch_all():
    print("═" * 60)
    print("Step 1: 抓取每支股票的價格與三大法人資料...")
    print("═" * 60)

    price_all    = {}  # stock_id -> DataFrame (date, open, high, low, close, volume)
    dealer_all   = {}  # stock_id -> DataFrame (date, dealer_buy, dealer_sell)

    for i, sid in enumerate(STOCK_IDS):
        print(f"[{i+1}/{len(STOCK_IDS)}] {sid} 抓取中...")

        # 日成交資料
        df_p = finmind_get("TaiwanStockPrice", sid)
        if not df_p.empty:
            df_p["date"] = pd.to_datetime(df_p["date"])
            df_p = df_p.sort_values("date").reset_index(drop=True)
            price_all[sid] = df_p

        # 三大法人（自營商）
        df_d = finmind_get("TaiwanStockInstitutionalInvestorsBuySell", sid)
        if not df_d.empty:
            df_d["date"] = pd.to_datetime(df_d["date"])
            df_d = df_d.sort_values("date").reset_index(drop=True)
            dealer_all[sid] = df_d

        time.sleep(0.25)  # rate limit

    return price_all, dealer_all


def build_snapshots(price_all, dealer_all):
    print("\n═" * 60)
    print("Step 2: 計算指標 & 產生每日 snapshot...")
    print("═" * 60)

    # 取出實際有交易的日期（以台積電為準）
    ref = price_all.get("2330", pd.DataFrame())
    if ref.empty:
        print("ERROR: 找不到台積電價格資料")
        return {}

    may_trading_dates = ref[
        (ref["date"] >= "2026-05-01") & (ref["date"] <= "2026-05-31")
    ]["date"].tolist()

    snapshots = {}

    for target_date in may_trading_dates:
        date_str = target_date.strftime("%Y-%m-%d")
        stocks_out = []

        for rank, sid in enumerate(STOCK_IDS, start=1):
            df_p = price_all.get(sid)
            if df_p is None or df_p.empty:
                continue

            # 只取到 target_date 的資料
            hist = df_p[df_p["date"] <= target_date].copy()
            if len(hist) < 2:
                continue

            today = hist.iloc[-1]
            prev  = hist.iloc[-2]

            close      = float(today["close"])
            change_pct = round((close - float(prev["close"])) / float(prev["close"]) * 100, 2)

            # KD
            if len(hist) >= 9:
                k_series, d_series = calc_kd(hist)
                k_val = round(float(k_series.iloc[-1]), 2)
                d_val = round(float(d_series.iloc[-1]), 2)
            else:
                k_val = d_val = None

            # MA
            ma20 = calc_ma(hist, 20)
            ma60 = calc_ma(hist, 60)
            ma20_val = float(ma20.iloc[-1]) if not pd.isna(ma20.iloc[-1]) else None
            ma60_val = float(ma60.iloc[-1]) if not pd.isna(ma60.iloc[-1]) else None

            close_above_ma60  = bool(close > ma60_val) if ma60_val else False
            ma20_slope5_up    = slope_up(ma20, 5) if ma20_val else False
            ma60_slope5_up    = slope_up(ma60, 5) if ma60_val else False
            trend_ok          = close_above_ma60 and ma20_slope5_up and ma60_slope5_up

            # maUpDays（連續 MA20 上升天數）
            up_days = 0
            for j in range(len(ma20)-2, -1, -1):
                if not pd.isna(ma20.iloc[j]) and not pd.isna(ma20.iloc[j+1]):
                    if float(ma20.iloc[j+1]) > float(ma20.iloc[j]):
                        up_days += 1
                    else:
                        break
                else:
                    break

            # drawdown30
            dd30_series = drawdown30(hist)
            dd30 = round(float(dd30_series.iloc[-1]), 2) if not pd.isna(dd30_series.iloc[-1]) else None

            # 自營商資料
            df_d = dealer_all.get(sid, pd.DataFrame())
            dealer_net    = None
            dealer_buy_pct = None
            dealer_holdings = None

            if not df_d.empty:
                # 篩選自營商
                dealer_rows = df_d[
                    (df_d["date"] <= target_date) &
                    (df_d.get("name", pd.Series(dtype=str)) == "自營商")
                ] if "name" in df_d.columns else df_d[df_d["date"] <= target_date]

                if not dealer_rows.empty:
                    # 最近5日
                    last5 = dealer_rows.tail(5)
                    buy_col  = next((c for c in ["buy","Buy","buy_volume","proprietary_dealer_buy"] if c in last5.columns), None)
                    sell_col = next((c for c in ["sell","Sell","sell_volume","proprietary_dealer_sell"] if c in last5.columns), None)

                    if buy_col and sell_col:
                        buy5  = float(last5[buy_col].sum())
                        sell5 = float(last5[sell_col].sum())
                        net5_shares = buy5 - sell5
                        # dealerNet 以千元為單位
                        dealer_net = round(net5_shares * close / 1000)

                    # dealerBuyPct：近5日買進 / 持股
                    today_dealer = dealer_rows[dealer_rows["date"] == target_date]
                    if not today_dealer.empty and buy_col:
                        today_buy = float(today_dealer[buy_col].iloc[0])
                        holdings_col = next((c for c in ["hold","holdings","hold_volume"] if c in today_dealer.columns), None)
                        if holdings_col:
                            holdings = float(today_dealer[holdings_col].iloc[0])
                            if holdings > 0:
                                dealer_buy_pct = round(today_buy / holdings * 100, 2)
                                dealer_holdings = round(holdings / 1000)  # 換算成張

            # dealerHoldings 備援推算（如直接持股欄位不存在）
            if dealer_holdings is None and dealer_buy_pct and dealer_buy_pct > 0 and dealer_net and dealer_net > 0 and close > 0:
                dealer_holdings = round(dealer_net * 100 / (close * dealer_buy_pct))

            # kdGolden（第二個篩選條件）
            kd_golden = bool(
                trend_ok and
                dealer_buy_pct is not None and dealer_buy_pct > 5 and
                dealer_holdings is not None and dealer_holdings > 5000 and
                dd30 is not None and dd30 > 5 and
                k_val is not None and k_val < 35 and
                d_val is not None and d_val < 40
            )

            stocks_out.append({
                "id":               sid,
                "name":             str(today.get("stock_name", sid)),
                "rank":             rank,
                "close":            close,
                "changePct":        change_pct,
                "k":                k_val,
                "d":                d_val,
                "dealerNet":        dealer_net or 0,
                "date":             date_str,
                "maUpDays":         up_days,
                "drawdown30":       dd30,
                "dealerBuyPct":     dealer_buy_pct,
                "dealerHoldings":   dealer_holdings,
                "volSpike":         None,
                "trendOK":          trend_ok,
                "volSpike5":        None,
                "high20Break":      False,
                "foreignNet5":      None,
                "retailDropWoW":    None,
                "ma20Slope5Up":     ma20_slope5_up,
                "ma60Slope5Up":     ma60_slope5_up,
                "closeAboveMa60":   close_above_ma60,
                "aboveBreakoutLow": False,
                "kdGolden":         kd_golden,
            })

        snapshots[date_str] = {
            "generatedAt": f"{date_str}T18:00:00+08:00",
            "tradingDate": date_str,
            "stocks": stocks_out,
        }

        # 統計
        kd_hits = [s for s in stocks_out if s["kdGolden"]]
        print(f"  {date_str}: {len(stocks_out)} 檔, kdGolden={len(kd_hits)} 筆 {[s['id']+' '+s['name'] for s in kd_hits]}")

    return snapshots


def commit_snapshots(snapshots):
    print("\n═" * 60)
    print("Step 3: Commit 每日 snapshot 進 git...")
    print("═" * 60)

    for date_str in sorted(snapshots.keys()):
        snap = snapshots[date_str]
        with open("snapshot.json", "w") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)

        subprocess.run(["git", "add", "snapshot.json"], check=True)
        subprocess.run([
            "git", "commit", "-m", f"data: snapshot {date_str}"
        ], check=True)
        print(f"  Committed {date_str}")

    print("\n完成！執行以下指令推送：")
    print("  git push -u origin claude/stock-filter-criteria-9rwic3")


if __name__ == "__main__":
    price_all, dealer_all = fetch_all()
    snapshots = build_snapshots(price_all, dealer_all)
    if snapshots:
        commit_snapshots(snapshots)
    else:
        print("沒有產生任何 snapshot，請確認網路可存取 FinMind API")
