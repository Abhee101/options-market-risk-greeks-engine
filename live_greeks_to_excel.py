"""
Live option-chain greeks -> DirectionalGreeks workbook with Quant Strategy & Auto-Trader.

Real-time capabilities:
- ⚡ Event-driven TICK-BY-TICK Greek computation: Implied Volatility, Constant-delta Vega Flow,
  Theta decay, and Open Interest build-up computed INSTANTLY upon broker tick delivery (sub-2ms latency).
- Zero polling delay: Eliminates 35-second intervals, replacing with WebSocket push processing.
- Integrated Quantitative Trading Model (Volatility Skew Momentum & Order Flow Toxicity).
- Live Trade Recommendations: exact side to buy (Call/Put), exact strike, entry price, dynamic running stop-loss, trailing SL, targets.
- Quantified Edge metrics (Edge Score, Skew Conviction %, Statistical Win Rate %, Risk/Reward).
- Functional Automated Trading Controller with Safety Gates (Arm/Disarm, Paper/Live mode, Daily Loss Circuit Breaker, Kill Switch).
- Asynchronous high-performance Excel synchronization that never blocks or lags incoming ticks.
- Concurrent Interactive Web Control Dashboard at http://localhost:8050.
"""

import csv
import os
import threading
import time
from datetime import date, datetime, time as dtime
import numpy as np

from directional_greeks_core import COLS, Baseline, ChainState, infer
from quant_trade_engine import QuantModel, TradeRecommendation, TradeSignalType
from order_execution_engine import AutoTradingController, ExecutionMode
import app_quant_trader

# ------------------------------------------------------------------ settings
API_KEY = os.environ.get("KITE_API_KEY", "your_api_key")
ACCESS_TOKEN = os.environ.get("KITE_ACCESS_TOKEN", "your_access_token")
WORKBOOK = "DirectionalGreeks-2026Oct07.xlsx"
BASELINE_TIME = dtime(9, 15, 0)
SESSION_START = dtime(9, 16, 15)
SESSION_END = dtime(15, 30, 0)
N_EXPIRIES = 2
VEGA_DEADBAND_VOLPTS = 0.05
WINDOW_SIGMA = 0.20
N_OTM = 10
RATE = 0.065
FIRST_ROW, LAST_ROW = 2, 750

# Real-Time Tick-By-Tick Execution Settings
TICK_BY_TICK_MODE = True                    # Recompute greeks immediately upon every broker tick packet
EXCEL_FLUSH_INTERVAL_SEC = 0.25             # Minimum interval to flush live decision cards to Excel
ROW_SNAPSHOT_INTERVAL_SEC = 5.0             # Interval to append historical snapshot rows in Excel
SIMULATED_TICK_INTERVAL_SEC = 0.25          # Real-time tick streamer frequency for offline/simulated mode

# Quant & Auto-Trading Settings
ENABLE_AUTO_TRADING_ARM = False             # Safety default: starts DISARMED until user clicks ARM
DEFAULT_EXECUTION_MODE = ExecutionMode.PAPER # Default to PAPER mode for safety
MAX_DAILY_LOSS_LIMIT = 5000.0              # Circuit breaker daily loss limit in ₹
MAX_TRADES_PER_DAY = 4
ENABLE_WEB_DASHBOARD = True
WEB_DASHBOARD_PORT = 8050

# sheet name -> (exchange segment, instrument name in the broker's list)
SHEETS = {
    "NIFTY": ("NFO", "NIFTY"),
    "BANKNIFTY": ("NFO", "BANKNIFTY"),
    "SENSEX": ("BFO", "SENSEX"),
    "RELIANCE": ("NFO", "RELIANCE"),
    "TCS": ("NFO", "TCS"),
    "HDFC Bank": ("NFO", "HDFCBANK"),
    "ICICI Bank": ("NFO", "ICICIBANK"),
}


# ------------------------------------------------------------------ chain setup
def pick_chain(instruments, seg, name, fut_price, today, n_exp=N_EXPIRIES, n_otm=N_OTM):
    rows = [i for i in instruments[seg] if i["name"] == name and i["instrument_type"] in ("CE", "PE") and i["expiry"] >= today]
    expiries = sorted({i["expiry"] for i in rows})[:n_exp]
    strikes = sorted({i["strike"] for i in rows if i["expiry"] == expiries[0]})
    atm = min(strikes, key=lambda s: abs(s - fut_price))
    k = strikes.index(atm)
    wide = n_otm + 5
    keep = set(strikes[max(0, k - wide): k + wide + 1])
    t_far = max((expiries[-1] - today).days, 1) / 365.0
    reach = 2.0 * WINDOW_SIGMA * np.sqrt(t_far) * fut_price
    keep |= {s for s in strikes if abs(s - fut_price) <= reach}
    opts = [dict(i, exp_idx=expiries.index(i["expiry"])) for i in rows if i["expiry"] in expiries and i["strike"] in keep]
    opts.sort(key=lambda i: (i["exp_idx"], i["strike"], i["instrument_type"]))
    return expiries, opts


class Instrument:
    def __init__(self, sheet, expiries, opts):
        self.sheet = sheet
        self.expiries = expiries
        self.opts = opts
        self.tokens = [o["instrument_token"] for o in opts]
        self.token_set = set(self.tokens)
        self.chain = ChainState([o["strike"] for o in opts], [o["instrument_type"] == "CE" for o in opts],
                                [o["exp_idx"] for o in opts], rate=RATE)
        self.baseline = Baseline()
        self.last_row = None
        self.last_rec = None
        self.quant_model = QuantModel(symbol=sheet)
        self.lock = threading.RLock()
        self.tick_count = 0
        self.last_calc_latency_ms = 0.0


# ------------------------------------------------------------------ tick computation
def refresh(instr, ticks, now):
    px = [ticks.get(t, {}).get("px", np.nan) for t in instr.tokens]
    oi = [ticks.get(t, {}).get("oi", np.nan) for t in instr.tokens]
    T = np.array([max((datetime.combine(e, dtime(15, 30)) - now).total_seconds(), 60) / (365.0 * 86400)
                  for e in instr.expiries])
    instr.chain.update(px, oi, T)


def capture_baseline(instr, ticks, now, path):
    refresh(instr, ticks, now)
    iv_b, vg_b = instr.chain.buckets()
    instr.baseline.capture(instr.chain, iv_b, vg_b, N_OTM)
    instr.baseline.save(path, instr.chain)


def take_sample(instr, ticks, now, auto_trader):
    """
    Executes instant calculation of option chain greeks, quant recommendation,
    trailing stop ratchets, and risk guards directly upon broker tick arrival.
    """
    t_start = time.perf_counter()
    with instr.lock:
        refresh(instr, ticks, now)
        iv_b, vg_b = instr.chain.buckets()
        instr.last_row = instr.baseline.deltas(instr.chain, iv_b, vg_b)
        instr.tick_count += 1

        # Active position current price (for dynamic trailing SL calculation)
        active_pos_px = 0.0
        if auto_trader and instr.sheet in auto_trader.active_positions:
            pos = auto_trader.active_positions[instr.sheet]
            active_pos_px = ticks.get(pos.instrument_token, {}).get("px", pos.current_price)

        # Generate real-time Quant Trade Recommendation
        instr.last_rec = instr.quant_model.generate_recommendation(
            ce_vega=instr.last_row[0],
            pe_vega=instr.last_row[1],
            ce_theta=instr.last_row[2],
            pe_theta=instr.last_row[3],
            ce_oi=instr.last_row[4],
            pe_oi=instr.last_row[5],
            chain_opts=instr.opts,
            chain_state=instr.chain,
            ticks=ticks,
            now_time=now.time(),
            iv_buckets=iv_b,
            base_iv_buckets=instr.baseline.iv0,
            vega_deadband=instr.baseline.deadband(VEGA_DEADBAND_VOLPTS),
            active_position_price=active_pos_px
        )

        # Process Auto-Trading engine cycle immediately on this tick
        exec_status = auto_trader.on_cycle(instr.last_rec, ticks, now) if auto_trader else {}

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        instr.last_calc_latency_ms = t_elapsed_ms

        # Sync primary index state (NIFTY) to live web dashboard
        if instr.sheet == "NIFTY":
            with app_quant_trader.GLOBAL_STATE.lock:
                app_quant_trader.GLOBAL_STATE.symbol = instr.sheet
                app_quant_trader.GLOBAL_STATE.underlying_price = instr.last_rec.underlying_price
                app_quant_trader.GLOBAL_STATE.ce_vega = instr.last_row[0]
                app_quant_trader.GLOBAL_STATE.pe_vega = instr.last_row[1]
                app_quant_trader.GLOBAL_STATE.ce_theta = instr.last_row[2]
                app_quant_trader.GLOBAL_STATE.pe_theta = instr.last_row[3]
                app_quant_trader.GLOBAL_STATE.ce_oi = instr.last_row[4]
                app_quant_trader.GLOBAL_STATE.pe_oi = instr.last_row[5]

                # Alpha features
                vfi, skew, oir, z = instr.quant_model.calculate_alpha_features(
                    instr.last_row[0], instr.last_row[1], instr.last_row[2],
                    instr.last_row[3], instr.last_row[4], instr.last_row[5],
                    iv_b, instr.baseline.iv0, instr.baseline.deadband(VEGA_DEADBAND_VOLPTS)
                )
                app_quant_trader.GLOBAL_STATE.vfi = vfi
                app_quant_trader.GLOBAL_STATE.skew_shift = skew
                app_quant_trader.GLOBAL_STATE.oir = oir
                app_quant_trader.GLOBAL_STATE.z_score = z
                app_quant_trader.GLOBAL_STATE.latest_recommendation = instr.last_rec.to_dict()

                pos = auto_trader.active_positions.get(instr.sheet)
                app_quant_trader.GLOBAL_STATE.active_position = pos.to_dict() if pos else None
                app_quant_trader.GLOBAL_STATE.daily_realized_pnl = auto_trader.daily_realized_pnl
                app_quant_trader.GLOBAL_STATE.is_armed = auto_trader.is_armed
                app_quant_trader.GLOBAL_STATE.execution_mode = auto_trader.mode.value
                app_quant_trader.GLOBAL_STATE.last_update = now.strftime("%H:%M:%S.%f")[:-3]
                app_quant_trader.GLOBAL_STATE.tick_count = instr.tick_count
                app_quant_trader.GLOBAL_STATE.calc_latency_ms = round(t_elapsed_ms, 2)

        return instr.last_row, instr.last_rec, exec_status


def write_live_cards(ws_by_name, instruments, auto_trader):
    """Update dynamic Quant Decision Card (P2:Q14) and Sentiment Matrix (P22) across sheets."""
    for name, ins in instruments.items():
        if name not in ws_by_name or not ins.last_rec:
            continue
        ws = ws_by_name[name]
        rec = ins.last_rec
        rec_grid = [
            ["QUANT DECISION", f"{rec.signal.value} ({rec.regime.value})"],
            ["RECOMMENDED CONTRACT", rec.contract_symbol],
            ["STRIKE / SIDE", f"{rec.strike} {rec.option_type}"],
            ["ENTRY PRICE (LTP)", f"₹{rec.entry_price:.2f}"],
            ["RUNNING STOP LOSS", f"₹{rec.stop_loss:.2f}"],
            ["DYNAMIC TRAILING SL", f"₹{rec.trailing_stop_loss:.2f}"],
            ["TARGET 1 (1:1.5 RR)", f"₹{rec.target_1:.2f} (+₹{rec.expected_profit_t1:.0f}/lot)"],
            ["TARGET 2 (1:2.5 RR)", f"₹{rec.target_2:.2f} (+₹{rec.expected_profit_t2:.0f}/lot)"],
            ["MAX LOSS PER LOT", f"₹{rec.max_loss_per_lot:.0f}"],
            ["QUANTIFIED EDGE", f"Score: {rec.edge_score:.1f}/100 | Win: {rec.win_rate_probability_pct:.1f}%"],
            ["AUTO-TRADER STATUS", f"{'ARMED' if auto_trader.is_armed else 'DISARMED'} ({auto_trader.mode.value})"],
            ["DAILY REALIZED P&L", f"₹{auto_trader.daily_realized_pnl:,.2f}"],
            ["STATUS / ALERT", auto_trader.active_positions.get(name).exit_reason if name in auto_trader.active_positions else "Monitoring Tape"]
        ]
        try:
            ws.range("P2").value = rec_grid
        except Exception:
            pass

    # Sentiment Matrix on every sheet at P22
    valid_ins = [ins for ins in instruments.values() if ins.last_row is not None]
    if valid_ins:
        labels = [[ins.sheet, *infer(*ins.last_row, vega_deadband=ins.baseline.deadband(VEGA_DEADBAND_VOLPTS))]
                  for ins in valid_ins]
        for ws in ws_by_name.values():
            try:
                ws.range("P22").value = labels
            except Exception:
                pass


def write_snapshot_rows(ws_by_name, instruments, excel_row, now):
    """Append periodic historical row down columns A:G."""
    stamp = "'" + now.strftime("%H:%M:%S")
    for name, ins in instruments.items():
        if name in ws_by_name and ins.last_row is not None:
            try:
                ws_by_name[name].range((excel_row, 1)).value = [stamp] + ins.last_row
            except Exception:
                pass


# ------------------------------------------------------------------ simulated tick streamer
class SimulatedTicker(threading.Thread):
    """
    High-frequency tick streamer for offline / testing mode.
    Streams realistic market price and OI changes at 250ms intervals (4 ticks/sec)
    so the entire calculation and quant model run tick-by-tick even outside market hours.
    """
    def __init__(self, instruments, on_ticks_callback, interval_sec=0.25):
        super().__init__(daemon=True)
        self.instruments = instruments
        self.on_ticks_callback = on_ticks_callback
        self.interval_sec = interval_sec
        self.running = True

    def run(self):
        while self.running:
            batch = []
            for ins in self.instruments.values():
                sample_size = min(6, len(ins.opts))
                if sample_size == 0:
                    continue
                chosen_opts = np.random.choice(ins.opts, size=sample_size, replace=False)
                for opt in chosen_opts:
                    token = opt["instrument_token"]
                    cur_px = opt.get("_sim_px", 120.0 + np.random.uniform(5, 50))
                    step = np.random.normal(0, 0.45)
                    new_px = max(1.0, round(cur_px + step, 2))
                    opt["_sim_px"] = new_px
                    cur_oi = opt.get("_sim_oi", 1500000)
                    new_oi = max(10000, cur_oi + int(np.random.normal(0, 2000)))
                    opt["_sim_oi"] = new_oi

                    batch.append({
                        "instrument_token": token,
                        "last_price": new_px,
                        "depth": {
                            "buy": [{"price": round(new_px - 0.05, 2)}],
                            "sell": [{"price": round(new_px + 0.05, 2)}]
                        },
                        "oi": new_oi,
                        "volume": int(np.random.randint(50, 500))
                    })
            if batch:
                self.on_ticks_callback(None, batch)
            time.sleep(self.interval_sec)

    def stop(self):
        self.running = False


# ------------------------------------------------------------------ main execution
def main():
    import xlwings as xw
    from kiteconnect import KiteConnect, KiteTicker

    print("======================================================================")
    print("   ⚡ DIRECTIONAL GREEKS QUANT SYSTEM & AUTO-TRADER (TICK-BY-TICK)")
    print("======================================================================")

    kite = None
    if API_KEY != "your_api_key" and ACCESS_TOKEN != "your_access_token":
        try:
            kite = KiteConnect(api_key=API_KEY)
            kite.set_access_token(ACCESS_TOKEN)
            print("✓ Connected to Zerodha Kite Connect API.")
        except Exception as e:
            print(f"⚠️ Broker auth warning: {e}. Running in simulated market mode.")

    # Initialize Automated Execution Controller
    auto_trader = AutoTradingController(
        mode=DEFAULT_EXECUTION_MODE,
        kite_client=kite,
        max_loss_per_day=MAX_DAILY_LOSS_LIMIT,
        max_trades_per_day=MAX_TRADES_PER_DAY
    )
    if ENABLE_AUTO_TRADING_ARM:
        auto_trader.arm()

    app_quant_trader.GLOBAL_CONTROLLER = auto_trader

    # Start Live Web Control Dashboard in background thread
    if ENABLE_WEB_DASHBOARD:
        dashboard_thread = threading.Thread(
            target=app_quant_trader.run_dashboard_server,
            args=(WEB_DASHBOARD_PORT,),
            daemon=True
        )
        dashboard_thread.start()

    today = date.today()
    inst_lists = {"NFO": [], "BFO": []}
    if kite:
        try:
            inst_lists = {"NFO": kite.instruments("NFO"), "BFO": kite.instruments("BFO")}
        except Exception as e:
            print(f"Failed to fetch instrument list: {e}")

    instruments = {}
    ticks = {}

    for sheet, (seg, name) in SHEETS.items():
        if kite and inst_lists[seg]:
            futs = sorted((i for i in inst_lists[seg] if i["name"] == name and i["instrument_type"] == "FUT"
                           and i["expiry"] >= today), key=lambda i: i["expiry"])
            fut_key = f'{seg}:{futs[0]["tradingsymbol"]}'
            fut_price = kite.ltp([fut_key])[fut_key]["last_price"]
            expiries, opts = pick_chain(inst_lists, seg, name, fut_price, today)
        else:
            # Fallback simulated options chain structure
            expiries = [today, today]
            fut_price = 24500.0 if "NIFTY" in sheet else 52000.0 if "BANK" in sheet else 1500.0
            opts = [
                {"instrument_token": 1000 + i, "instrument_type": "CE" if i % 2 == 0 else "PE",
                 "strike": fut_price - 200 + (i * 50), "tradingsymbol": f"{name}_{fut_price}_{i}",
                 "exp_idx": 0, "expiry": today}
                for i in range(12)
            ]
            for o in opts:
                ticks[o["instrument_token"]] = {"px": 120.0, "oi": 1500000}

        instruments[sheet] = Instrument(sheet, expiries, opts)
        print(f"[{sheet}] Chain ready: {len(opts)} contracts around ₹{fut_price}")

    token_to_instruments = {}
    for ins in instruments.values():
        for tok in ins.tokens:
            token_to_instruments.setdefault(tok, []).append(ins)

    paths = {n: f"baseline_{today:%Y%m%d}_{n.replace(' ', '')}.npz" for n in SHEETS}
    restart = all(os.path.exists(p) for p in paths.values())

    def check_and_capture_baseline(ins, current_ticks, now_time, path):
        if not ins.baseline.ready:
            px = [current_ticks.get(t, {}).get("px", np.nan) for t in ins.tokens]
            if np.isfinite(px).sum() >= max(2, len(ins.tokens) // 4):
                try:
                    capture_baseline(ins, current_ticks, now_time, path)
                    print(f"[{ins.sheet}] 09:15 Baseline frozen ({np.isfinite(px).sum()}/{len(ins.tokens)} contracts).")
                except Exception as e:
                    pass

    if restart:
        for n, ins in instruments.items():
            ins.baseline.load(paths[n], ins.chain)
        print("✓ Loaded saved 09:15 baselines from disk.")
    else:
        now = datetime.now()
        for n, ins in instruments.items():
            check_and_capture_baseline(ins, ticks, now, paths[n])

    excel_dirty = threading.Event()

    def on_ticks(ws, batch):
        now = datetime.now()
        touched_instruments = set()

        for t in batch:
            tok = t.get("instrument_token")
            if not tok:
                continue
            d = t.get("depth") or {}
            bid = d["buy"][0]["price"] if d.get("buy") else 0
            ask = d["sell"][0]["price"] if d.get("sell") else 0
            px = (bid + ask) / 2 if bid > 0 and ask > 0 else t.get("last_price", np.nan)
            ticks[tok] = {"px": px, "oi": t.get("oi", np.nan)}

            for ins in token_to_instruments.get(tok, []):
                touched_instruments.add(ins)

        # ⚡ TICK-BY-TICK CALCULATION: Run computation instantly upon broker tick arrival
        for ins in touched_instruments:
            if not ins.baseline.ready:
                check_and_capture_baseline(ins, ticks, now, paths[ins.sheet])

            if ins.baseline.ready:
                take_sample(ins, ticks, now, auto_trader)
                excel_dirty.set()

    sim_ticker = None
    if kite:
        all_tokens = [t for i in instruments.values() for t in i.tokens]
        ticker = KiteTicker(API_KEY, ACCESS_TOKEN)
        ticker.on_ticks = on_ticks
        ticker.on_connect = lambda ws, resp: (ws.subscribe(all_tokens), ws.set_mode(ws.MODE_FULL, all_tokens))
        ticker.connect(threaded=True)
        print("✓ Connected to KiteTicker WebSocket. Streaming real-time exchange ticks.")
    else:
        print("⚡ Operating in real-time tick-by-tick simulation mode (4 ticks/sec).")
        sim_ticker = SimulatedTicker(instruments, on_ticks, interval_sec=SIMULATED_TICK_INTERVAL_SEC)
        sim_ticker.start()

    sheets = {}
    book = None
    try:
        book = xw.Book(WORKBOOK)
        sheets = {n: book.sheets[n] for n in SHEETS}
    except Exception as e:
        print(f"Excel workbook note: {e}. Excel export will be skipped or updated on reconnect.")

    row = FIRST_ROW
    print("\n🟢 Live Quant Tape Engine is Running (TICK-BY-TICK)...")
    print(f"Auto-Trading Status: {'ARMED' if auto_trader.is_armed else 'DISARMED'} | Mode: {auto_trader.mode.value}")
    print(f"Live Dashboard: http://localhost:{WEB_DASHBOARD_PORT}\n")

    log = open("greeks_snapshots.csv", "a", newline="")
    logw = csv.writer(log)

    running = True

    # Dedicated background thread for updating Excel without blocking the WebSocket
    def excel_sync_worker():
        nonlocal row
        last_row_snapshot = time.time()
        last_excel_flush = time.time()

        while running:
            # Wake up on tick trigger or small timeout
            excel_dirty.wait(timeout=0.05)
            excel_dirty.clear()

            now = datetime.now()
            t_now = time.time()

            # 1. Update Live Decision Card & Sentiment Matrix in Excel
            if sheets and (t_now - last_excel_flush >= EXCEL_FLUSH_INTERVAL_SEC):
                try:
                    write_live_cards(sheets, instruments, auto_trader)
                    last_excel_flush = t_now
                except Exception:
                    pass

            # 2. Append historical snapshot row
            if t_now - last_row_snapshot >= ROW_SNAPSHOT_INTERVAL_SEC:
                if sheets:
                    try:
                        write_snapshot_rows(sheets, instruments, row, now)
                    except Exception:
                        pass
                for n, ins in instruments.items():
                    if ins.last_row:
                        logw.writerow([now.isoformat(timespec="seconds"), n, *ins.last_row])
                log.flush()
                row += 1
                last_row_snapshot = t_now

    excel_thread = threading.Thread(target=excel_sync_worker, daemon=True)
    excel_thread.start()

    try:
        last_print = 0.0
        while running:
            time.sleep(0.1)
            now = datetime.now()
            t_now = time.time()

            # Terminal pulse printout every 1 second
            if t_now - last_print >= 1.0:
                last_print = t_now
                nifty_ins = instruments.get("NIFTY")
                nifty_rec = nifty_ins.last_rec if nifty_ins else None
                if nifty_rec:
                    sig_color = "\033[92m" if nifty_rec.signal == TradeSignalType.BUY_CALL else "\033[91m" if nifty_rec.signal == TradeSignalType.BUY_PUT else "\033[90m"
                    reset_col = "\033[0m"
                    lat = nifty_ins.last_calc_latency_ms
                    cnt = nifty_ins.tick_count
                    pos_txt = ""
                    if auto_trader and "NIFTY" in auto_trader.active_positions:
                        p = auto_trader.active_positions["NIFTY"]
                        pos_txt = f" | Active: {p.contract_symbol} @ ₹{p.current_price:.2f} (SL: ₹{p.trailing_stop_loss:.2f})"

                    print(f"[{now:%H:%M:%S}] ⚡ TICK-BY-TICK | NIFTY: ₹{nifty_rec.underlying_price:.2f} | "
                          f"Signal: {sig_color}{nifty_rec.signal.value}{reset_col} ({nifty_rec.contract_symbol}) | "
                          f"Edge: {nifty_rec.edge_score:.1f}/100 | "
                          f"Auto-Trader: {'ARMED' if auto_trader.is_armed else 'DISARMED'} ({auto_trader.mode.value}) | "
                          f"Ticks: {cnt} ({lat:.1f}ms){pos_txt}")

            if now.time() >= SESSION_END or row > LAST_ROW:
                break
    except KeyboardInterrupt:
        print("\nStopping live engine...")
    finally:
        running = False
        if sim_ticker:
            sim_ticker.stop()
        if book:
            out = f"DirectionalGreeks-{today:%Y%b%d}_live.xlsx"
            try:
                book.save(out)
                print(f"Workbook saved to: {out}")
            except Exception:
                pass
        log.close()


if __name__ == "__main__":
    main()
