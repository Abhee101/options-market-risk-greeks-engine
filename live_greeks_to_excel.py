"""
Live option-chain greeks -> your DirectionalGreeks workbook (same sheets, columns, charts, inference table).

Method (constant-delta version you sent):
    Flow = sum over expiries and delta buckets {50, 25, 10 delta} of Vega_live(m) * (IV_live(m) - IV_base(m))
  * Each bucket follows whichever strike currently has that delta; IV_base is the 09:15 IV of that delta bucket.
  * CE_Vega = the three call buckets, PE_Vega = the three put buckets, summed over the tracked expiries.
  * The 09:15:00 baseline is captured ONCE, frozen, and saved to disk (baseline_<date>_<sheet>.npz).
    A restart reloads that file; it never recaptures. If you start after 09:15 with no saved file, it stops.
  * Theta and OI columns are my extension: fixed ATM->OTM strikes, live minus 09:15 baseline (OI in millions).

The workbook is NOT rebuilt: the script opens YOUR file, clears A2:G750 on each sheet and fills it, so the
layout, the three charts per sheet and the inference box stay exactly as in DirectionalGreeks-2026Oct07.xlsx.

The calculation part is tested. The broker and Excel parts are NOT (I had neither here).
    pip install kiteconnect xlwings numpy      (keep directional_greeks_core.py in the same folder)
"""
import csv
import os
import time
from datetime import date, datetime, time as dtime

import numpy as np

from directional_greeks_core import COLS, Baseline, ChainState, infer

# ------------------------------------------------------------------ settings
API_KEY = "your_api_key"
ACCESS_TOKEN = "your_access_token"              # regenerate every trading day
WORKBOOK = "DirectionalGreeks-2026Oct07.xlsx"   # your template, opened in Excel
BASELINE_TIME = dtime(9, 15, 0)
SESSION_START = dtime(9, 16, 15)                # first row written, as in your file
SESSION_END = dtime(15, 30, 0)
SAMPLE_SECONDS = 35                             # your file averages ~37 s between rows
N_EXPIRIES = 2                                  # "all specified expiries": nearest N
VEGA_DEADBAND_VOLPTS = 0.05                     # a Vega score smaller than a 0.05-vol-point average IV move is treated as zero
WINDOW_SIGMA = 0.20                             # rough IV used only to size the subscribed strike window
N_OTM = 10                                      # OTM strikes counted beyond ATM on each side (my choice; the method does not say)
RATE = 0.065                                    # risk-free rate, set to a current short-term yield
FIRST_ROW, LAST_ROW = 2, 750                    # chart ranges run to row 750

# sheet name -> (exchange segment, instrument name in the broker's list). Order = order of the inference box.
SHEETS = {
    "NIFTY": ("NFO", "NIFTY"),
    "BANKNIFTY": ("NFO", "BANKNIFTY"),
    "SENSEX": ("BFO", "SENSEX"),
    "RELIANCE": ("NFO", "RELIANCE"),
    "TCS": ("NFO", "TCS"),
    "HDFC Bank": ("NFO", "HDFCBANK"),
    "ICICI Bank": ("NFO", "ICICIBANK"),
}


# ------------------------------------------------------------------ chain setup (needs the broker's instrument list)
def pick_chain(instruments, seg, name, fut_price, today, n_exp=N_EXPIRIES, n_otm=N_OTM):
    rows = [i for i in instruments[seg] if i["name"] == name and i["instrument_type"] in ("CE", "PE") and i["expiry"] >= today]
    expiries = sorted({i["expiry"] for i in rows})[:n_exp]
    strikes = sorted({i["strike"] for i in rows if i["expiry"] == expiries[0]})
    atm = min(strikes, key=lambda s: abs(s - fut_price))
    k = strikes.index(atm)
    wide = n_otm + 5                                    # extra strikes so a gap-open still leaves the 09:15 ATM inside the window
    keep = set(strikes[max(0, k - wide): k + wide + 1])
    t_far = max((expiries[-1] - today).days, 1) / 365.0  # the 10-delta strike of the furthest expiry must also be inside
    reach = 2.0 * WINDOW_SIGMA * np.sqrt(t_far) * fut_price
    keep |= {s for s in strikes if abs(s - fut_price) <= reach}
    opts = [dict(i, exp_idx=expiries.index(i["expiry"])) for i in rows if i["expiry"] in expiries and i["strike"] in keep]
    opts.sort(key=lambda i: (i["exp_idx"], i["strike"], i["instrument_type"]))
    return expiries, opts


class Instrument:
    def __init__(self, sheet, expiries, opts):
        self.sheet, self.expiries, self.opts = sheet, expiries, opts
        self.tokens = [o["instrument_token"] for o in opts]
        self.chain = ChainState([o["strike"] for o in opts], [o["instrument_type"] == "CE" for o in opts],
                                [o["exp_idx"] for o in opts], rate=RATE)
        self.baseline = Baseline()
        self.last_row = None


# ------------------------------------------------------------------ one cycle (testable with fake inputs)
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


def take_sample(instr, ticks, now):
    refresh(instr, ticks, now)
    iv_b, vg_b = instr.chain.buckets()
    instr.last_row = instr.baseline.deltas(instr.chain, iv_b, vg_b)
    return instr.last_row


def write_sample(ws_by_name, instruments, excel_row, now):
    stamp = "'" + now.strftime("%H:%M:%S")                       # apostrophe -> stored as text, like your file
    for name, ins in instruments.items():
        ws_by_name[name].range((excel_row, 1)).value = [stamp] + ins.last_row
    labels = [[name, *infer(*ins.last_row, vega_deadband=ins.baseline.deadband(VEGA_DEADBAND_VOLPTS))]
              for name, ins in instruments.items()]
    for ws in ws_by_name.values():                               # identical inference box on every sheet
        ws.range("P22").value = labels


# ------------------------------------------------------------------ main
def main():
    import xlwings as xw
    from kiteconnect import KiteConnect, KiteTicker

    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)
    today = date.today()
    inst_lists = {"NFO": kite.instruments("NFO"), "BFO": kite.instruments("BFO")}

    instruments = {}
    for sheet, (seg, name) in SHEETS.items():
        futs = sorted((i for i in inst_lists[seg] if i["name"] == name and i["instrument_type"] == "FUT"
                       and i["expiry"] >= today), key=lambda i: i["expiry"])
        fut_key = f'{seg}:{futs[0]["tradingsymbol"]}'
        fut_price = kite.ltp([fut_key])[fut_key]["last_price"]
        expiries, opts = pick_chain(inst_lists, seg, name, fut_price, today)
        instruments[sheet] = Instrument(sheet, expiries, opts)
        print(f"{sheet}: expiries {expiries}, {len(opts)} contracts around {fut_price}")

    ticks = {}

    def on_ticks(ws, batch):                                     # only ever touches `ticks`, never the baseline
        for t in batch:
            d = t.get("depth") or {}
            bid = d["buy"][0]["price"] if d.get("buy") else 0
            ask = d["sell"][0]["price"] if d.get("sell") else 0
            px = (bid + ask) / 2 if bid > 0 and ask > 0 else t.get("last_price", np.nan)
            ticks[t["instrument_token"]] = {"px": px, "oi": t.get("oi", np.nan)}

    all_tokens = [t for i in instruments.values() for t in i.tokens]
    ticker = KiteTicker(API_KEY, ACCESS_TOKEN)
    ticker.on_ticks = on_ticks
    ticker.on_connect = lambda ws, resp: (ws.subscribe(all_tokens), ws.set_mode(ws.MODE_FULL, all_tokens))
    ticker.connect(threaded=True)

    paths = {n: f"baseline_{today:%Y%m%d}_{n.replace(' ', '')}.npz" for n in SHEETS}
    restart = all(os.path.exists(p) for p in paths.values())
    book = xw.Book(WORKBOOK)
    sheets = {n: book.sheets[n] for n in SHEETS}

    if restart:                                                  # saved 09:15 baseline found: reuse, do not recapture
        for n, ins in instruments.items():
            ins.baseline.load(paths[n], ins.chain)
        filled = sheets["NIFTY"].range(f"A{FIRST_ROW}:A{LAST_ROW}").options(ndim=1).value
        row = FIRST_ROW + sum(v is not None for v in filled)
        print("Restart: loaded saved 09:15 baselines, continuing at row", row)
    else:
        if datetime.now().time() > dtime(9, 15, 30):
            raise SystemExit("It is past 09:15 and no saved baseline exists. The 09:15 reference cannot be rebuilt, so stopping.")
        for ws in sheets.values():
            ws.range(f"A{FIRST_ROW}:G{LAST_ROW}").clear_contents()   # formatting untouched
        while datetime.now().time() < BASELINE_TIME:
            time.sleep(0.2)
        now = datetime.now()
        for n, ins in instruments.items():
            capture_baseline(ins, ticks, now, paths[n])
            print(f"{n}: baseline frozen at {now:%H:%M:%S}, {int(ins.baseline.active.sum())} theta/OI strikes in, "
                  f"{ins.baseline.excluded} strikes and {ins.baseline.buckets_missing} delta buckets left out (no 09:15 quote)")
        row = FIRST_ROW
        while datetime.now().time() < SESSION_START:
            time.sleep(0.5)

    log = open("greeks_snapshots.csv", "a", newline="")
    logw = csv.writer(log)
    while True:
        started = time.time()
        now = datetime.now()
        for ins in instruments.values():
            take_sample(ins, ticks, now)
        write_sample(sheets, instruments, row, now)
        for n, ins in instruments.items():
            logw.writerow([now.isoformat(timespec="seconds"), n, *ins.last_row])
        log.flush()
        row += 1
        if now.time() >= SESSION_END or row > LAST_ROW:
            break
        time.sleep(max(0.0, SAMPLE_SECONDS - (time.time() - started)))

    out = f"DirectionalGreeks-{today:%Y%b%d}.xlsx"                # e.g. DirectionalGreeks-2026Oct07.xlsx
    if os.path.exists(out):
        out = out.replace(".xlsx", "_live.xlsx")                  # never overwrite your template
    book.save(out)
    ticker.close()
    print("saved", out)


if __name__ == "__main__":
    main()
