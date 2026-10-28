"""
Calculation layer for the DirectionalGreeks sheets. No broker or Excel code in here, so it can be tested alone.

VEGA COLUMNS (constant-delta method, as you described it):
    Flow = sum over expiries, over delta buckets m in {50, 25, 10 delta}, of
           Vega_live(m) * ( IV_live(m) - IV_base(m) )            IV in vol points, vega per 1 vol point
    CE_Vega = the call buckets, PE_Vega = the put buckets. Each bucket is re-mapped to whichever strike
    currently has that delta (IV and vega are interpolated between strikes in delta space).
    IV_base(m) is captured once at 09:15, frozen (read-only), saved to disk and never overwritten.

THETA AND OI COLUMNS (my extension - the method only covers vega): same as the previous version,
    sum over the fixed ATM->OTM strikes chosen at 09:15 of (live - 09:15 baseline); OI shown in millions.
    The fixed-strike theta column still moves with spot; it is NOT constant-delta.
"""
import math

import numpy as np

_erf = np.vectorize(math.erf)
TARGET_DELTAS = (0.50, 0.25, 0.10)          # absolute delta of each bucket, used for both calls and puts


def _cdf(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, float) / math.sqrt(2.0)))


def _pdf(x):
    x = np.asarray(x, float)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ------------------------------------------------------------------ Black-76 on a forward
def _d1d2(F, K, T, sig):
    sd = sig * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd * sd) / sd
    return d1, d1 - sd


def black76_price(F, K, T, sig, r, is_call):
    F, K, T, sig = (np.asarray(a, float) for a in (F, K, T, sig))
    d1, d2 = _d1d2(F, K, T, sig)
    df = np.exp(-r * T)
    call = df * (F * _cdf(d1) - K * _cdf(d2))
    put = df * (K * _cdf(-d2) - F * _cdf(-d1))
    return np.where(is_call, call, put)


def black76_greeks(F, K, T, sig, r, is_call):
    """vega per 1 vol point, theta per calendar day (365-day year), forward delta."""
    F, K, T, sig = (np.asarray(a, float) for a in (F, K, T, sig))
    d1, _ = _d1d2(F, K, T, sig)
    df = np.exp(-r * T)
    price = black76_price(F, K, T, sig, r, is_call)
    vega = df * F * _pdf(d1) * np.sqrt(T) / 100.0
    theta = (r * price - df * F * _pdf(d1) * sig / (2.0 * np.sqrt(T))) / 365.0
    delta = np.where(is_call, df * _cdf(d1), -df * _cdf(-d1))
    return vega, theta, delta


def implied_vol(price, F, K, T, r, is_call, lo=1e-3, hi=4.0, iters=60):
    price = np.asarray(price, float)
    K = np.asarray(K, float)
    a = np.full(price.shape, lo)
    b = np.full(price.shape, hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        below = black76_price(F, K, T, m, r, is_call) < price
        a = np.where(below, m, a)
        b = np.where(below, b, m)
    iv = 0.5 * (a + b)
    p_lo = black76_price(F, K, T, np.full(price.shape, lo), r, is_call)
    p_hi = black76_price(F, K, T, np.full(price.shape, hi), r, is_call)
    bad = ~np.isfinite(price) | (price <= 0) | (price <= p_lo + 1e-9) | (price >= p_hi - 1e-9)
    return np.where(bad, np.nan, iv)


def parity_forward(K, call_px, put_px, T, r):
    """Forward implied by the strike where call and put prices are closest (same expiry)."""
    K, c, p = (np.asarray(a, float) for a in (K, call_px, put_px))
    ok = np.isfinite(c) & np.isfinite(p) & (c > 0) & (p > 0)
    if not ok.any():
        return float("nan")
    i = np.argmin(np.where(ok, np.abs(c - p), np.inf))
    return float(K[i] + (c[i] - p[i]) * math.exp(r * T))


# ------------------------------------------------------------------ one instrument's option chain (several expiries)
class ChainState:
    def __init__(self, strikes, is_call, expiry_idx, rate=0.065):
        self.K = np.asarray(strikes, float)
        self.call = np.asarray(is_call, bool)
        self.e = np.asarray(expiry_idx, int)
        self.r = rate
        n = len(self.K)
        self.n_exp = int(self.e.max()) + 1
        self.theta = np.full(n, np.nan)       # carried forward (used by the fixed-strike theta column)
        self.oi = np.full(n, np.nan)
        self.iv_f = np.full(n, np.nan)        # fresh this update only (used for the delta buckets)
        self.delta_f = np.full(n, np.nan)
        self.vega_f = np.full(n, np.nan)
        self.F = np.full(self.n_exp, np.nan)
        self.pairs = []
        for x in range(self.n_exp):
            d = {}
            for i in np.where(self.e == x)[0]:
                d.setdefault(self.K[i], {})["c" if self.call[i] else "p"] = i
            both = [v for v in d.values() if len(v) == 2]
            self.pairs.append((np.array([v["c"] for v in both], int), np.array([v["p"] for v in both], int)))

    def update(self, px, oi, T):
        px = np.asarray(px, float)
        T = np.asarray(T, float)
        self.oi = np.where(np.isfinite(np.asarray(oi, float)), np.asarray(oi, float), self.oi)
        for x, (ci, pi) in enumerate(self.pairs):
            if len(ci):
                f = parity_forward(self.K[ci], px[ci], px[pi], T[x], self.r)
                if np.isfinite(f):
                    self.F[x] = f
        Fc, Tc = self.F[self.e], T[self.e]
        ok = np.isfinite(Fc) & (Tc > 0)
        self.iv_f[:] = self.delta_f[:] = self.vega_f[:] = np.nan
        if ok.any():
            iv = np.full(len(px), np.nan)
            iv[ok] = implied_vol(px[ok], Fc[ok], self.K[ok], Tc[ok], self.r, self.call[ok])
            idx = np.where(np.isfinite(iv))[0]
            if len(idx):
                v, th, de = black76_greeks(Fc[idx], self.K[idx], Tc[idx], iv[idx], self.r, self.call[idx])
                self.iv_f[idx], self.vega_f[idx], self.delta_f[idx] = iv[idx], v, de
                self.theta[idx] = th

    def buckets(self, targets=TARGET_DELTAS):
        """IV and vega at each target |delta|, per expiry and side. Shape (n_exp, 2 [call, put], n_targets). NaN if not bracketed."""
        IV = np.full((self.n_exp, 2, len(targets)), np.nan)
        VG = IV.copy()
        for x in range(self.n_exp):
            for s, is_call in ((0, True), (1, False)):
                m = (self.e == x) & (self.call == is_call) & np.isfinite(self.iv_f)
                if m.sum() < 2:
                    continue
                order = np.argsort(np.abs(self.delta_f[m]))
                d, iv, vg = np.abs(self.delta_f[m])[order], self.iv_f[m][order], self.vega_f[m][order]
                for b, t in enumerate(targets):
                    if d[0] <= t <= d[-1]:
                        IV[x, s, b] = np.interp(t, d, iv)
                        VG[x, s, b] = np.interp(t, d, vg)
        return IV, VG


COLS = ["CE_Vega", "PE_Vega", "CE_Theta", "PE_Theta", "CE_OI", "PE_OI"]
DECIMALS = [2, 2, 2, 2, 1, 1]          # matches the precision in your file


class Baseline:
    """The 09:15 reference: IV of each delta bucket plus the fixed ATM->OTM strike set. Frozen once captured."""

    def __init__(self):
        self.ready = False

    def capture(self, chain, iv_buckets, vg_buckets, n_otm):
        if self.ready:
            raise RuntimeError("Baseline already captured - refusing to overwrite it.")
        active = np.zeros(len(chain.K), bool)
        for x in range(chain.n_exp):
            if not np.isfinite(chain.F[x]):
                continue
            ks = np.unique(chain.K[chain.e == x])
            atm = ks[np.argmin(np.abs(ks - chain.F[x]))]
            ce_ks, pe_ks = ks[ks >= atm][: n_otm + 1], ks[ks <= atm][::-1][: n_otm + 1]
            active |= (chain.e == x) & chain.call & np.isin(chain.K, ce_ks)
            active |= (chain.e == x) & ~chain.call & np.isin(chain.K, pe_ks)
        valid = active & np.isfinite(chain.theta) & np.isfinite(chain.oi)
        self.excluded = int((active & ~valid).sum())               # no quote at 09:15: left out all day, never back-filled
        self.buckets_missing = int(np.isnan(iv_buckets).sum())     # delta buckets with no 09:15 IV: left out all day
        self._set(valid, chain.theta, chain.oi, iv_buckets, vg_buckets)

    def _set(self, active, t0, o0, iv0, vg0):
        self.active, self.t0, self.o0, self.iv0, self.vg0 = (np.array(a).copy() for a in (active, t0, o0, iv0, vg0))
        for a in (self.active, self.t0, self.o0, self.iv0, self.vg0):
            a.setflags(write=False)
        self.ready = True

    def deltas(self, chain, iv_b, vg_b):
        """[CE_Vega, PE_Vega, CE_Theta, PE_Theta, CE_OI, PE_OI]."""
        out = {}
        ok = np.isfinite(self.iv0) & np.isfinite(iv_b) & np.isfinite(vg_b)
        flow = np.where(ok, vg_b * (iv_b - self.iv0) * 100.0, 0.0)          # vega * change in IV (vol points)
        out["CE_Vega"], out["PE_Vega"] = float(flow[:, 0, :].sum()), float(flow[:, 1, :].sum())
        m = self.active & np.isfinite(chain.theta) & np.isfinite(chain.oi)
        for side, sel in (("CE", chain.call), ("PE", ~chain.call)):
            s = m & sel
            out[side + "_Theta"] = float(np.sum(chain.theta[s] - self.t0[s]))
            out[side + "_OI"] = float(np.sum(chain.oi[s] - self.o0[s])) / 1e6
        return [round(out[c] + 0.0, d) + 0.0 for c, d in zip(COLS, DECIMALS)]

    def deadband(self, vol_points):
        """Size of a Vega score that equals a vega-weighted average IV move of `vol_points` (scale-free noise floor)."""
        ok = np.isfinite(self.iv0) & np.isfinite(self.vg0)
        per_side = [float(np.where(ok[:, s, :], self.vg0[:, s, :], 0.0).sum()) for s in (0, 1)]
        return vol_points * float(np.mean(per_side))

    def save(self, path, chain):
        np.savez(path, K=chain.K, call=chain.call, e=chain.e, active=self.active, t0=self.t0, o0=self.o0,
                 iv0=self.iv0, vg0=self.vg0)

    def load(self, path, chain):
        z = np.load(path)
        if not (np.array_equal(z["K"], chain.K) and np.array_equal(z["call"], chain.call) and np.array_equal(z["e"], chain.e)):
            raise RuntimeError("Saved baseline was made for a different set of contracts - not using it.")
        self._set(z["active"], z["t0"], z["o0"], z["iv0"], z["vg0"])


# ------------------------------------------------------------------ inference labels
def infer(ce_vega, pe_vega, ce_theta, pe_theta, ce_oi, pe_oi, vega_deadband=0.0):
    """Vega: your 'distribution' reading (put side bid up while call side is hit = Bearish); a side whose score is
    inside +/- vega_deadband counts as zero. Theta and OI: rules fitted to your original file (all 21 labels matched)."""
    if abs(ce_vega) <= vega_deadband:
        ce_vega = 0.0
    if abs(pe_vega) <= vega_deadband:
        pe_vega = 0.0
    if pe_vega > 0 and ce_vega < 0:
        vega = "Bearish"
    elif ce_vega > 0 and pe_vega < 0:
        vega = "Bullish"
    else:
        vega = "Sideways"

    if ce_theta > 0 and pe_theta < 0:
        theta = "Bearish"
    elif ce_theta < 0 and pe_theta > 0:
        theta = "Bullish"
    elif ce_theta >= 0 and pe_theta >= 0:
        theta = "Sideways"
    else:
        theta = "No View"

    oi = "Bearish" if ce_oi < pe_oi else "Bullish" if pe_oi < ce_oi else "No View"
    return vega, theta, oi
