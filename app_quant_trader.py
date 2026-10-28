"""
Interactive Live Dashboard and Auto-Trading Control Center for Directional Greeks.

Features:
- Live streaming of Greek flows, Skew velocity, and OI asymmetry
- Quant Model Decision Panel: exact side to buy (Call/Put), exact strike, entry, dynamic SL, trailing SL, targets
- Quantified Edge Metrics (Edge Score, Skew Conviction %, Statistical Win Rate %, Risk/Reward)
- Interactive Auto-Trading Arm/Disarm Button
- Paper vs Live Broker Execution Toggle
- Emergency Instant Kill Switch
- Active Position Monitoring with Mark-to-Market P&L
"""

import json
import os
import socket
import threading
import time
from datetime import datetime, date, time as dtime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
from typing import Dict, Any, Optional, List

from quant_trade_engine import QuantModel, TradeRecommendation, TradeSignalType, MarketRegime
from order_execution_engine import AutoTradingController, ExecutionMode, Position


# Global state shared with web dashboard
class LiveTraderState:
    def __init__(self):
        self.lock = threading.Lock()
        self.symbol = "NIFTY"
        self.underlying_price = 24500.0
        self.ce_vega = 0.0
        self.pe_vega = 0.0
        self.ce_theta = 0.0
        self.pe_theta = 0.0
        self.ce_oi = 0.0
        self.pe_oi = 0.0
        self.vfi = 0.0
        self.skew_shift = 0.0
        self.oir = 0.0
        self.z_score = 0.0
        self.latest_recommendation: Optional[Dict[str, Any]] = None
        self.active_position: Optional[Dict[str, Any]] = None
        self.closed_trades: list = []
        self.daily_realized_pnl = 0.0
        self.is_armed = False
        self.execution_mode = "PAPER"
        self.last_update = datetime.now().strftime("%H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "symbol": self.symbol,
                "underlying_price": round(self.underlying_price, 2),
                "ce_vega": round(self.ce_vega, 2),
                "pe_vega": round(self.pe_vega, 2),
                "ce_theta": round(self.ce_theta, 2),
                "pe_theta": round(self.pe_theta, 2),
                "ce_oi": round(self.ce_oi, 2),
                "pe_oi": round(self.pe_oi, 2),
                "vfi": round(self.vfi, 2),
                "skew_shift": round(self.skew_shift, 2),
                "oir": round(self.oir, 2),
                "z_score": round(self.z_score, 2),
                "recommendation": self.latest_recommendation,
                "active_position": self.active_position,
                "closed_trades": self.closed_trades[-10:],
                "daily_realized_pnl": round(self.daily_realized_pnl, 2),
                "is_armed": self.is_armed,
                "execution_mode": self.execution_mode,
                "last_update": self.last_update
            }


GLOBAL_STATE = LiveTraderState()
GLOBAL_CONTROLLER: Optional[AutoTradingController] = None


class QuantTraderHTTPHandler(SimpleHTTPRequestHandler):
    """HTTP handler serving the live trading dashboard and REST control API."""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(self.render_dashboard_html().encode("utf-8"))
            return

        elif path == "/api/state":
            data = GLOBAL_STATE.to_dict()
            self._send_json(200, data)
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/toggle_arm":
            with GLOBAL_STATE.lock:
                if GLOBAL_CONTROLLER:
                    if GLOBAL_CONTROLLER.is_armed:
                        msg = GLOBAL_CONTROLLER.disarm()
                        GLOBAL_STATE.is_armed = False
                    else:
                        msg = GLOBAL_CONTROLLER.arm()
                        GLOBAL_STATE.is_armed = GLOBAL_CONTROLLER.is_armed
                else:
                    GLOBAL_STATE.is_armed = not GLOBAL_STATE.is_armed
                    msg = "Auto-Trading Armed" if GLOBAL_STATE.is_armed else "Auto-Trading Disarmed"
            self._send_json(200, {"success": True, "is_armed": GLOBAL_STATE.is_armed, "message": msg})
            return

        elif path == "/api/toggle_mode":
            with GLOBAL_STATE.lock:
                new_mode = "LIVE" if GLOBAL_STATE.execution_mode == "PAPER" else "PAPER"
                if GLOBAL_CONTROLLER:
                    msg = GLOBAL_CONTROLLER.toggle_mode(ExecutionMode(new_mode))
                    GLOBAL_STATE.execution_mode = GLOBAL_CONTROLLER.mode.value
                else:
                    GLOBAL_STATE.execution_mode = new_mode
                    msg = f"Mode toggled to {new_mode}"
            self._send_json(200, {"success": True, "mode": GLOBAL_STATE.execution_mode, "message": msg})
            return

        elif path == "/api/kill_switch":
            with GLOBAL_STATE.lock:
                if GLOBAL_CONTROLLER:
                    msg = GLOBAL_CONTROLLER.emergency_kill_switch()
                    GLOBAL_STATE.is_armed = False
                    GLOBAL_STATE.active_position = None
                else:
                    GLOBAL_STATE.is_armed = False
                    GLOBAL_STATE.active_position = None
                    msg = "Emergency Kill Switch Activated: All positions cleared."
            self._send_json(200, {"success": True, "message": msg})
            return

        self.send_error(404, "Unknown endpoint")

    def _send_json(self, status: int, data: Dict[str, Any]):
        content = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def render_dashboard_html(self) -> str:
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Directional Greeks - Quant Trader & Auto-Execution</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        brand: { 500: '#2563eb', 600: '#1d4ed8' },
                        bull: '#10b981',
                        bear: '#ef4444'
                    }
                }
            }
        }
    </script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans antialiased selection:bg-blue-600">

    <!-- Top Navbar -->
    <nav class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center font-black text-xl shadow-lg shadow-blue-500/20">
                    ⚡
                </div>
                <div>
                    <span class="font-extrabold text-lg tracking-tight bg-gradient-to-r from-blue-400 to-emerald-300 bg-clip-text text-transparent">Directional Greeks Quant Engine</span>
                    <span class="text-[10px] ml-2 px-2 py-0.5 rounded-full bg-blue-900/50 text-blue-300 border border-blue-700/50 font-bold uppercase tracking-wider">Live Tape & Auto-Trader</span>
                </div>
            </div>
            <div class="flex items-center space-x-4">
                <span class="text-xs text-slate-400">Daily P&L: <strong id="dailyPnlVal" class="text-emerald-400 font-bold text-sm">₹0.00</strong></span>
                <span id="marketStatusPill" class="px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-950 text-emerald-400 border border-emerald-800">● LIVE 09:15 Baseline</span>
            </div>
        </div>
    </nav>

    <main class="max-w-7xl mx-auto px-4 py-6 space-y-6">

        <!-- Top Control Panel & Auto-Trading Switch -->
        <div class="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-2xl flex flex-wrap items-center justify-between gap-4">
            <div class="flex items-center space-x-4">
                <div>
                    <h2 class="text-lg font-bold text-white flex items-center gap-2">
                        <span>🤖 Auto-Trading Controller</span>
                        <span id="armBadge" class="text-xs px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700 font-bold">DISARMED</span>
                    </h2>
                    <p class="text-xs text-slate-400 mt-0.5">Executes automatically on qualified skew momentum with dynamic trailing stop-loss protection.</p>
                </div>
            </div>

            <div class="flex items-center space-x-3">
                <!-- Mode Toggle -->
                <button id="modeBtn" onclick="toggleMode()" class="px-3.5 py-2 rounded-xl text-xs font-bold bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 transition">
                    Mode: <span id="modeVal" class="text-amber-400">PAPER</span>
                </button>

                <!-- Arm / Disarm Button -->
                <button id="armBtn" onclick="toggleArm()" class="px-5 py-2 rounded-xl text-xs font-extrabold bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-600/30 transition flex items-center gap-1.5">
                    <span>⚡</span> <span id="armBtnText">ARM AUTO-TRADING</span>
                </button>

                <!-- Emergency Kill Switch -->
                <button onclick="emergencyKill()" class="px-4 py-2 rounded-xl text-xs font-extrabold bg-rose-600/20 hover:bg-rose-600 text-rose-300 hover:text-white border border-rose-600/40 transition flex items-center gap-1">
                    <span>🚨</span> KILL SWITCH
                </button>
            </div>
        </div>

        <!-- 3-Column Real-Time Overview -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">

            <!-- 1. Quant Signal & Exact Trade Recommendation Card -->
            <div class="lg:col-span-2 bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden">
                <div class="flex items-center justify-between pb-4 border-b border-slate-800">
                    <div>
                        <span class="text-xs uppercase tracking-wider text-slate-500 font-bold">Quantitative Model Decision</span>
                        <h3 id="recommendationTitle" class="text-2xl font-black text-white mt-1 flex items-center gap-2">
                            <span id="signalBadge" class="px-3 py-1 rounded-xl text-sm bg-slate-800 text-slate-400">NO_TRADE</span>
                            <span id="contractSymbolVal" class="text-blue-400">Scanning Tape...</span>
                        </h3>
                    </div>
                    <div class="text-right">
                        <div class="text-[10px] text-slate-500 uppercase tracking-wider">Quantified Edge Score</div>
                        <div id="edgeScoreVal" class="text-2xl font-black text-emerald-400">-- / 100</div>
                        <div id="winProbVal" class="text-xs text-slate-400 mt-0.5">Est. Win Rate: --%</div>
                    </div>
                </div>

                <!-- Recommendation Parameters Grid -->
                <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 my-5">
                    <div class="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3">
                        <div class="text-[10px] text-slate-500 uppercase font-bold">Entry Price (LTP)</div>
                        <div id="entryPriceVal" class="text-lg font-extrabold text-slate-200 mt-1">₹0.00</div>
                        <div class="text-[10px] text-slate-500 mt-0.5">Near-Month ATM</div>
                    </div>

                    <div class="bg-rose-950/20 border border-rose-900/30 rounded-xl p-3">
                        <div class="text-[10px] text-rose-400 uppercase font-bold">Dynamic Stop Loss</div>
                        <div id="stopLossVal" class="text-lg font-extrabold text-rose-300 mt-1">₹0.00</div>
                        <div id="trailingSlVal" class="text-[10px] text-rose-400/80 mt-0.5">Trailing SL: ₹0.00</div>
                    </div>

                    <div class="bg-emerald-950/20 border border-emerald-900/30 rounded-xl p-3">
                        <div class="text-[10px] text-emerald-400 uppercase font-bold">Target 1 (1:1.5 RR)</div>
                        <div id="target1Val" class="text-lg font-extrabold text-emerald-300 mt-1">₹0.00</div>
                        <div id="expP1Val" class="text-[10px] text-emerald-400/80 mt-0.5">+₹0 / lot</div>
                    </div>

                    <div class="bg-emerald-950/40 border border-emerald-700/50 rounded-xl p-3">
                        <div class="text-[10px] text-emerald-300 uppercase font-bold">Target 2 (1:2.5 RR)</div>
                        <div id="target2Val" class="text-lg font-extrabold text-emerald-200 mt-1">₹0.00</div>
                        <div id="expP2Val" class="text-[10px] text-emerald-300/80 mt-0.5">+₹0 / lot</div>
                    </div>
                </div>

                <!-- Strategic Reason & Invalidation Alert -->
                <div class="space-y-2 text-xs">
                    <div class="p-3 bg-slate-950/80 border-l-4 border-blue-500 rounded-r-lg text-slate-300">
                        <span class="font-bold text-blue-400">Institutional Order Flow Rationale:</span>
                        <span id="rationaleText" class="ml-1 text-slate-300">Scanning constant-delta vega flows against 09:15 baseline.</span>
                    </div>

                    <div class="p-3 bg-slate-950/80 border-l-4 border-amber-500 rounded-r-lg text-slate-300">
                        <span class="font-bold text-amber-400">Greek Invalidation Protocol:</span>
                        <span id="invalidationText" class="ml-1 text-slate-400">Exit trade immediately if counter-wing Vega expands beyond deadband.</span>
                    </div>
                </div>
            </div>

            <!-- 2. Live Greek Flows & Skew Velocity Gauges -->
            <div class="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-5">
                <div class="flex items-center justify-between pb-3 border-b border-slate-800">
                    <h3 class="font-bold text-white text-sm">Constant-Delta Greeks Flow</h3>
                    <span id="lastUpdatedVal" class="text-[10px] text-slate-500">Updated: --</span>
                </div>

                <!-- Vega Flow Imbalance Meter -->
                <div>
                    <div class="flex justify-between text-xs mb-1">
                        <span class="text-slate-400">Vega Flow (CE vs PE)</span>
                        <span id="vfiText" class="font-bold text-slate-200">0.00</span>
                    </div>
                    <div class="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden flex">
                        <div id="peVegaBar" class="bg-rose-500 h-full transition-all duration-300" style="width: 50%"></div>
                        <div id="ceVegaBar" class="bg-emerald-500 h-full transition-all duration-300" style="width: 50%"></div>
                    </div>
                    <div class="flex justify-between text-[10px] text-slate-500 mt-1">
                        <span>PE Vega: <strong id="peVegaVal" class="text-rose-400">0.0</strong></span>
                        <span>CE Vega: <strong id="ceVegaVal" class="text-emerald-400">0.0</strong></span>
                    </div>
                </div>

                <!-- Skew Velocity -->
                <div class="p-3 bg-slate-950/60 border border-slate-800 rounded-xl">
                    <div class="flex justify-between text-xs">
                        <span class="text-slate-400">25Δ Skew Velocity</span>
                        <span id="skewShiftVal" class="font-bold text-slate-200">+0.00 vol pts</span>
                    </div>
                    <div class="text-[10px] text-slate-500 mt-1">Difference in Put-Call skew relative to 09:15 baseline.</div>
                </div>

                <!-- OI Writing Asymmetry -->
                <div class="p-3 bg-slate-950/60 border border-slate-800 rounded-xl">
                    <div class="flex justify-between text-xs">
                        <span class="text-slate-400">OI Writing Asymmetry</span>
                        <span id="oirVal" class="font-bold text-slate-200">0.00</span>
                    </div>
                    <div class="flex justify-between text-[10px] text-slate-500 mt-1">
                        <span>CE OI Δ: <strong id="ceOiVal">0.0M</strong></span>
                        <span>PE OI Δ: <strong id="peOiVal">0.0M</strong></span>
                    </div>
                </div>

                <!-- Underlying Price -->
                <div class="pt-2 border-t border-slate-800 flex justify-between items-center text-xs">
                    <span class="text-slate-400">Underlying Futures / Spot:</span>
                    <span id="spotPriceVal" class="text-sm font-extrabold text-blue-400">₹0.00</span>
                </div>
            </div>
        </div>

        <!-- 3. Active Positions & Execution Monitoring -->
        <div class="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl">
            <h3 class="font-bold text-white text-base mb-4 flex items-center justify-between">
                <span>Active Live Trade Position</span>
                <span id="posStatusBadge" class="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-400">FLAT (No Open Trades)</span>
            </h3>

            <div id="activePositionContainer" class="text-center py-6 text-slate-500 text-sm">
                No active trade position open. The quantitative engine is monitoring ticks.
            </div>
        </div>

    </main>

    <script>
        async function fetchState() {
            try {
                const res = await fetch('/api/state');
                const data = await res.json();
                renderUI(data);
            } catch (e) {
                console.error("Fetch state error:", e);
            }
        }

        function renderUI(data) {
            document.getElementById('dailyPnlVal').innerText = (data.daily_realized_pnl >= 0 ? '+₹' : '-₹') + Math.abs(data.daily_realized_pnl).toFixed(2);
            document.getElementById('dailyPnlVal').className = data.daily_realized_pnl >= 0 ? 'text-emerald-400 font-bold text-sm' : 'text-rose-400 font-bold text-sm';

            // Controller State
            const isArmed = data.is_armed;
            const armBadge = document.getElementById('armBadge');
            const armBtn = document.getElementById('armBtn');
            const armBtnText = document.getElementById('armBtnText');
            if (isArmed) {
                armBadge.innerText = "ARMED";
                armBadge.className = "text-xs px-2.5 py-0.5 rounded-full bg-emerald-900/50 text-emerald-400 border border-emerald-600 font-bold";
                armBtn.className = "px-5 py-2 rounded-xl text-xs font-extrabold bg-rose-600 hover:bg-rose-500 text-white shadow-lg shadow-rose-600/30 transition flex items-center gap-1.5";
                armBtnText.innerText = "DISARM AUTO-TRADER";
            } else {
                armBadge.innerText = "DISARMED";
                armBadge.className = "text-xs px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700 font-bold";
                armBtn.className = "px-5 py-2 rounded-xl text-xs font-extrabold bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-600/30 transition flex items-center gap-1.5";
                armBtnText.innerText = "ARM AUTO-TRADING";
            }

            document.getElementById('modeVal').innerText = data.execution_mode;
            document.getElementById('modeVal').className = data.execution_mode === 'LIVE' ? 'text-rose-400 font-bold' : 'text-amber-400 font-bold';

            // Greek Flows
            document.getElementById('spotPriceVal').innerText = '₹' + data.underlying_price.toFixed(2);
            document.getElementById('ceVegaVal').innerText = data.ce_vega.toFixed(1);
            document.getElementById('peVegaVal').innerText = data.pe_vega.toFixed(1);
            document.getElementById('ceOiVal').innerText = data.ce_oi.toFixed(1) + 'M';
            document.getElementById('peOiVal').innerText = data.pe_oi.toFixed(1) + 'M';
            document.getElementById('skewShiftVal').innerText = (data.skew_shift >= 0 ? '+' : '') + data.skew_shift.toFixed(2) + ' vol pts';
            document.getElementById('oirVal').innerText = data.oir.toFixed(2);
            document.getElementById('lastUpdatedVal').innerText = 'Updated: ' + data.last_update;

            const totalVega = Math.abs(data.ce_vega) + Math.abs(data.pe_vega) + 0.001;
            const cePct = (Math.abs(data.ce_vega) / totalVega) * 100;
            const pePct = (Math.abs(data.pe_vega) / totalVega) * 100;
            document.getElementById('ceVegaBar').style.width = cePct + '%';
            document.getElementById('peVegaBar').style.width = pePct + '%';

            // Recommendation
            const rec = data.recommendation;
            if (rec) {
                const sigBadge = document.getElementById('signalBadge');
                sigBadge.innerText = rec.signal;
                if (rec.signal === 'BUY_CALL') {
                    sigBadge.className = 'px-3 py-1 rounded-xl text-sm bg-emerald-600 text-white font-extrabold shadow-lg shadow-emerald-600/30';
                } else if (rec.signal === 'BUY_PUT') {
                    sigBadge.className = 'px-3 py-1 rounded-xl text-sm bg-rose-600 text-white font-extrabold shadow-lg shadow-rose-600/30';
                } else {
                    sigBadge.className = 'px-3 py-1 rounded-xl text-sm bg-slate-800 text-slate-400 font-bold';
                }

                document.getElementById('contractSymbolVal').innerText = rec.contract_symbol !== 'NONE' ? rec.contract_symbol : 'Neutral Market';
                document.getElementById('edgeScoreVal').innerText = rec.edge_score.toFixed(1) + ' / 100';
                document.getElementById('winProbVal').innerText = 'Est. Win Rate: ' + rec.win_rate_probability_pct.toFixed(1) + '%';
                document.getElementById('entryPriceVal').innerText = '₹' + rec.entry_price.toFixed(2);
                document.getElementById('stopLossVal').innerText = '₹' + rec.stop_loss.toFixed(2);
                document.getElementById('trailingSlVal').innerText = 'Trailing SL: ₹' + rec.trailing_stop_loss.toFixed(2);
                document.getElementById('target1Val').innerText = '₹' + rec.target_1.toFixed(2);
                document.getElementById('target2Val').innerText = '₹' + rec.target_2.toFixed(2);
                document.getElementById('expP1Val').innerText = '+₹' + rec.expected_profit_t1.toFixed(0) + ' / lot';
                document.getElementById('expP2Val').innerText = '+₹' + rec.expected_profit_t2.toFixed(0) + ' / lot';
                document.getElementById('rationaleText').innerText = rec.reason;
                document.getElementById('invalidationText').innerText = rec.invalidation_reason;
            }

            // Position
            const pos = data.active_position;
            const posContainer = document.getElementById('activePositionContainer');
            const posBadge = document.getElementById('posStatusBadge');
            if (pos && pos.status === 'OPEN') {
                posBadge.innerText = 'OPEN: ' + pos.contract_symbol;
                posBadge.className = 'text-xs px-2.5 py-0.5 rounded-full bg-blue-900/50 text-blue-300 border border-blue-600 font-bold';
                const pnlClass = pos.unrealized_pnl >= 0 ? 'text-emerald-400' : 'text-rose-400';
                posContainer.innerHTML = `
                    <div class="grid grid-cols-2 sm:grid-cols-6 gap-3 text-left">
                        <div class="bg-slate-950 p-3 rounded-xl border border-slate-800">
                            <div class="text-[10px] text-slate-500 uppercase">Contract</div>
                            <div class="font-extrabold text-sm text-white mt-0.5">${pos.contract_symbol}</div>
                        </div>
                        <div class="bg-slate-950 p-3 rounded-xl border border-slate-800">
                            <div class="text-[10px] text-slate-500 uppercase">Entry Price</div>
                            <div class="font-extrabold text-sm text-slate-200 mt-0.5">₹${pos.entry_price.toFixed(2)}</div>
                        </div>
                        <div class="bg-slate-950 p-3 rounded-xl border border-slate-800">
                            <div class="text-[10px] text-slate-500 uppercase">Live Price</div>
                            <div class="font-extrabold text-sm text-blue-400 mt-0.5">₹${pos.current_price.toFixed(2)}</div>
                        </div>
                        <div class="bg-slate-950 p-3 rounded-xl border border-slate-800">
                            <div class="text-[10px] text-slate-500 uppercase">Trailing Stop</div>
                            <div class="font-extrabold text-sm text-rose-300 mt-0.5">₹${pos.trailing_stop_loss.toFixed(2)}</div>
                        </div>
                        <div class="bg-slate-950 p-3 rounded-xl border border-slate-800">
                            <div class="text-[10px] text-slate-500 uppercase">Target 2</div>
                            <div class="font-extrabold text-sm text-emerald-300 mt-0.5">₹${pos.target_2.toFixed(2)}</div>
                        </div>
                        <div class="bg-slate-950 p-3 rounded-xl border border-slate-800">
                            <div class="text-[10px] text-slate-500 uppercase">Unrealized P&L</div>
                            <div class="font-black text-base ${pnlClass} mt-0.5">${pos.unrealized_pnl >= 0 ? '+₹' : '-₹'}${Math.abs(pos.unrealized_pnl).toFixed(2)}</div>
                        </div>
                    </div>
                `;
            } else {
                posBadge.innerText = 'FLAT (No Open Trades)';
                posBadge.className = 'text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-400';
                posContainer.innerHTML = `<p class="text-slate-500 text-sm">No active trade position open. The quantitative engine is monitoring live tape.</p>`;
            }
        }

        async function toggleArm() {
            await fetch('/api/toggle_arm', { method: 'POST' });
            fetchState();
        }

        async function toggleMode() {
            await fetch('/api/toggle_mode', { method: 'POST' });
            fetchState();
        }

        async function emergencyKill() {
            if (confirm("Are you sure you want to activate the EMERGENCY KILL SWITCH? This immediately cancels and exits all positions!")) {
                await fetch('/api/kill_switch', { method: 'POST' });
                fetchState();
            }
        }

        setInterval(fetchState, 2500);
        fetchState();
    </script>
</body>
</html>"""


def run_dashboard_server(port: int = 8050):
    server = HTTPServer(("0.0.0.0", port), QuantTraderHTTPHandler)
    print(f"\n🚀 Directional Greeks Quant & Auto-Trader Dashboard is live at: http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_dashboard_server()
