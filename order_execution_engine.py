"""
Production Automated Execution & Risk Management Controller for Directional Greeks.

Provides:
- Auto-Trading Arm/Disarm safety toggle
- Paper Trading Mode (default) vs Live Broker Mode (Zerodha Kite Connect)
- Dynamic Trailing Stop-Loss and Multi-Tier Target management
- Pre-trade and in-trade risk limits (Max daily loss, max trades, cooldown)
- Emergency Instant Kill Switch
- Persistent CSV trade logging
"""

import os
import csv
import time
from dataclasses import dataclass, field
from datetime import datetime, date, time as dtime
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np

from quant_trade_engine import TradeRecommendation, TradeSignalType


class ExecutionMode(str, Enum):
    PAPER = "PAPER"
    LIVE = "LIVE"


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


@dataclass
class Position:
    """Live open or historical closed position."""
    trade_id: str
    symbol: str                                    # e.g. "NIFTY"
    contract_symbol: str                           # e.g. "NIFTY26OCT24500CE"
    instrument_token: int
    option_type: str                               # CE / PE
    strike: float
    lot_size: int
    quantity: int
    entry_price: float
    entry_time: datetime
    stop_loss: float
    trailing_stop_loss: float
    target_1: float
    target_2: float
    current_price: float = 0.0
    highest_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    exit_reason: str = "OPEN"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "contract_symbol": self.contract_symbol,
            "option_type": self.option_type,
            "strike": self.strike,
            "quantity": self.quantity,
            "entry_price": round(self.entry_price, 2),
            "entry_time": self.entry_time.strftime("%H:%M:%S"),
            "current_price": round(self.current_price, 2),
            "stop_loss": round(self.stop_loss, 2),
            "trailing_stop_loss": round(self.trailing_stop_loss, 2),
            "target_1": round(self.target_1, 2),
            "target_2": round(self.target_2, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "status": self.status.value,
            "exit_price": round(self.exit_price, 2) if self.exit_price else 0.0,
            "exit_time": self.exit_time.strftime("%H:%M:%S") if self.exit_time else "",
            "exit_reason": self.exit_reason,
        }


class AutoTradingController:
    """
    Automated Trading & Risk Management Controller.
    
    Can be armed by the user. If disarmed, it runs in recommendation-only mode.
    """

    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.PAPER,
        kite_client: Optional[Any] = None,
        max_loss_per_day: float = 5000.0,      # Maximum daily allowable loss in ₹
        max_trades_per_day: int = 4,           # Max trade entries per day
        max_open_positions: int = 1,           # Max concurrent positions
        cooldown_seconds: int = 180,           # Cooldown window after an exit (~3 mins)
        log_csv_path: str = "trades_audit_log.csv"
    ):
        self.mode = mode
        self.kite = kite_client
        self.is_armed = False                   # Must be explicitly armed by user
        self.max_loss_per_day = max_loss_per_day
        self.max_trades_per_day = max_trades_per_day
        self.max_open_positions = max_open_positions
        self.cooldown_seconds = cooldown_seconds
        self.log_csv_path = log_csv_path

        # State tracking
        self.active_positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.daily_realized_pnl: float = 0.0
        self.trades_taken_today: int = 0
        self.last_exit_time: float = 0.0
        self.circuit_breaker_tripped: bool = False

        # Initialize audit CSV
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.log_csv_path):
            with open(self.log_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Trade_ID", "Mode", "Symbol", "Contract",
                    "Action", "Price", "Qty", "SL", "Trailing_SL",
                    "Target1", "Target2", "Realized_PnL", "Exit_Reason"
                ])

    def arm(self) -> str:
        """Arms the auto-trading engine."""
        if self.circuit_breaker_tripped:
            return "❌ Cannot arm: Daily loss circuit breaker is tripped."
        self.is_armed = True
        return f"⚡ Auto-Trading ARMED in {self.mode.value} mode. Monitoring for live setups."

    def disarm(self) -> str:
        """Disarms auto-trading. Active positions remain tracked, but no new trades open."""
        self.is_armed = False
        return "🛑 Auto-Trading DISARMED. Recommendation-only mode active."

    def toggle_mode(self, new_mode: ExecutionMode) -> str:
        """Switch between PAPER and LIVE execution."""
        if self.active_positions:
            return "❌ Cannot switch execution mode while open positions exist."
        self.mode = new_mode
        return f"Execution mode updated to: {self.mode.value}"

    def emergency_kill_switch(self) -> str:
        """
        Immediately disarms the engine and closes all open positions at market.
        """
        self.is_armed = False
        closed_count = 0
        for symbol, pos in list(self.active_positions.items()):
            self._close_position(pos, pos.current_price, "EMERGENCY_KILL_SWITCH")
            closed_count += 1
        return f"🚨 EMERGENCY KILL SWITCH TRIGGERED: Disarmed. {closed_count} positions closed at market."

    def on_cycle(
        self,
        recommendation: TradeRecommendation,
        ticks: Dict[int, Dict[str, float]],
        now: datetime
    ) -> Dict[str, Any]:
        """
        Processes a live tick cycle:
        1. Updates open positions (mark-to-market, trailing SL, target hits, invalidation exits).
        2. Evaluates new trade entries if flat and auto-trader is ARMED.
        """
        symbol = recommendation.symbol
        token = recommendation.instrument_token
        tick = ticks.get(token, {})
        live_price = tick.get("px", 0.0)

        status_msg = "Idle"

        # 1. Update Active Position
        if symbol in self.active_positions:
            pos = self.active_positions[symbol]
            if live_price > 0:
                pos.current_price = live_price
                pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.quantity

                if pos.current_price > pos.highest_price:
                    pos.highest_price = pos.current_price

                # Dynamic Trailing Stop Loss adjustment
                profit_run = pos.highest_price - pos.entry_price
                risk_amount = pos.entry_price - pos.stop_loss
                
                # If gained past 1:1.5 RR (Target 1), trail SL to lock 50% of peak gains
                if profit_run >= (risk_amount * 1.5):
                    new_trailing = pos.entry_price + (profit_run * 0.50)
                    if new_trailing > pos.trailing_stop_loss:
                        pos.trailing_stop_loss = new_trailing
                elif profit_run >= (risk_amount * 0.60):
                    # Move to Breakeven
                    if pos.entry_price > pos.trailing_stop_loss:
                        pos.trailing_stop_loss = pos.entry_price

            # Exit Checks
            # A. Target 2 Hit (Full Profit)
            if pos.current_price >= pos.target_2 and pos.target_2 > 0:
                self._close_position(pos, pos.target_2, "TARGET_2_HIT (1:2.5 RR)")
                status_msg = f"🏆 Target 2 Hit on {pos.contract_symbol} at ₹{pos.target_2}!"

            # B. Trailing Stop Hit (Profit Protection)
            elif pos.current_price <= pos.trailing_stop_loss and pos.trailing_stop_loss > pos.stop_loss:
                self._close_position(pos, pos.trailing_stop_loss, "TRAILING_STOP_HIT (Profit Locked)")
                status_msg = f"🔒 Trailing SL Hit on {pos.contract_symbol} at ₹{pos.trailing_stop_loss}."

            # C. Initial Stop Loss Hit
            elif pos.current_price <= pos.stop_loss and pos.stop_loss > 0:
                self._close_position(pos, pos.stop_loss, "STOP_LOSS_HIT")
                status_msg = f"🛑 Stop Loss Hit on {pos.contract_symbol} at ₹{pos.stop_loss}."

            # D. Greek Invalidation Exit (Institutional flow reversed)
            elif "⚠️ Immediate Exit" in recommendation.invalidation_reason:
                self._close_position(pos, pos.current_price, f"GREEK_INVALIDATION: {recommendation.invalidation_reason}")
                status_msg = f"⚠️ Greek Invalidation Exit on {pos.contract_symbol}."

            # E. End of Day Square-Off (15:15 IST)
            elif now.time() >= dtime(15, 15, 0):
                self._close_position(pos, pos.current_price, "EOD_SQUARE_OFF (15:15 IST)")
                status_msg = f"⏰ End-of-Day Square-Off for {pos.contract_symbol}."

            return {
                "action": "MONITORING_POSITION",
                "message": status_msg,
                "position": pos.to_dict(),
                "daily_realized_pnl": self.daily_realized_pnl,
                "is_armed": self.is_armed,
            }

        # 2. Check for New Trade Entry
        if not self.is_armed:
            return {
                "action": "RECOMMENDATION_ONLY",
                "message": "Auto-trader is DISARMED. Signal displayed for user decision.",
                "recommendation": recommendation.to_dict(),
                "daily_realized_pnl": self.daily_realized_pnl,
                "is_armed": self.is_armed,
            }

        # Safety Gates
        if self.circuit_breaker_tripped:
            return {
                "action": "CIRCUIT_BREAKER",
                "message": f"Daily loss limit (-₹{self.max_loss_per_day:,.2f}) hit. Trading halted.",
                "is_armed": False
            }

        if self.daily_realized_pnl <= -self.max_loss_per_day:
            self.circuit_breaker_tripped = True
            self.is_armed = False
            return {
                "action": "CIRCUIT_BREAKER_TRIGGERED",
                "message": f"Daily loss limit (-₹{self.max_loss_per_day:,.2f}) breached. Disarmed.",
                "is_armed": False
            }

        if self.trades_taken_today >= self.max_trades_per_day:
            return {
                "action": "MAX_TRADES_HIT",
                "message": f"Max daily trades ({self.max_trades_per_day}) reached.",
                "is_armed": self.is_armed
            }

        if time.time() - self.last_exit_time < self.cooldown_seconds:
            remaining = int(self.cooldown_seconds - (time.time() - self.last_exit_time))
            return {
                "action": "COOLDOWN",
                "message": f"In post-trade cooldown ({remaining}s remaining).",
                "is_armed": self.is_armed
            }

        if len(self.active_positions) >= self.max_open_positions:
            return {
                "action": "MAX_POSITIONS",
                "message": f"Max concurrent positions ({self.max_open_positions}) active.",
                "is_armed": self.is_armed
            }

        # Evaluate Qualified Entry Signal
        if recommendation.signal in (TradeSignalType.BUY_CALL, TradeSignalType.BUY_PUT):
            if recommendation.entry_price > 0 and recommendation.contract_symbol != "NONE":
                pos = self._execute_entry(recommendation, now)
                return {
                    "action": "ENTRY_EXECUTED",
                    "message": f"🚀 {recommendation.signal.value} EXECUTED: {pos.contract_symbol} @ ₹{pos.entry_price:.2f}",
                    "position": pos.to_dict(),
                    "daily_realized_pnl": self.daily_realized_pnl,
                    "is_armed": self.is_armed
                }

        return {
            "action": "WAITING_FOR_EDGE",
            "message": "Scanning live Greek flows for high-probability skew divergence.",
            "is_armed": self.is_armed
        }

    def _execute_entry(self, rec: TradeRecommendation, now: datetime) -> Position:
        """Executes trade entry in Paper or Live mode."""
        qty = rec.lot_size
        trade_id = f"TRD_{rec.symbol}_{int(time.time())}"

        pos = Position(
            trade_id=trade_id,
            symbol=rec.symbol,
            contract_symbol=rec.contract_symbol,
            instrument_token=rec.instrument_token,
            option_type=rec.option_type,
            strike=rec.strike,
            lot_size=rec.lot_size,
            quantity=qty,
            entry_price=rec.entry_price,
            entry_time=now,
            stop_loss=rec.stop_loss,
            trailing_stop_loss=rec.trailing_stop_loss,
            target_1=rec.target_1,
            target_2=rec.target_2,
            current_price=rec.entry_price,
            highest_price=rec.entry_price,
            status=PositionStatus.OPEN
        )

        # Place live order if armed in LIVE mode
        if self.mode == ExecutionMode.LIVE and self.kite is not None:
            try:
                # Zerodha Kite Connect live order placement
                seg = "NFO" if rec.symbol != "SENSEX" else "BFO"
                order_id = self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=seg,
                    tradingsymbol=rec.contract_symbol,
                    transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                    quantity=qty,
                    order_type=self.kite.ORDER_TYPE_MARKET,
                    product=self.kite.PRODUCT_MIS
                )
                print(f"[LIVE BROKER] Order Placed: {order_id} for {rec.contract_symbol}")
            except Exception as e:
                print(f"[LIVE BROKER ERROR] Failed to place order: {e}")

        self.active_positions[rec.symbol] = pos
        self.trades_taken_today += 1

        # Audit log
        self._log_trade(pos, "BUY_ENTRY", rec.entry_price, 0.0, "SIGNAL_ENTERED")
        return pos

    def _close_position(self, pos: Position, exit_price: float, reason: str):
        """Closes position and locks P&L."""
        now = datetime.now()
        pos.exit_price = exit_price
        pos.exit_time = now
        pos.exit_reason = reason
        pos.status = PositionStatus.CLOSED
        pos.realized_pnl = (exit_price - pos.entry_price) * pos.quantity
        pos.unrealized_pnl = 0.0

        # Place live market sell order if in LIVE mode
        if self.mode == ExecutionMode.LIVE and self.kite is not None:
            try:
                seg = "NFO" if pos.symbol != "SENSEX" else "BFO"
                order_id = self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=seg,
                    tradingsymbol=pos.contract_symbol,
                    transaction_type=self.kite.TRANSACTION_TYPE_SELL,
                    quantity=pos.quantity,
                    order_type=self.kite.ORDER_TYPE_MARKET,
                    product=self.kite.PRODUCT_MIS
                )
                print(f"[LIVE BROKER] Sell Order Placed: {order_id} for {pos.contract_symbol}")
            except Exception as e:
                print(f"[LIVE BROKER ERROR] Failed to close order: {e}")

        self.daily_realized_pnl += pos.realized_pnl
        self.last_exit_time = time.time()
        self.closed_positions.append(pos)
        self.active_positions.pop(pos.symbol, None)

        # Audit log
        self._log_trade(pos, "SELL_EXIT", exit_price, pos.realized_pnl, reason)

    def _log_trade(self, pos: Position, action: str, price: float, pnl: float, reason: str):
        try:
            with open(self.log_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    pos.trade_id,
                    self.mode.value,
                    pos.symbol,
                    pos.contract_symbol,
                    action,
                    round(price, 2),
                    pos.quantity,
                    round(pos.stop_loss, 2),
                    round(pos.trailing_stop_loss, 2),
                    round(pos.target_1, 2),
                    round(pos.target_2, 2),
                    round(pnl, 2),
                    reason
                ])
        except Exception as e:
            print(f"[LOG ERROR] {e}")
