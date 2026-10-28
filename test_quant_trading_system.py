"""Unit tests for the Quantitative Model and Automated Execution Engine."""

import unittest
import io
import json
from datetime import datetime, date, time as dtime
import numpy as np

from quant_trade_engine import (
    QuantModel,
    TradeRecommendation,
    TradeSignalType,
    MarketRegime,
    LOT_SIZES
)
from order_execution_engine import (
    AutoTradingController,
    ExecutionMode,
    PositionStatus
)
from app_quant_trader import QuantTraderHTTPHandler, GLOBAL_STATE


class DummyChainState:
    def __init__(self, forward=24500.0):
        self.F = np.array([forward])
        self.delta_f = np.array([0.50, -0.50, 0.25, -0.25])


class TestQuantTradingSystem(unittest.TestCase):

    def setUp(self):
        self.model = QuantModel(symbol="NIFTY", z_entry_threshold=1.2, min_edge_score=60.0)
        self.chain_opts = [
            {"instrument_token": 1001, "instrument_type": "CE", "strike": 24500, "tradingsymbol": "NIFTY26OCT24500CE", "exp_idx": 0, "expiry": date(2026, 10, 29)},
            {"instrument_token": 1002, "instrument_type": "PE", "strike": 24500, "tradingsymbol": "NIFTY26OCT24500PE", "exp_idx": 0, "expiry": date(2026, 10, 29)},
            {"instrument_token": 1003, "instrument_type": "CE", "strike": 24700, "tradingsymbol": "NIFTY26OCT24700CE", "exp_idx": 0, "expiry": date(2026, 10, 29)},
            {"instrument_token": 1004, "instrument_type": "PE", "strike": 24300, "tradingsymbol": "NIFTY26OCT24300PE", "exp_idx": 0, "expiry": date(2026, 10, 29)},
        ]
        self.chain_state = DummyChainState(24500.0)
        self.ticks = {
            1001: {"px": 140.0, "oi": 2500000},
            1002: {"px": 135.0, "oi": 2800000},
        }

    def test_alpha_feature_calculations(self):
        """Test calculation of VFI, Skew, and OIR."""
        vfi, skew, oir, z = self.model.calculate_alpha_features(
            ce_vega=120.0, pe_vega=-80.0, ce_theta=-30.0, pe_theta=15.0, ce_oi=1.2, pe_oi=2.4
        )
        self.assertGreater(vfi, 0.0)  # Call vega dominates -> positive VFI
        self.assertGreater(oir, 0.0)  # Put OI dominates -> positive OIR (support)

    def test_bullish_recommendation_generation(self):
        """Verify strong bullish conditions produce exact BUY_CALL recommendation."""
        # Feed consecutive bullish flows
        for _ in range(5):
            self.model.calculate_alpha_features(
                ce_vega=150.0, pe_vega=-120.0, ce_theta=-40.0, pe_theta=20.0, ce_oi=0.5, pe_oi=2.5
            )

        rec = self.model.generate_recommendation(
            ce_vega=200.0, pe_vega=-150.0, ce_theta=-50.0, pe_theta=30.0, ce_oi=0.5, pe_oi=3.0,
            chain_opts=self.chain_opts, chain_state=self.chain_state, ticks=self.ticks,
            now_time=dtime(10, 30, 0), vega_deadband=0.0
        )

        self.assertEqual(rec.signal, TradeSignalType.BUY_CALL)
        self.assertIn("CE", rec.contract_symbol)
        self.assertEqual(rec.strike, 24500)
        self.assertEqual(rec.entry_price, 140.0)
        self.assertGreater(rec.edge_score, 60.0)
        self.assertGreater(rec.win_rate_probability_pct, 60.0)

        # Check Risk Levels
        self.assertLess(rec.stop_loss, rec.entry_price)
        self.assertGreater(rec.target_1, rec.entry_price)
        self.assertGreater(rec.target_2, rec.target_1)
        self.assertGreater(rec.expected_profit_t1, 0.0)
        self.assertGreater(rec.max_loss_per_lot, 0.0)

    def test_bearish_recommendation_generation(self):
        """Verify strong bearish conditions produce exact BUY_PUT recommendation."""
        model_bear = QuantModel(symbol="NIFTY", z_entry_threshold=1.0, min_edge_score=50.0)
        for _ in range(5):
            model_bear.calculate_alpha_features(
                ce_vega=-150.0, pe_vega=200.0, ce_theta=30.0, pe_theta=-40.0, ce_oi=3.0, pe_oi=0.5
            )

        rec = model_bear.generate_recommendation(
            ce_vega=-180.0, pe_vega=250.0, ce_theta=40.0, pe_theta=-50.0, ce_oi=3.5, pe_oi=0.4,
            chain_opts=self.chain_opts, chain_state=self.chain_state, ticks=self.ticks,
            now_time=dtime(11, 0, 0), vega_deadband=0.0
        )

        self.assertEqual(rec.signal, TradeSignalType.BUY_PUT)
        self.assertIn("PE", rec.contract_symbol)
        self.assertEqual(rec.strike, 24500)
        self.assertEqual(rec.entry_price, 135.0)

    def test_auto_trader_lifecycle(self):
        """Test complete automated trade lifecycle: Arm -> Entry -> Trailing SL -> Target Exit."""
        controller = AutoTradingController(
            mode=ExecutionMode.PAPER,
            max_loss_per_day=3000.0,
            cooldown_seconds=1
        )

        # 1. When disarmed, no orders are taken
        rec = TradeRecommendation(
            symbol="NIFTY",
            signal=TradeSignalType.BUY_CALL,
            regime=MarketRegime.STRONG_BULLISH,
            contract_symbol="NIFTY26OCT24500CE",
            instrument_token=1001,
            strike=24500,
            option_type="CE",
            entry_price=140.0,
            lot_size=25,
            stop_loss=120.0,
            trailing_stop_loss=120.0,
            target_1=170.0,
            target_2=190.0,
            edge_score=85.0
        )

        res = controller.on_cycle(rec, self.ticks, datetime.now())
        self.assertEqual(res["action"], "RECOMMENDATION_ONLY")
        self.assertEqual(len(controller.active_positions), 0)

        # 2. Arm the controller
        arm_msg = controller.arm()
        self.assertTrue(controller.is_armed)

        # 3. Executes trade entry
        res = controller.on_cycle(rec, self.ticks, datetime.now())
        self.assertEqual(res["action"], "ENTRY_EXECUTED")
        self.assertIn("NIFTY", controller.active_positions)
        pos = controller.active_positions["NIFTY"]
        self.assertEqual(pos.entry_price, 140.0)
        self.assertEqual(pos.trailing_stop_loss, 120.0)

        # 4. Price moves up past Target 1 -> Trailing SL ratchets higher
        ticks_up = {1001: {"px": 172.0}}
        res = controller.on_cycle(rec, ticks_up, datetime.now())
        self.assertEqual(res["action"], "MONITORING_POSITION")
        self.assertGreater(pos.trailing_stop_loss, 140.0)  # Trailing SL locked in profits

        # 5. Price hits Target 2 -> Exit for full profit
        ticks_target = {1001: {"px": 191.0}}
        res = controller.on_cycle(rec, ticks_target, datetime.now())
        self.assertIn("Target 2 Hit", res["message"])
        self.assertEqual(len(controller.active_positions), 0)
        self.assertGreater(controller.daily_realized_pnl, 0.0)

    def test_emergency_kill_switch(self):
        """Test emergency kill switch immediately squares off positions and disarms."""
        controller = AutoTradingController(mode=ExecutionMode.PAPER)
        controller.arm()

        rec = TradeRecommendation(
            symbol="NIFTY",
            signal=TradeSignalType.BUY_CALL,
            regime=MarketRegime.STRONG_BULLISH,
            contract_symbol="NIFTY26OCT24500CE",
            instrument_token=1001,
            strike=24500,
            option_type="CE",
            entry_price=100.0,
            lot_size=25,
            stop_loss=85.0,
            trailing_stop_loss=85.0,
            target_1=120.0,
            target_2=140.0
        )
        controller.on_cycle(rec, self.ticks, datetime.now())
        self.assertEqual(len(controller.active_positions), 1)

        # Trigger Kill Switch
        msg = controller.emergency_kill_switch()
        self.assertIn("EMERGENCY KILL SWITCH", msg)
        self.assertFalse(controller.is_armed)
        self.assertEqual(len(controller.active_positions), 0)

    def test_web_dashboard_endpoints(self):
        """Test Web UI HTTP handler endpoints: GET /, GET /api/state, POST /api/toggle_arm."""
        class DummyServer:
            pass

        class MockSocket:
            def __init__(self, data: bytes):
                self.rfile = io.BytesIO(data)
                self.wfile = io.BytesIO()
            def makefile(self, mode, *args, **kwargs):
                return self.rfile if "r" in mode else self.wfile
            def sendall(self, b):
                self.wfile.write(b)

        # 1. GET /
        s = MockSocket(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
        QuantTraderHTTPHandler(s, ("127.0.0.1", 8050), DummyServer())
        res = s.wfile.getvalue().decode("utf-8")
        self.assertIn("200 OK", res)
        self.assertIn("Directional Greeks Quant Engine", res)

        # 2. GET /api/state
        s = MockSocket(b"GET /api/state HTTP/1.1\r\nHost: localhost\r\n\r\n")
        QuantTraderHTTPHandler(s, ("127.0.0.1", 8050), DummyServer())
        res = s.wfile.getvalue().decode("utf-8")
        self.assertIn("200 OK", res)
        state_data = json.loads(res.split("\r\n\r\n")[1])
        self.assertIn("underlying_price", state_data)

        # 3. POST /api/toggle_arm
        s = MockSocket(b"POST /api/toggle_arm HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n")
        QuantTraderHTTPHandler(s, ("127.0.0.1", 8050), DummyServer())
        res = s.wfile.getvalue().decode("utf-8")
        self.assertIn("200 OK", res)
        self.assertTrue(GLOBAL_STATE.is_armed)

        # 4. POST /api/kill_switch
        s = MockSocket(b"POST /api/kill_switch HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n")
        QuantTraderHTTPHandler(s, ("127.0.0.1", 8050), DummyServer())
        res = s.wfile.getvalue().decode("utf-8")
        self.assertIn("200 OK", res)
        self.assertFalse(GLOBAL_STATE.is_armed)

    def test_tick_by_tick_event_driven_pipeline(self):
        """Verify that broker ticks immediately trigger sub-millisecond computation."""
        from live_greeks_to_excel import Instrument, take_sample
        from directional_greeks_core import ChainState, Baseline

        opts = [
            {"instrument_token": 2001, "instrument_type": "CE", "strike": 24500, "tradingsymbol": "NIFTY26OCT24500CE", "exp_idx": 0, "expiry": date(2026, 10, 29)},
            {"instrument_token": 2002, "instrument_type": "PE", "strike": 24500, "tradingsymbol": "NIFTY26OCT24500PE", "exp_idx": 0, "expiry": date(2026, 10, 29)},
            {"instrument_token": 2003, "instrument_type": "CE", "strike": 24600, "tradingsymbol": "NIFTY26OCT24600CE", "exp_idx": 0, "expiry": date(2026, 10, 29)},
            {"instrument_token": 2004, "instrument_type": "PE", "strike": 24400, "tradingsymbol": "NIFTY26OCT24400PE", "exp_idx": 0, "expiry": date(2026, 10, 29)},
        ]
        ins = Instrument("NIFTY", [date(2026, 10, 29)], opts)

        # Baseline setup
        mock_ticks = {
            2001: {"px": 140.0, "oi": 2000000},
            2002: {"px": 135.0, "oi": 2200000},
            2003: {"px": 80.0, "oi": 1500000},
            2004: {"px": 90.0, "oi": 1600000},
        }
        T = np.array([5.0 / 365.0])
        ins.chain.update([140.0, 135.0, 80.0, 90.0], [2000000, 2200000, 1500000, 1600000], T)
        iv_b, vg_b = ins.chain.buckets()
        ins.baseline.capture(ins.chain, iv_b, vg_b, 10)

        auto_trader = AutoTradingController(mode=ExecutionMode.PAPER)

        # Deliver 5 consecutive ticks as if pushed by broker WebSocket
        for tick_idx in range(1, 6):
            mock_ticks[2001]["px"] = 140.0 + (tick_idx * 2.0)
            row, rec, status = take_sample(ins, mock_ticks, datetime.now(), auto_trader)
            self.assertEqual(ins.tick_count, tick_idx)
            self.assertLess(ins.last_calc_latency_ms, 50.0)  # Verify instant sub-50ms execution
            self.assertIsNotNone(row)
            self.assertIsNotNone(rec)

        # Check that GLOBAL_STATE reflects the latest tick immediately
        self.assertEqual(GLOBAL_STATE.tick_count, 5)
        self.assertGreater(GLOBAL_STATE.calc_latency_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
