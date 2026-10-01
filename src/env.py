"""
Single-day multi-asset trading environment.

- Starting capital: $20 (configurable).
- Goal baked into the reward: reach >= $30 (1.5x) by day's end.
- Action: target portfolio weights over [cash, BTC, ETH, LTC] (sums to 1).
- Transaction costs applied on rebalancing to discourage churn.
- Episode = one simulated trading day (1440 minutes), randomly sampled.
"""
import numpy as np

from .data import SYMBOLS, MINUTES_PER_DAY

START_CAPITAL = 20.0
TARGET_CAPITAL = 30.0
TXN_FEE = 0.0008            # 8 bps per rebalance, roughly like a retail exchange taker fee
BANKRUPTCY_FLOOR = 3.0      # end episode early (big penalty) if value collapses


class TradingEnv:
    def __init__(self, start_capital=START_CAPITAL, target_capital=TARGET_CAPITAL):
        self.start_capital = start_capital
        self.target_capital = target_capital
        self.n_assets = len(SYMBOLS)

    def reset(self, day_arrays: dict):
        self.day_arrays = day_arrays
        self.t = 0
        self.cash = self.start_capital
        self.holdings = np.zeros(self.n_assets, dtype=np.float64)  # units held per asset
        self.history = {"net_worth": [], "weights": [], "prices": []}
        prices = self._prices_at(self.t)
        self.history["prices"].append(prices.copy())
        return self._portfolio_state(prices)

    def _prices_at(self, t):
        return np.array([self.day_arrays[s]["close"][t] for s in SYMBOLS], dtype=np.float64)

    def net_worth(self, prices):
        return self.cash + float(np.dot(self.holdings, prices))

    def _portfolio_state(self, prices):
        nw = self.net_worth(prices)
        asset_vals = self.holdings * prices
        fracs = asset_vals / (nw + 1e-9)
        cash_frac = self.cash / (nw + 1e-9)
        norm_nw = nw / self.start_capital
        return np.concatenate([[cash_frac], fracs, [norm_nw]]).astype(np.float32)

    def step(self, target_weights: np.ndarray):
        """
        target_weights: length n_assets+1 array (cash, btc, eth, ltc), already
        softmax-normalized to sum to 1.
        """
        prices = self._prices_at(self.t)
        nw_before = self.net_worth(prices)

        target_values = target_weights * nw_before
        target_cash = target_values[0]
        target_asset_vals = target_values[1:]
        current_asset_vals = self.holdings * prices

        traded_notional = np.sum(np.abs(target_asset_vals - current_asset_vals))
        fee = traded_notional * TXN_FEE

        self.holdings = np.where(prices > 0, target_asset_vals / (prices + 1e-9), self.holdings)
        self.cash = target_cash - fee

        # advance time
        self.t += 1
        done = self.t >= MINUTES_PER_DAY - 1
        prices_next = self._prices_at(self.t)
        nw_after = self.net_worth(prices_next)

        reward = np.log(max(nw_after, 1e-6) / max(nw_before, 1e-6))

        bankrupt = nw_after <= BANKRUPTCY_FLOOR
        if bankrupt:
            reward -= 2.0
            done = True

        if done:
            if nw_after >= self.target_capital:
                overshoot = (nw_after - self.target_capital) / self.target_capital
                reward += 1.0 + min(overshoot, 2.0)
            else:
                shortfall = (self.target_capital - nw_after) / self.target_capital
                reward -= min(shortfall, 1.0)

        self.history["net_worth"].append(nw_after)
        self.history["weights"].append(target_weights.copy())
        self.history["prices"].append(prices_next.copy())

        obs = self._portfolio_state(prices_next)
        info = {"net_worth": nw_after, "bankrupt": bankrupt}
        return obs, reward, done, info
