"""
Three cooperating networks:

1. PricePredictor  - per-asset LSTM regressor -> predicts next-minute return.
2. MarketAnalyser  - per-asset LSTM/MLP encoder -> regime embedding + auxiliary
                     trend/volatility classification heads.
3. TradeMaker      - actor-critic policy that CONSUMES the (detached) outputs
                     of the two networks above, plus raw market features and
                     current portfolio state, to decide a target allocation
                     across {cash, BTC, ETH, LTC} every minute.

All three are trained with separate optimizers (see train.py) so they can run
on separate GPUs/devices and literally train "in parallel" each epoch.
"""
import torch
import torch.nn as nn

from .indicators import FEATURE_COLUMNS

N_FEATURES = len(FEATURE_COLUMNS)
N_ASSETS = 3
TREND_CLASSES = 3     # down / flat / up
VOL_CLASSES = 3        # low / medium / high


class PricePredictor(nn.Module):
    def __init__(self, hidden=128, layers=2, pred_lookback=30):
        super().__init__()
        self.pred_lookback = pred_lookback
        self.lstm = nn.LSTM(N_FEATURES, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, window):
        # window: (B, L, F) -> use only the most recent pred_lookback steps
        x = window[:, -self.pred_lookback:, :]
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        pred_return = self.head(last).squeeze(-1)
        return pred_return, last  # scalar predicted next-minute return + hidden feat


class MarketAnalyser(nn.Module):
    def __init__(self, hidden=128, layers=2, embed=16):
        super().__init__()
        self.lstm = nn.LSTM(N_FEATURES, hidden, num_layers=layers, batch_first=True)
        self.embed_head = nn.Linear(hidden, embed)
        self.trend_head = nn.Linear(hidden, TREND_CLASSES)
        self.vol_head = nn.Linear(hidden, VOL_CLASSES)

    def forward(self, window):
        # window: (B, L, F) full lookback window
        out, _ = self.lstm(window)
        last = out[:, -1, :]
        embed = torch.tanh(self.embed_head(last))
        trend_logits = self.trend_head(last)
        vol_logits = self.vol_head(last)
        return embed, trend_logits, vol_logits


class TradeMaker(nn.Module):
    """
    Actor-critic over a continuous allocation action in R^4 (pre-softmax
    logits for [cash, BTC, ETH, LTC]). PPO samples logits ~ Normal(mu, std)
    then applies softmax to get the actual target weights.
    """

    def __init__(self, per_asset_in_dim, portfolio_dim=5, hidden=256, n_assets=N_ASSETS):
        super().__init__()
        in_dim = per_asset_in_dim * n_assets + portfolio_dim
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor_mu = nn.Linear(hidden, n_assets + 1)      # cash + 3 assets
        self.actor_logstd = nn.Parameter(torch.zeros(n_assets + 1) - 0.5)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        mu = self.actor_mu(h)
        std = torch.exp(self.actor_logstd).clamp(min=1e-3)
        value = self.critic(h).squeeze(-1)
        return mu, std, value
