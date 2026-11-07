# Moneyline: winner model vs line movement (evaluation only)

Data: ESPN core API open/close moneylines (DraftKings/Caesars/ESPN BET preferred), available 2023-25 only
(2019-22 have no open/close). 3,686 games matched to the out-of-sample winner model. Betting splits
(ticket %/handle %) have no free historical source, so they were not tested.

## 70%+ winner predictions (1,380 games)
| Line move (no-vig, our team) | n | won | ROI at close | ROI at open |
|---|---|---|---|---|
| moved against us (2+ pts) | 250 | 78.0% | +7.1% | -1.1% |
| flat | 585 | 82.7% | -0.1% | -0.1% |
| moved toward us (2+ pts) | 545 | 82.0% | +1.1% | +10.8% |
All 70%+: won 81.6%, ROI +1.7% at close, +4.0% at open. Movement direction does not change who wins;
ROI gap toward vs against -5.9 pts vs placebo SD 4.5 (noise).

## Model vs closing market (any confidence, bet the side the model rates higher)
| model minus no-vig close | bets | ROI | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|
| >= 3 pts | 2,520 | +0.2% | -0.2% | +2.4% | -1.5% |
| >= 5 pts | 1,906 | +1.1% | +1.3% | +1.7% | +0.2% |
| >= 8 pts | 1,231 | +5.2% | +8.7% | +3.3% | +3.7% |
| >= 10 pts | 860 | +6.5% | -2.1% | +8.4% | +13.3% |

Caveat: the backtest's missing-player input uses who actually sat (late scratches are known to the model
but may not be at the 1-hour lock); only 3 seasons. Treat as a paper-track candidate, not a pick rule.
