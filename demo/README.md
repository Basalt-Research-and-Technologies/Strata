"""
demo/README.md
==============
Self-contained demo for basalt-strata.
"""

# Basalt Strata — Demo

Run this once to see the full library end-to-end with no setup:

```bash
pip install basalt-strata
py demo/run_demo.py
```

That's it. The demo:
- Generates 3 years of synthetic Nifty data (no CSV needed)
- Runs EMA crossover, RSI reversion, and Dual Momentum strategies
- Prints a side-by-side comparison table
- Saves tearsheets to `demo/outputs/`

All output stays inside `demo/outputs/`. Nothing is written elsewhere.
