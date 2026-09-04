# QuantEngine

A professional-grade quantitative finance library for pricing derivatives and fixed-income instruments. Built with Python, this engine emphasizes institutional mathematical rigor, exact day-count conventions, and no-arbitrage pricing principles. Designed as a foundational analytics library, it completely decouples instrument representation from market data and pricing models.

## 🚀 Core Capabilities

### Derivatives Pricing & Analytics

- **Black-Scholes-Merton Engine**: Exact analytical pricing for European call and put options.
- **Complete Greeks**: First and second-order risk sensitivities including Delta, Gamma, Theta, Vega, and Rho, with robust handling for at-expiration boundary conditions.
- **Implied Volatility Solver**: Back-solves for implied volatility from market prices using Brent's method root-finding.

### Fixed Income & Yield Curves

- **Exact Cash Flow Modeling**: Accurate periodic compounding and day-count mechanics including Actual/Actual ICMA, Actual/365, and 30/360.
- **Pricing & Risk**: Computation of Clean/Dirty Prices, Yield to Maturity, Macaulay/Modified Duration, Convexity, and DV01.
- **Curve Bootstrapping**: Recursive solver to strip zero-coupon discount factors from a basket of market coupon bonds.
- **Curve Interpolation**: Log-linear interpolation on discount factors to guarantee piecewise-constant instantaneous forward rates.

## 🛠️ Technical Architecture

- **Modern Python**: Built using strict typing and `dataclasses` for immutable financial instrument modeling.
- **Mathematical Tooling**: Leverages `numpy` for vectorized logic, `scipy.stats` for distributions, and `scipy.optimize` for root-finding algorithms.
- **Dependency Management**: Managed via `uv` for lightning-fast, deterministic virtual environment resolution.

## 🧪 Testing Rigor

The engine is backed by a comprehensive `pytest` suite that validates outputs against fundamental mathematical laws:

- **Put-Call Parity**: Asserts that synthetic forward positions hold perfectly across the volatility surface.
- **Volatility Round-Tripping**: Validates that feeding solved implied volatility back into the Black-Scholes pricer perfectly recovers the initial market price.
- **Taylor Series Expansion**: Validates duration and convexity by proving that −D<sub>mod</sub>Δy + ½C(Δy)² matches actual discrete repricing bumps.
- **Bootstrapper Repricing**: Proves the solver's integrity by discounting benchmark bonds back through the calibrated curve to exactly recover their market dirty prices.

## 💻 Quickstart

### Prerequisites

Ensure you have [uv](https://github.com/astral-sh/uv) installed on your system.

### Installation & Testing

Clone the repository and run the test suite directly. `uv` will automatically handle environment creation and dependency syncing.

```bash
git clone https://github.com/YOUR_USERNAME/quant-engine.git
cd quant-engine
uv run pytest
```

## 📂 Project Structure

```
quant-engine/
├── src/
│   └── quant_engine/
│       ├── black_scholes.py      # Options pricing, Greeks, and implied volatility
│       ├── options.py            # Option dataclasses
│       ├── bonds.py              # Bond dataclass, day-counts, and cash flows
│       ├── bond_analytics.py     # YTM, pricing, duration, and convexity
│       ├── curves.py             # ZeroCurve state and interpolation
│       └── curve_bootstrapper.py # Recursive zero-curve solver
├── tests/                        # Pytest suite validating mathematical identities
├── pyproject.toml                # Project metadata and dependencies
└── uv.lock                       # Deterministic lockfile
```
