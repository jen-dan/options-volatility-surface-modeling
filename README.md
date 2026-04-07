# Arbitrage-Free Volatility Surface Prototype

A Python project for parsing Bloomberg option chain exports, cleaning listed options data, and fitting SVI slices across expiries for SPY options.

## Current features
- Bloomberg CSV parsing
- Call/put data cleaning
- Quote filtering
- Volatility smiles
- ATM term structure
- Call vs put comparison
- IV heatmap
- Raw SVI calibration across expiries

## Project structure
- `src/` - data cleaning, plotting, and SVI fitting scripts
- `data/raw/` - raw Bloomberg exports (not tracked)
- `data/processed/` - cleaned datasets (not tracked)
- `outputs/` - generated charts and fit outputs (not tracked)

## Notes
This project uses sparse Bloomberg snapshot data, so some expiries have limited strike coverage. SVI calibration is implemented as a prototype and evaluated slice by slice.

## Next steps
- validation report
- stronger no-arbitrage checks
- improved surface interpolation
- buy-side extension