
# Safer Scenario Simulator

A scenario simulator for the **Safer** (Simple Agreement for Future Equity with Repurchase) financial instrument. Generates quarterly cash flow projections, liquidity event analysis, and interactive HTML reports with charts.

This simulator is a companion to [*Venture Capital Without Unicorns*](https://www.amazon.com/dp/B0GR9D7S2W) by James Thomason. For portfolio-level Monte Carlo analysis, see the [Safer Monte Carlo Simulator](https://github.com/next-wave-partners/safer-monte-carlo).

## What It Does

The simulator models a single Safer investment through its full lifecycle: honeymoon period, quarterly revenue share payments, Safer Amount evolution, and resolution at a liquidity event (acquisition, IPO, or direct listing). It produces terminal output with a complete cash flow table, return waterfall, and Safer Amount formula breakdown, plus an HTML report with embedded charts.

## Quick Start

```bash
# Run the default scenario (Pre-Seed SaaS, steady growth to $120M acquisition)
python safer.py

# Run a different built-in scenario
python safer.py --scenario explosive-growth-ipo

# Customize the Safer terms
python safer.py \
    --investment 750000 \
    --valuation-cap 5000000 \
    --target-return-multiple 2.0 \
    --revenue-share 0.05 \
    --scenario steady-growth-acquisition

# Fully custom scenario
python safer.py \
    --investment 500000 \
    --valuation-cap 5000000 \
    --target-return-multiple 3.0 \
    --revenue-share 0.05 \
    --repurchase-percent 0.90 \
    --honeymoon-months 12 \
    --initial-arr 200000 \
    --growth-rates 0,0.8,0.6,0.5,0.4 \
    --exit-year 5 \
    --exit-valuation 50000000 \
    --output my_scenario.html
```

## Built-In Scenarios

| Scenario | Description |
|----------|-------------|
| `steady-growth-acquisition` | Pre-seed SaaS, steady growth to $120M acquisition in Year 7 |
| `explosive-growth-ipo` | AI startup, explosive growth to $2B IPO in Year 7 |
| `sustainable-complete-repurchase` | Seed SaaS, sustainable growth, full 3x target repaid through operations |
| `failure-low-revenue` | Company fails to achieve significant revenue, eventual shutdown |
| `modest-growth-small-exit` | Modest growth, small $25M acquisition in Year 4 |
| `custom` | Define your own revenue profile, growth rates, and exit parameters |

## Safer Terms (CLI Arguments)

| Argument | Default | Description |
|----------|---------|-------------|
| `--investment`, `-i` | $1,000,000 | Purchase Amount (the investment) |
| `--valuation-cap`, `-v` | $10,000,000 | Post-Money Valuation Cap for conversion |
| `--target-return-multiple`, `-t` | 3.0x | Target Return as multiple of investment |
| `--revenue-share`, `-r` | 5% | Revenue Percentage paid quarterly |
| `--repurchase-percent`, `-p` | 90% | Percentage of Purchase Amount bought back at target |
| `--honeymoon-months`, `-m` | 12 | Months before revenue share payments begin |

## Custom Scenario Parameters

| Argument | Description |
|----------|-------------|
| `--initial-arr` | Starting annual recurring revenue |
| `--growth-rates` | Comma-separated annual growth rates (e.g., `0,0.8,0.6,0.5`) |
| `--exit-year` | Year of liquidity event |
| `--exit-valuation` | Exit valuation in dollars |
| `--description` | Scenario description for the report |

## Output

| Argument | Default | Description |
|----------|---------|-------------|
| `--output`, `-o` | `safer_scenario_report.html` | HTML report output path |
| `--json` | off | Also output results as JSON |

The HTML report includes three embedded charts (requires `matplotlib` and `numpy`):

1. **Revenue and Investor Cash Flow Timeline** showing ARR growth alongside quarterly repurchase payments and cumulative returns (log scale)
2. **Safer Amount Evolution** tracking the investor's equity claim as it declines with each payment
3. **Investment Return Waterfall** breaking down the total return into repurchase payments and liquidity event payout

## How the Safer Works

The simulator implements the Safer contract mechanics exactly as defined in the legal agreement.

**Safer Amount Formula:**

```
Safer Amount = Purchase Amount
             - [(Cumulative Payments / Target Return) × Repurchase Amount]
             + [Target Return - Cumulative Payments]
```

At a liquidity event, the investor receives the **greater of**:

- **Cash-Out Amount:** The Safer Amount (downside protection)
- **Conversion Amount:** The Safer Amount divided by the Valuation Cap, multiplied by the exit valuation (equity upside)

This creates a fork: in strong exits, the investor converts and participates like an equity holder with uncapped upside. In weak exits, the investor takes the Cash-Out Amount, which includes any unpaid Target Return as a liquidation preference.

As the company makes quarterly revenue share payments, the Safer Amount decreases. This is automatic cap table repair: the investor's equity claim shrinks as the company buys it back through operations.

## Requirements

Python 3.8+. No required dependencies beyond the standard library.

Optional (for charts in HTML reports):
```bash
pip install matplotlib numpy
```

## License

BSD 4-Clause License. Copyright (c) 2025 Next Wave Publishing LLC. See [LICENSE](LICENSE) for full terms.

The Safer contract, term sheets, and accounting guidance are available at [nextwave.partners/safer](https://nextwave.partners/safer).
