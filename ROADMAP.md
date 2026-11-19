# Forecast Improvement Roadmap

Where the sales forecast is heading, at a high level. The detailed working plan is kept separately by the team.

## How changes are decided

Every change is tested the same way before it reaches production:
- It is replayed on two years of past months, as if it had been running at the time, and compared with the current
  model on exactly the same months.
- It is adopted only if it is clearly better, and it must not make any part of the forecast noticeably worse (any
  horizon, small or large products, unusual market periods).
- Results that could be down to chance are rerun before any decision.

## Current focus

**1. Steadier forecasts from month to month**
Today each forecast horizon (1 to 6 months ahead) has its own model, so the forecast for a given month can jump when it
moves from one horizon to the next. We are testing a single shared model across horizons so the forecast changes more
smoothly as the month approaches.

**2. Faster and more reliable testing**
Tooling that makes each experiment cheaper and easier to trust:
- report whether a change tends to push forecasts up or down overall, not only how far off they are;
- record every experiment in the experiment-tracking system;
- train on a smaller, representative sample when screening ideas;
- make results identical across machines, so several experiments can run in parallel.

## Next

**3. Combining the model with existing planning methods**
Test whether blending the model with the current planning method, and with the sales team's own forecast, gives a more
stable result than the model alone.

**4. Small and fast-growing products**
Small products tend to be over-forecast, and fast-growing ones under-forecast. Planned tests: limits based on each
product's own history, a rule-based way to spot growing products, and a model that learns from relative changes rather
than raw quantities.

**5. Model and settings**
Compare other model types, review the settings that make the forecast deliberately cautious for some product groups,
give more weight to recent history, and remove inputs that add nothing.

## Later

**6. Prices and market events**
Bring price data into the forecast, and handle unusual market periods (for example price changes and sudden demand
surges) better. This includes telling a temporary spike from lasting growth, and detecting such periods from the data
rather than from manually set dates.

**7. More signals from existing data**
Look for indicators of customer stock levels and buying patterns in the order data we already have.

**8. Fine-tuning (optional)**
A systematic search over model settings, only if time allows after the structural changes above.

## For planners

**Typical range next to each forecast**
A short note per product, for example "in the past 12 months, the actual usually landed between 15% below and 3% above
this forecast", so planners can judge how far to rely on each number. To be agreed with the business before it is
built.
