
# Pricing MVP

Joint price recommendations for Samsung S-Series (Base / Edge / Ultra and their SKUs). Maximizes **profit** portfolio-wide while accounting for **cannibalization** between models and within each model.

## Inputs

* **transactions.csv** — one row per sale
  Columns: `SKU_id, model_name, cost_unit, price_sold, promo_flag, discount_value, date_time`
  (We use `net_price = price_sold - discount_value`.)

* **elasticities.csv** — one row per model
  Columns: `model_name, elasticity` (must be negative)

* **min_price.csv** — one row per SKU
  Columns: `SKU_id, min_price`


## Steps

1. Propose SKU prices.
2. Compresses each model’s SKUs into one representative model price.
3. Uses those rep. model prices to predict how buyers reallocate across models (cheaper model gets more share) (Softmax).
4. Those predicted model shares then set the category’s average price, which drives total demand via the category elasticity (Isoelastic demand).
5. With the total pie set, we split each model’s units across its SKUs based on relative SKU prices (Softmax).
6. We score the candidate prices by profit across the whole portfolio.
7. We start the search from last observed prices and make them valid.
8. From that valid start, we nudge one SKU at a time by +€5 or −€5 and re-evaluate Steps 2–6.
9. We repeat these passes across all SKUs until nothing improves (or we hit the pass limit).
10. We return the final price set.


## Algorithms

* **Discrete-choice (logit / nested-logit style) sharing** is used to capture **substitution** and **cannibalization** among differentiated products; it is widely used in multi-product pricing optimization. ([Econometrics Laboratory][1])
* **Joint price optimization** under an (nested) MNL demand system is a core topic in **revenue management and pricing**. Our implementation is: two softmax layers + a category demand curve. ([Google Books][2])
* The **category-level elasticity** on **average price** is a standard simplification when you want portfolio moves without estimating full outside-option models; it mirrors choice-model practice while staying lightweight for an MVP. ([Stanford University Press][3])


## Files

* `runner.py` — Orchestrates the end-to-end run. Loads inputs and config, builds baselines, constructs the profit objective, runs the optimizer, applies final validity checks, and writes the recommended prices.
* `data_io.py` — Reads and validates the CSVs.
* `baselines.py` — Derives average values from a recent window: model shares, SKU shares within each model, average SKU prices (fallback), last observed price per SKU, and a single category elasticity (share-weighted average of model elasticities).
* `demand_shares.py` — Core behavioral pieces used during evaluation: representative (typical) model price, model-level share rule (between models), SKU-level share rule (within a model), and the category demand multiplier from average price and elasticity.
* `constraints.py` — Makes any candidate price vector valid: never below the SKU’s minimum price, stay within the allowed ±% band around the last price (minimum price takes precedence if they conflict), and snap to the price tick (e.g., 5).
* `optimize.py` — Builds the portfolio profit function (margin × demand level × model share × SKU share) and searches with coordinate descent over the price grid. Tries ± one tick per SKU, keeps improvements, and stops when a pass finds no gains.

## Limitations

* No absolute sales forecasting. This means we can’t report forecasted units/revenue—only pick prices.
* Average-price shortcut. Total S-series demand depends on the category average price with one category elasticity. This is a simplification of an outside-option model and can miss nuances (e.g. promos that increase sales more than average price implies).
* One sensitivity per layer. A single k_model (between models) and k_sku (within model) control switching. Real behavior may differ by model/SKU; calibration is needed for precision.
* Baseline anchoring. Representative model price uses recent SKU shares. If the window is unrepresentative (low volume, heavy promos), the representative price can be biased.
* Local search. Coordinate descent with step=5 finds a good local optimum, not a guaranteed global optimum. Multi-starts or broader search would improve robustness.
* No inventory or capacity. The optimizer doesn’t consider stock limits, lead times, or channel capacity.
* No competitor or cross-brand effects. Rival prices, availability, and marketing are not modeled.
* Static, price-only. No explicit seasonality, day-of-week, or non-price drivers (ads, store traffic). Elasticity is assumed locally valid; large moves may drift outside its range.
* Promo handling is minimal. We include promo sales in baselines by default; deep discounting can distort recent mix unless you toggle that off or weight down promos.
* Rule precedence. If the minimum price conflicts with the ±% move band, minimum price wins, which can override stabilization intent for some SKUs.
* Tick constraint. Prices must be multiples of 5; this can block small fine-tuning moves near optima.
* Data quality assumptions. We assume costs are current and timestamps/prices are clean; missing or lagged costs will bias the margin trade-off.

## References

* Train, **Discrete Choice Methods with Simulation** — classic text on logit/nested-logit share models. ([Econometrics Laboratory][1])
* Talluri & van Ryzin, **The Theory and Practice of Revenue Management** — foundational for multi-product pricing and RM. ([Google Books][2])
* Recent assortment/price optimization under MNL (illustrative). ([INFORMS Pubsonline][4])
* Gallego & Wang, **Optimal Pricing with Two-Level Nested Logit Models** — similar nested-logit pricing model. ([Columbia PDF][5])

---

[1]: https://eml.berkeley.edu/books/train1201.pdf?utm_source=chatgpt.com "Discrete Choice Methods with Simulation"
[2]: https://books.google.com/books/about/The_Theory_and_Practice_of_Revenue_Manag.html?id=u7hcyBraOCwC&utm_source=chatgpt.com "The Theory and Practice of Revenue Management"
[3]: https://www.sup.org/books/business/pricing-and-revenue-optimization/excerpt/introduction?utm_source=chatgpt.com "Pricing and Revenue Optimization: Introduction"
[4]: https://pubsonline.informs.org/doi/10.1287/opre.2021.2127?utm_source=chatgpt.com "Assortment Optimization and Pricing Under the Multinomial ..."
[5]: https://www.columbia.edu/~gmg2/nestedpricing.pdf "Optimal Pricing with Two-Level Nested Logit Models"
