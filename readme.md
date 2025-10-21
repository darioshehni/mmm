
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

## What the tool does

1. **Learn mix from recent sales**
   Compute recent **model shares** (Base/Edge/Ultra) and **SKU shares** within each model from a short time window. Derive a **category elasticity** as a share-weighted average of model elasticities.

2. **Predict demand & cannibalization with simple choice rules**

   * A **representative model price** is the baseline-share-weighted average of that model’s SKU prices.
   * **Model shares** respond to **relative model prices** (softmax): a cheaper model gains share from others.
   * **SKU shares** within a model respond to **relative SKU prices** (softmax): a cheaper SKU gains share from siblings.
   * **Category average price** = Σ (predicted model share × representative model price).
   * **Demand level** ∝ (category average price)^(category elasticity).
     Cannibalization is automatic because shares sum to 1 at both levels (between models and within a model). This is the standard discrete-choice framing for multi-product pricing. ([Econometrics Laboratory][1])

3. **Choose prices to maximize profit**
   Profit = Σ (price − cost) × demand level × model share × SKU share.
   We search the price grid with **coordinate descent** (step = tick) and keep changes that improve profit. Business rules always apply:

   * Not below **minimum price** per SKU
   * Not beyond **±% band vs last observed price**
   * Prices must be **multiples of 5**

## Algorithms

* **Discrete-choice (logit / nested-logit style) sharing** is used to capture **substitution** and **cannibalization** among differentiated products; it is widely used in multi-product pricing optimization. ([Econometrics Laboratory][1])
* **Joint price optimization** under an (nested) MNL demand system is a core topic in **revenue management and pricing**. Our implementation is: two softmax layers + a category demand curve. ([Google Books][2])
* The **category-level elasticity** on **average price** is a standard simplification when you want portfolio moves without estimating full outside-option models; it mirrors choice-model practice while staying lightweight for an MVP. ([Stanford University Press][3])

## References

* Train, **Discrete Choice Methods with Simulation** — classic text on logit/nested-logit share models. ([Econometrics Laboratory][1])
* Talluri & van Ryzin, **The Theory and Practice of Revenue Management** — foundational for multi-product pricing and RM. ([Google Books][2])
* Recent assortment/price optimization under MNL (illustrative). ([INFORMS Pubsonline][4])

---

[1]: https://eml.berkeley.edu/books/train1201.pdf?utm_source=chatgpt.com "Discrete Choice Methods with Simulation"
[2]: https://books.google.com/books/about/The_Theory_and_Practice_of_Revenue_Manag.html?id=u7hcyBraOCwC&utm_source=chatgpt.com "The Theory and Practice of Revenue Management"
[3]: https://www.sup.org/books/business/pricing-and-revenue-optimization/excerpt/introduction?utm_source=chatgpt.com "Pricing and Revenue Optimization: Introduction"
[4]: https://pubsonline.informs.org/doi/10.1287/opre.2021.2127?utm_source=chatgpt.com "Assortment Optimization and Pricing Under the Multinomial ..."
