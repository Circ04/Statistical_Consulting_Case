## Slide 4

"If you regress bookings on price with the obvious controls, you get β equals plus 0.29. Read literally: raising prices increases bookings. That's impossible.

Why? Endogeneity. ECG's pricer already reacts to demand — fills up, price goes up; slow week, discount. So in the data, high prices and high bookings appear together — not because price causes bookings, but because both respond to the same underlying demand signal.

Here's the picture (point to the DAG). Latent demand drives bookings directly — that's what we want to measure. But demand also drives bookings-on-books, which the pricer reads to set price. That's the backdoor.

We close it by conditioning on bookings-on-books. Result: β flips to minus 0.76. Negative as theory requires, and inelastic — meaning revenue rises with price within our data range."

## Slide 5

Now we have a defensible elasticity, but we need two different things from two different models.

Bayesian on the left. Its job: estimate the causal elasticity. Minimal features, only what's needed for identification. It gives us the minus 0.76 we just saw, with credible intervals. It's not a forecaster — it doesn't know about lead time or calendar.

LightGBM on the right. Its job: predict bookings at any candidate price, for any single snapshot of the booking horizon. Adds calendar, lead time, accommodation features that the Bayesian deliberately omits.

Here's the cross-check that matters: LightGBM's local price slope is minus 0.75. Bayesian said minus 0.76. Two completely different methods, same answer.

Out-of-sample error: 27% per ROMGID — within industry norms. And because LightGBM learns a flexible curve, it can find revenue maxima inside the safe range, which is where Andrew picks up."