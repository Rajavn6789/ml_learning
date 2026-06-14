# ML Learning Notes

## Choosing the number of hidden neurons

Rule of thumb: more hidden neurons = more capacity to learn complex patterns,
but also more risk of overfitting and slower training. Start small and only grow
it if accuracy is poor. If you later plug in messier real data and 3 isn't enough,
just bump `n_hidden` back up and run `python student_nn.py train` again.

## Why does the loss keep getting so low?

On a tiny dataset (e.g. the 4 points in `ak_micrograd.py`), a model with far more
parameters than data points doesn't really "learn" — it **memorizes** the answers.
With nothing to stop it, MSE keeps shrinking toward zero almost indefinitely as the
predictions creep closer and closer to the exact targets.

What the numbers look like:

| steps  | loss          | what's happening                 |
| ------ | ------------- | -------------------------------- |
| 0–80   | 1.05 → 0.10   | real learning                    |
| 80–360 | 0.10 → 0.01   | refining                         |
| 360+   | 0.01 → 0.0008 | diminishing returns / memorizing |

## How much training is enough?

Don't pick a fixed step count — stop when the loss
stops improving. That's what the early-stopping logic in `ak_micrograd.py` does:
track the best loss, and if it doesn't improve by `min_delta` for `patience` steps,
break. For that toy problem ~200–300 steps already "solves" it; everything after is
squeezing decimals that don't matter.
