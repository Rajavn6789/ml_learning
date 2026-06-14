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

## How many hidden layers are needed?

There's no formula — it's empirical. Pick the smallest network that fits the data
well, then grow only if it underfits.

Rough guide by problem:

| Problem                         | Typical depth                      |
| ------------------------------- | ---------------------------------- |
| Linearly separable / simple     | 0 hidden layers (logistic is fine) |
| Most tabular & toy problems     | 1–2 hidden layers                  |
| Complex, moderate non-linearity | 2–3 hidden layers                  |
| Images, audio, language         | Many — use a proven architecture   |

**Method:** start with 1 hidden layer, then watch two error signals.

| Symptom                   | Meaning      | Action                                 |
| ------------------------- | ------------ | -------------------------------------- |
| High train error          | Underfitting | Add neurons, then layers               |
| Low train, high val error | Overfitting  | Fewer layers; add dropout/weight decay |
| Both low                  | Just right   | Stop                                   |

Prefer fewer layers with small datasets (overfit fast) and because returns
diminish quickly (1→2 helps far more than 3→4).

## How many data points are needed?

Rule of thumb: **~10× more examples than the model has parameters.** Far fewer than
that and the network memorizes instead of learning.

To actually _see_ overfitting you also need a real train/val split — aim for a few
hundred points (~300–500), split 80/20.

## How to count parameters

A `Linear(in, out)` layer has `in × out` weights plus `out` biases, so:

```
params = (in × out) + out  =  out × (in + 1)
```

Activations (Tanh, ReLU, …) have **no** parameters. For the `ak_micrograd.py`
model (2 → 4 → 4 → 1):

| Layer        | Weights (in × out) | Biases (out) | Subtotal |
| ------------ | ------------------ | ------------ | -------- |
| Linear(2, 4) | 2 × 4 = 8          | 4            | 12       |
| Tanh         | 0                  | 0            | 0        |
| Linear(4, 4) | 4 × 4 = 16         | 4            | 20       |
| Tanh         | 0                  | 0            | 0        |
| Linear(4, 1) | 4 × 1 = 4          | 1            | 5        |
| **Total**    |                    |              | **37**   |

Quick check in code:

```python
sum(p.numel() for p in model.parameters())   # -> 37
```

## How to calculate model size (bytes)

Parameter **count** tells you how many numbers there are; the **dtype** tells you
how many bytes each one takes. Model size = `params × bytes_per_param`.

| dtype              | Bytes | Notes                              |
| ------------------ | ----- | ---------------------------------- |
| float64 (double)   | 8     | Plain Python `float` (micrograd)   |
| float32 (single)   | 4     | PyTorch/TensorFlow default         |
| float16 / bfloat16 | 2     | Half precision                     |
| int8 / float8      | 1     | Quantized models                   |

The `ak_micrograd.py` model uses `dtype=torch.float32`, so 37 × 4 = **148 bytes**.

```python
# total bytes of all parameters
sum(p.numel() * p.element_size() for p in model.parameters())   # -> 148

# bytes per parameter
next(model.parameters()).element_size()   # -> 4  (float32)
```

`p.element_size()` returns bytes-per-element for the tensor's dtype, so
`numel() × element_size()` gives exact storage. Real-world models scale the same
way: e.g. a 7B-param model at float16 = 7e9 × 2 ≈ 14 GB.
