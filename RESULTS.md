# Results

## Historical prototype

The original saved notebook run is preserved as historical evidence.

| Quantity | Historical value |
|---|---:|
| Training examples | 5 |
| Epochs | 3 |
| Student parameters | 494,032,768 |
| Trainable parameters | 195,784,192 |
| Trainable percentage | 39.6% |
| Epoch 1 loss | 0.8547 |
| Epoch 2 loss | 0.8547 |
| Epoch 3 loss | 0.8225 |
| Distillation loss (reported) | ~0.0001 |
| NaN/Inf CE batches | observed and skipped |

## Interpretation

These measurements demonstrate execution of the prototype.

They do not establish that distillation improved coding ability.

The repository currently has no valid held-out comparison between:

- Base Qwen
- SFT-only Qwen
- KD Qwen

## Reproduction status

| Evidence | Status |
|---|---|
| Teacher generation pipeline | executed historically |
| Student training pipeline | executed historically |
| Loss metrics | measured historically |
| Capability improvement | not demonstrated |
| Held-out benchmark | missing |
| Multi-seed variance | missing |
