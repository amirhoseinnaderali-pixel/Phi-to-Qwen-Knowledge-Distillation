# Research Audit — Phi-4 → Qwen Knowledge Distillation

## Scope

This audit is based on the repository README and the saved Kaggle notebook output in the original commit. No new training run is claimed.

## Research question

The project asks whether reasoning-oriented knowledge from a larger teacher can be transferred to a much smaller student while reducing model size and inference cost.

A defensible experimental question is:

> Does adding teacher-derived soft-target supervision improve a small Qwen student's coding performance over supervised training alone, under the same training budget?

This formulation is stronger than simply demonstrating that a distillation pipeline runs.

## Models

- Teacher: `microsoft/Phi-4-reasoning`
- Student: `Qwen/Qwen2.5-0.5B-Instruct`

The stated teacher/student parameter ratio is approximately 28×.

## Original training design

The saved notebook uses:

- 5 coding questions
- 3 epochs
- batch size 1
- gradient accumulation 8
- learning rate 5e-6
- CE weight alpha = 0.7
- distillation weight beta = 0.3
- distillation temperature = 1.5 during student training
- FP16
- DeepSpeed ZeRO-2
- last 4 transformer layers + LM head unfrozen

## Historical execution evidence

The notebook output records:

- 5 training samples
- 3 epochs
- trainable parameters: 195,784,192 / 494,032,768 = 39.6%
- epoch 1 average loss: 0.8547
- epoch 2 average loss: 0.8547
- epoch 3 average loss: 0.8225
- reported distillation loss: approximately 0.0001
- multiple CE NaN/Inf batches were skipped during training

Therefore the original run is evidence of a functioning prototype, but not evidence of successful capability transfer.

## Important inconsistencies

### A1 — The README's ~7% trainable-parameter claim is contradicted

The actual saved training output reports:

> 195,784,192 / 494,032,768 = 39.6%

The README's "~7%" figure therefore must not be presented as measured for this run.

### A2 — Only five training examples

Five hand-written coding questions cannot support a meaningful capability claim.

This is a feasibility demonstration, not a benchmark.

### A3 — No baseline

There is no stored comparison against:
- the original Qwen model
- Qwen fine-tuned with ordinary supervised learning
- a student trained without distillation

Without those baselines, the effect of distillation cannot be isolated.

### A4 — No held-out benchmark

The same tiny set is used as the training material. There is no valid held-out coding evaluation.

### A5 — NaN batches were skipped

The training log explicitly records CE NaN/Inf batches being skipped. This means the effective training data seen by the optimizer is not identical to the nominal dataset.

This must be surfaced rather than hidden.

### A6 — Distillation loss carries almost no observed magnitude

The recorded distillation loss is approximately 0.0001 throughout the visible run.

This does not establish that teacher information meaningfully influenced optimization.

### A7 — README claims expected performance as if measured

The README lists approximate inference speed, memory reduction, and quality retention. Those values are described as typical/expected and are not demonstrated by the saved run.

They should remain clearly labelled as unmeasured expectations or be removed.

### A8 — Teacher generation and student objective need stronger validation

The teacher stage stores only top-K logits. The student stage reconstructs a sparse teacher distribution and computes a KL term.

This is a reasonable prototype, but the implementation needs ablations and sanity checks before the KL term can be interpreted as effective knowledge transfer.

### A9 — Stage-1 configuration and actual execution differ in documentation

The README presents a teacher temperature of 2.0, while student training uses 1.5. These are different roles and should be explicitly separated in the configuration.

### A10 — Configuration flags are partly aspirational

The teacher config lists hidden-state and attention storage as enabled, while the generation call actually disables them. Documentation should describe what is actually stored.

## Status ladder

- Implemented: yes
- Executed: yes
- Measured: training loss / CE loss / distillation loss / trainable parameter count
- Demonstrated capability transfer: no

## Research redesign

The controlled study should compare:

1. Base student
2. SFT-only student
3. SFT + KD student

All three should use the same student, data split, optimizer budget, and evaluation benchmark.

The primary outcome should be coding task performance, not training loss.

Recommended held-out benchmarks:
- HumanEval+
- MBPP
- optionally a small manually audited reasoning/coding set

Metrics should include:
- pass@1
- execution success rate
- syntax/compile success
- inference latency
- peak memory
- student parameter count

## High-value ablations

1. CE only vs CE + KD
2. KD weight beta
3. distillation temperature
4. top-K size
5. full-student fine-tuning vs selective last-four-layer training

Do not run all ablations before establishing one reliable baseline.

## Main scientific hypothesis

> Under a fixed student training budget, teacher-derived soft targets will improve held-out coding performance relative to supervised training alone.

This hypothesis is falsifiable.

## Next research question

If KD improves coding performance, the next question becomes:

> Can teacher reasoning be compressed into a small student without requiring the teacher at inference time, and how does this compare with spending the same compute on test-time reasoning?
