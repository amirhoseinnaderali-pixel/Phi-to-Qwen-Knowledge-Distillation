# Phi-4 → Qwen Knowledge Distillation

This repository studies whether reasoning-oriented knowledge from a larger coding model can be transferred to a much smaller student model.

## Research question

> Does teacher-derived soft-target supervision improve a small Qwen student's held-out coding performance over supervised training alone under the same training budget?

## Historical prototype

Teacher:

`microsoft/Phi-4-reasoning`

Student:

`Qwen/Qwen2.5-0.5B-Instruct`

The original notebook:
- generated teacher responses for 5 coding questions
- retained top-K teacher token probabilities
- trained the student for 3 epochs
- used CE + KL-style distillation loss
- used DeepSpeed ZeRO-2
- unfroze the last 4 transformer layers and LM head

The stored run reports:

- student parameters: 494,032,768
- trainable parameters: 195,784,192 (39.6%)
- epoch losses: 0.8547 → 0.8547 → 0.8225
- distillation loss: approximately 0.0001
- multiple CE NaN/Inf batches were skipped

These are historical prototype measurements.

## What is and is not established

The prototype demonstrates that the pipeline can execute.

It does **not** demonstrate that knowledge distillation improved coding capability.

The original repository has no valid held-out comparison among:
- Base Qwen
- SFT-only Qwen
- KD-trained Qwen

The README therefore does not claim a quality-retention percentage, inference speedup, or memory reduction as a measured result.

## Controlled study

The intended comparison is:

`Base → SFT → SFT + KD`

with the same student model, data split, training budget, and evaluation protocol.

Primary evaluation:

- HumanEval+
- MBPP

Metrics:

- pass@1 / execution success
- syntax/compile success
- inference latency
- peak memory
- trainable parameters

## Scientific focus

The broader research trajectory is:

`Efficient adaptation → DPO → Distillation → Reasoning → Test-time compute`

This project addresses the compression side:

> Can useful reasoning behavior be moved into a much smaller model so that the expensive teacher is not required at inference time?

## Repository status

| Stage | Status |
|---|---|
| Prototype implementation | complete |
| Historical execution | documented |
| Research audit | complete |
| Controlled Base/SFT/KD comparison | pending |
| Held-out coding benchmark | pending |
| Multi-seed evaluation | pending |
| Efficiency analysis | pending |

## Documentation

- `RESEARCH_AUDIT.md`
- `RESULTS.md`
- `REPRODUCIBILITY.md`
- `configs/research.yaml`
- `results/historical/prototype_run.json`

## Safety and scope

This is a coding-model research experiment. It does not establish general reasoning ability, general model compression performance, or deployment quality outside the measured tasks.
