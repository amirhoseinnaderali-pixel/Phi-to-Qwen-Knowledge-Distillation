# Reproducibility

## Historical environment

The notebook was executed in a Kaggle GPU environment and reports Python 3.12.12 in notebook metadata.

The historical environment is not fully locked in the repository.

## Historical models

Teacher:
`microsoft/Phi-4-reasoning`

Student:
`Qwen/Qwen2.5-0.5B-Instruct`

## Historical training configuration

- batch size: 1
- gradient accumulation: 8
- epochs: 3
- learning rate: 5e-6
- alpha: 0.7
- beta: 0.3
- student temperature: 1.5
- FP16
- DeepSpeed ZeRO-2
- max sequence length: 512

## Required controlled-run metadata

Every new run should record:

- model revisions
- dataset revision
- random seed
- Python
- PyTorch
- Transformers
- DeepSpeed
- CUDA
- GPU model/count
- git SHA
- configuration hash
- wall-clock time
- peak memory

## Important

Historical notebook outputs must remain separate from newly reproduced measurements.
