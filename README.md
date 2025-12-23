# 🎓 Phi-to-Qwen Knowledge Distillation

A memory-optimized knowledge distillation pipeline that transfers reasoning capabilities from Microsoft's Phi-4 (14B teacher) to Qwen 2.5 (0.5B student) for coding tasks.

## 📋 Overview

This project implements a two-stage knowledge distillation process:

1. **Stage 1: Data Generation** - Extract soft labels and reasoning patterns from a large teacher model
2. **Stage 2: Student Training** - Train a smaller student model using both the teacher's knowledge and traditional supervised learning

**Simple Explanation:** Think of it like a master teacher (Phi-4) teaching a quick learner (Qwen). Instead of just copying answers, the student learns *how* the teacher thinks by studying the teacher's confidence levels and reasoning process for each answer.

## 🎯 Project Goals

- **Compress knowledge** from a 14B parameter model into a 0.5B model
- **Maintain reasoning quality** while drastically reducing computational costs
- **Optimize for coding tasks** with minimal memory footprint
- **Enable deployment** on resource-constrained devices

## 🏗️ Architecture

### Stage 1: Teacher Knowledge Extraction

```
Input: Coding Questions
         ↓
    Phi-4 Model (Teacher)
         ↓
    Generate Responses
         ↓
    Extract:
    • Token predictions
    • Probability distributions (soft labels)
    • Top-K most likely tokens per step
         ↓
    Save to JSON Dataset
```

**Key Features:**
- Aggressive GPU memory management
- Streaming generation (batch_size=1)
- Selective data retention (top-50 logits only)
- Checkpoint saving every 2 questions
- Automatic cache clearing after each sample

### Stage 2: Student Model Training

```
Input: Distillation Dataset
         ↓
    Qwen 2.5 Model (Student)
         ↓
    Combined Loss Training:
    • α × Cross-Entropy Loss (hard labels)
    • β × KL Divergence Loss (soft labels from teacher)
         ↓
    Selective Fine-Tuning (last 4 layers only)
         ↓
    Trained Student Model
```

**Key Features:**
- DeepSpeed ZeRO-2 optimization
- Selective layer unfreezing (only last 4 layers + LM head)
- Anti-NaN protection in distillation loss
- Gradient clipping and accumulation
- Mixed precision training (FP16)

## 📊 Technical Details

### Memory Optimization Techniques

1. **Aggressive CPU Offloading** - Move tensors to CPU immediately after processing
2. **Explicit Cache Clearing** - `torch.cuda.empty_cache()` at critical points
3. **Gradient Checkpointing** - Trade computation for memory
4. **Selective Data Retention** - Store only top-K logits instead of full distributions
5. **Streaming Generation** - Process one sample at a time
6. **Reduced Token Length** - Max 512 tokens per generation (instead of 1024)
7. **Disabled Heavy Outputs** - Skip hidden states and attention weights during generation

### Loss Function

The student is trained with a weighted combination:

```
Total Loss = α × CE_Loss + β × Distillation_Loss
```

- **CE_Loss**: Standard cross-entropy with ground truth labels
- **Distillation_Loss**: KL divergence between student and teacher probability distributions
- **Default weights**: α=0.7, β=0.3
- **Temperature**: 1.5 (controls softness of probability distributions)

### Dataset Format

Each training sample contains:

```json
{
  "instruction": "Write a Python function to check if a string is a palindrome.",
  "output": "def is_palindrome(s): return s == s[::-1]",
  "soft_labels": [
    {
      "step": 0,
      "top_k_indices": [123, 456, 789],
      "top_k_logits": [5.2, 4.8, 4.1],
      "top_k_probs": [0.45, 0.32, 0.23]
    }
  ],
  "token_ids": [123, 456, 789],
  "category": "string_manipulation"
}
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch>=2.0.0
pip install transformers>=4.35.0
pip install deepspeed>=0.12.0
pip install accelerate>=0.25.0
pip install tqdm
```

### Stage 1: Generate Distillation Dataset

```bash
python knowledge_distillation_stage1.py
```

**Output Files:**
- `complete_distillation_dataset.json` - Full dataset with metadata
- `student_distillation_data.json` - Lightweight format for training
- `temp_dataset_N.json` - Checkpoint files (auto-saved every 2 questions)

**Expected Runtime:** ~10-15 minutes for 5 questions on a single GPU

### Stage 2: Train Student Model

```bash
deepspeed --num_gpus=1 knowledge_distillation_stage2.py
```

**Output Files:**
- `selective_distillation_qwen/checkpoint-N/` - Periodic checkpoints
- `selective_distillation_qwen/best/` - Best performing model
- `selective_distillation_qwen/final/` - Final trained model

**Expected Runtime:** ~30-45 minutes for 3 epochs on a single GPU

## ⚙️ Configuration

### Teacher Model Settings

```python
DISTILLATION_CONFIG = {
    "temperature": 2.0,           # Softness of probability distributions
    "top_k_logits": 50,           # Number of top predictions to save
    "max_new_tokens": 512,        # Maximum generation length
    "generation_batch_size": 1,   # Process one at a time
    "offload_to_cpu": True,       # Immediate CPU transfer
    "clear_cache_per_step": True  # Clean GPU after each step
}
```

### Student Training Settings

```python
CONFIG = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "batch_size": 1,              # Effective batch size
    "grad_accum": 8,              # Gradient accumulation steps
    "epochs": 3,                  # Training epochs
    "lr": 5e-6,                   # Learning rate
    "alpha": 0.7,                 # Weight for CE loss
    "beta": 0.3,                  # Weight for distillation loss
    "temperature": 1.5,           # Distillation temperature
    "max_grad_norm": 1.0          # Gradient clipping threshold
}
```

## 📈 Expected Results

### Model Compression

- **Teacher Size**: 14B parameters
- **Student Size**: 0.5B parameters
- **Compression Ratio**: 28× smaller
- **Trainable Parameters**: ~7% of student model (last 4 layers only)

### Performance Metrics

On coding tasks:
- **Inference Speed**: ~15-20× faster than teacher
- **Memory Usage**: ~30× less GPU memory
- **Quality Retention**: ~80-85% of teacher performance (typical for distillation)

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce `max_new_tokens` to 256 or lower
- Ensure `batch_size=1` in both stages
- Enable CPU offloading in DeepSpeed config

**2. NaN Losses During Training**
- The code includes anti-NaN protection
- If persistent, reduce learning rate to 1e-6
- Increase gradient clipping (`max_grad_norm=0.5`)

**3. Slow Generation**
- Use a smaller teacher model if available
- Reduce `top_k_logits` to 20-30
- Disable attention/hidden state extraction (already done)

**4. Checkpoint Not Saving**
- Check disk space availability
- Verify write permissions in output directory
- Reduce checkpoint frequency if needed

## 📁 Project Structure

```
.
├── knowledge_distillation_stage1.py    # Teacher data generation
├── knowledge_distillation_stage2.py    # Student training
├── complete_distillation_dataset.json  # Full dataset (generated)
├── student_distillation_data.json      # Training dataset (generated)
├── ds_config.json                      # DeepSpeed config (auto-generated)
└── selective_distillation_qwen/        # Model checkpoints (generated)
    ├── checkpoint-50/
    ├── checkpoint-100/
    ├── best/
    └── final/
```

## 🎓 How It Works

### Knowledge Distillation Explained

Traditional training: Student learns from hard labels (correct answer = 1, others = 0)

```
Question: Is this a palindrome?
Hard Label: [0, 0, 1, 0]  ← Only one correct answer
```

Knowledge distillation: Student learns from teacher's probability distribution

```
Question: Is this a palindrome?
Teacher Soft Labels: [0.05, 0.12, 0.75, 0.08]  ← Shows confidence levels
                              ↑
                    Teacher is 75% confident, but also considers alternatives
```

**Why it works:** The teacher's uncertainty and alternative predictions contain valuable information about the problem structure and edge cases.

### Selective Fine-Tuning

Instead of training all 500M parameters, we only train:
- **Last 4 transformer layers** (~35M parameters)
- **Language modeling head** (~5M parameters)
- **Total trainable**: ~40M parameters (7% of model)

**Benefits:**
- Much faster training
- Prevents catastrophic forgetting of pre-trained knowledge
- Focuses learning on task-specific adaptations

## 📚 References

- [Knowledge Distillation (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

## ⚡ Quick Start Example

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate training data (10-15 min)
python knowledge_distillation_stage1.py

# 3. Train student model (30-45 min)
deepspeed --num_gpus=1 knowledge_distillation_stage2.py

# 4. Use your trained model
python inference.py --model ./selective_distillation_qwen/best
```

---

**Note:** This implementation prioritizes memory efficiency over speed. For production use, consider adding multi-GPU support and optimizing the data pipeline further.