# Qwen3 0.6B Sentiment Analysis Full-Parameter Fine-Tuning

This project demonstrates how to perform full-parameter fine-tuning on the Qwen3 0.6B model for a Chinese sentiment analysis task using MLX-LM.

## Project Overview

- **Model**: Qwen3 0.6B (in BF16 format)
- **Task**: Binary Sentiment Classification (Positive=1, Negative=0)
- **Method**: Full-parameter fine-tuning (not LoRA)
- **Framework**: MLX-LM
- **Hardware**: Recommended 16GB+ RAM (tested on a Mac Studio with 70GB RAM).

## Data Format

The project uses a custom prompt-completion format for training.

### Input Data (TSV Format)
The initial data should be in a TSV file with the following columns:
```
qid	label	text_a
0	1	这间酒店环境和服务态度亦算不错...
1	1	推荐所有喜欢红楼的红迷们...
2	0	商品的不足暂时还没发现...
```

### Processed Data (JSONL Format)
The `prepare_data.py` script converts the TSV file into the required JSONL format for training:
```json
{"prompt": "请判断以下文本的情感倾向，正面回复'1'，负面回复'0'。\n文本：这间酒店环境和服务态度亦算不错...\n情感：", "completion": "1"}
```
*Note: The prompt is in Chinese as the task is Chinese sentiment analysis.*

## Project Structure

```
qwen3_sentiment_finetune/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── config/
│   └── training_config.yaml      # Training configuration
├── data/
│   ├── prepare_data.py           # Data preprocessing script
│   └── processed/                # (Git-ignored) Processed data
├── src/
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   └── inference.py              # Inference script
├── models/                       # (Git-ignored) Saved models
└── logs/                         # (Git-ignored) Training logs
```

## How to Use

### 1. Setup Environment

Ensure you have MLX and the required Python packages installed.
```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place your raw TSV data file (e.g., `train.tsv`) in the `data/` directory and run the preprocessing script.
```bash
python data/prepare_data.py --input_file data/train.tsv --output_dir data/processed
```
This will generate `train.jsonl`, `valid.jsonl`, and `test.jsonl` in the `data/processed/` directory.

### 3. Start Training

You can start training using the default configuration file. The script will automatically use the data from `data/processed` and save the model to `models/qwen3_sentiment`.
```bash
python src/train.py --config config/training_config.yaml
```

Alternatively, you can override specific parameters via the command line:
```bash
python src/train.py \
    --model_path "/path/to/your/qwen3_0.6b_bf16" \
    --data_path "data/processed" \
    --output_dir "models/qwen3_sentiment_custom" \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --num_epochs 3
```

### 4. Evaluate the Model

Evaluate the fine-tuned model on the test set.
```bash
python src/evaluate.py \
    --model_path "models/qwen3_sentiment" \
    --test_data "data/processed/test.jsonl"
```

### 5. Run Inference

Test the model with a sample text.
```bash
python src/inference.py \
    --model_path "models/qwen3_sentiment" \
    --text "这个产品质量很好，我很满意"
```

## Technical Details

### Memory Usage Estimation
- **Qwen3 0.6B Parameters**: ~1.2GB (BF16)
- **Gradients**: ~1.2GB
- **AdamW Optimizer States**: ~2.4GB
- **Batch Data & Activations**: ~2-5GB
- **Total Estimated**: ~7-10GB

### Training Configuration (`training_config.yaml`)

| Parameter | Default Value | Description |
|---|---|---|
| `model_path` | ... | Path to the base Qwen3 model. |
| `data_path` | `data/processed` | Path to the processed dataset. |
| `output_dir` | `models/qwen3_sentiment` | Directory to save checkpoints. |
| `batch_size` | 8 | Batch size. Adjust based on VRAM. |
| `learning_rate` | 5e-5 | Learning rate for full-parameter tuning. |
| `max_seq_length` | 256 | Max sequence length. |
| `num_epochs` | 3 | Number of training epochs. |
| `warmup_steps` | 100 | Learning rate warmup steps. |
| `eval_steps` | 200 | Evaluate every N steps. |
| `save_steps` | 500 | Save a checkpoint every N steps. |
| `gradient_checkpointing` | `false` | Enable to save memory at the cost of speed. |

## FAQ

**Q: I'm running out of memory.**
**A:** Try the following:
1.  Reduce `batch_size` in `training_config.yaml`.
2.  Set `gradient_checkpointing` to `true`.
3.  Consider using LoRA for a less memory-intensive approach.

**Q: Training is too slow.**
**A:**
1.  Increase `batch_size` if you have available memory.
2.  Set `gradient_checkpointing` to `false`.
3.  Reduce `max_seq_length` if your text is generally short.

**Q: How to prevent overfitting?**
**A:**
1.  Decrease the `learning_rate`.
2.  Use a smaller `num_epochs`.
3.  Add more diverse training data.

## Resources

- [MLX-LM Official Documentation](https://github.com/ml-explore/mlx-lm)
- [Qwen3 Model Series](https://github.com/QwenLM/Qwen)
