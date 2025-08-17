#!/usr/bin/env python3
"""
Qwen3-0.5B Full-Parameter Fine-Tuning Script

This script uses the MLX-LM tuner API to perform full-parameter fine-tuning.
"""

import argparse
import json
import logging
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten

# MLX-LM imports
from mlx_lm import load
from mlx_lm.utils import save_model
from mlx_lm.tuner import train, TrainingArgs
from mlx_lm.tuner.datasets import load_local_dataset, CacheDataset
from mlx_lm.tuner.utils import dequantize, print_trainable_parameters, build_schedule
from mlx_lm.tuner.callbacks import TrainingCallback


class SentimentTrainingCallback(TrainingCallback):
    """Custom training callback to monitor and log the training process."""
    
    def __init__(self, log_dir: str, save_steps: int = 500):
        self.log_dir = log_dir
        self.save_steps = save_steps
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_log = []
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logger
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Sets up the logger."""
        logger = logging.getLogger('sentiment_training')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(
            os.path.join(self.log_dir, 'training.log'), 
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def on_train_loss_report(self, info: Dict[str, Any]):
        """Callback for reporting training loss."""
        self.logger.info(
            f"Step {info['iteration']}: "
            f"Training Loss {info['train_loss']:.4f}, "
            f"Learning Rate {info['learning_rate']:.2e}, "
            f"Throughput {info['iterations_per_second']:.2f} it/s, "
            f"Peak Memory {info['peak_memory']:.2f} GB"
        )
        
        # Append to training log
        self.training_log.append({
            'type': 'train',
            'iteration': info['iteration'],
            'loss': info['train_loss'],
            'learning_rate': info['learning_rate'],
            'timestamp': time.time()
        })
    
    def on_val_loss_report(self, info: Dict[str, Any]):
        """Callback for reporting validation loss."""
        val_loss = info['val_loss']
        iteration = info['iteration']
        
        self.logger.info(
            f"Step {iteration}: Validation Loss {val_loss:.4f}"
        )
        
        # Append to training log
        self.training_log.append({
            'type': 'val',
            'iteration': iteration,
            'loss': val_loss,
            'timestamp': time.time()
        })
        
        # Check for best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.logger.info(f"🎉 New best model found! Validation Loss: {val_loss:.4f}")
        else:
            self.patience_counter += 1
        
        # Save training log
        self._save_training_log()
    
    def _save_training_log(self):
        """Saves the training log to a file."""
        log_file = os.path.join(self.log_dir, 'training_metrics.json')
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_log, f, ensure_ascii=False, indent=2)


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Failed to load config file: {e}")
        sys.exit(1)


def setup_model_and_tokenizer(config: Dict[str, Any]):
    """Sets up the model and tokenizer."""
    model_config = config['model']
    model_path = model_config['path']
    
    print(f"🔄 Loading model from: {model_path}")
    
    # Load model and tokenizer
    try:
        model, tokenizer = load(
            model_path,
            tokenizer_config={"trust_remote_code": model_config.get('trust_remote_code', True)}
        )
        print(f"✅ Model loaded successfully")
        
        # Dequantize if the model is quantized, for full-parameter tuning
        if hasattr(model, 'layers') and any(
            hasattr(layer, 'scales') for layer in model.layers
        ):
            print("🔄 Detected quantized model, dequantizing for full-parameter tuning...")
            model = dequantize(model)
            print("✅ Model dequantized successfully")
        
        # Set to training mode
        model.train()
        
        # Print trainable parameters
        print_trainable_parameters(model)
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)


def setup_datasets(config: Dict[str, Any], tokenizer):
    """Sets up the datasets."""
    data_config = config['data']
    
    print(f"🔄 Loading datasets...")
    
    try:
        # Create data path object
        data_path = Path("data/processed")
        if not data_path.exists():
            print(f"❌ Data directory not found: {data_path}")
            print("Please run the data preprocessing script first: python data/prepare_data.py")
            sys.exit(1)
        
        # Create a simple namespace object for dataset config
        import types
        dataset_config = types.SimpleNamespace(**data_config)
        
        # Load datasets
        train_dataset, valid_dataset, test_dataset = load_local_dataset(
            data_path, tokenizer, dataset_config
        )
        
        # Wrap datasets with CacheDataset for correct token processing
        train_dataset = CacheDataset(train_dataset)
        valid_dataset = CacheDataset(valid_dataset)
        test_dataset = CacheDataset(test_dataset)
        
        print(f"✅ Datasets loaded successfully")
        print(f"   Training set: {len(train_dataset)} records")
        print(f"   Validation set: {len(valid_dataset)} records")
        print(f"   Test set: {len(test_dataset)} records")
        
        return train_dataset, valid_dataset, test_dataset
        
    except Exception as e:
        print(f"❌ Failed to load datasets: {e}")
        sys.exit(1)


def setup_optimizer(model, config: Dict[str, Any]):
    """Sets up the optimizer and learning rate scheduler."""
    training_config = config['training']
    optimizer_config = config['optimizer']
    
    learning_rate = training_config['learning_rate']
    weight_decay = training_config.get('weight_decay', 0.01)
    
    # Create learning rate schedule
    if 'lr_scheduler' in training_config:
        lr_schedule = build_schedule(training_config['lr_scheduler'])
    else:
        lr_schedule = learning_rate
    
    # Create optimizer
    optimizer_name = optimizer_config.get('name', 'adamw').lower()
    
    if optimizer_name == 'adamw':
        optimizer = opt.AdamW(
            learning_rate=lr_schedule,
            betas=optimizer_config.get('betas', [0.9, 0.999]),
            eps=optimizer_config.get('eps', 1e-8),
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd':
        optimizer = opt.SGD(
            learning_rate=lr_schedule,
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    print(f"✅ Optimizer configured: {optimizer_name}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Weight Decay: {weight_decay}")
    
    return optimizer


def setup_training_args(config: Dict[str, Any]) -> TrainingArgs:
    """Sets up the training arguments."""
    training_config = config['training']
    eval_config = config['evaluation']
    output_config = config['output']
    data_config = config['data']
    hardware_config = config.get('hardware', {})
    
    # Calculate total training iterations
    if training_config.get('max_steps', -1) > 0:
        iters = training_config['max_steps']
    else:
        # Estimate iters based on epochs (this is a rough estimate)
        epochs = training_config.get('num_epochs', 3)
        # A placeholder value, as the actual steps depend on dataset size.
        # The trainer will handle the exact number of steps per epoch.
        iters = epochs * 1000  # Rough estimation
    
    args = TrainingArgs(
        batch_size=training_config['batch_size'],
        iters=iters,
        val_batches=eval_config.get('eval_batches', 25),
        steps_per_report=eval_config.get('logging_steps', 50),
        steps_per_eval=eval_config.get('eval_steps', 200),
        steps_per_save=eval_config.get('save_steps', 500),
        max_seq_length=data_config.get('max_seq_length', 256),
        adapter_file=os.path.join(output_config['output_dir'], "adapters.safetensors"),
        grad_checkpoint=hardware_config.get('gradient_checkpointing', False)
    )
    
    print(f"✅ Training arguments configured")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Max Sequence Length: {args.max_seq_length}")
    print(f"   Training Iterations: {args.iters}")
    print(f"   Gradient Checkpointing: {args.grad_checkpoint}")
    
    return args


def main():
    parser = argparse.ArgumentParser(description="Qwen3-0.5B Sentiment Analysis Full-Parameter Fine-Tuning")
    parser.add_argument("--config", type=str, default="config/training_config.yaml", 
                       help="Path to the configuration file.")
    parser.add_argument("--model_path", type=str, help="Path to the model (overrides config file).")
    parser.add_argument("--output_dir", type=str, help="Output directory (overrides config file).")
    parser.add_argument("--batch_size", type=int, help="Batch size (overrides config file).")
    parser.add_argument("--learning_rate", type=float, help="Learning rate (overrides config file).")
    
    cli_args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Starting Qwen3-0.5B Sentiment Analysis Full-Parameter Fine-Tuning 🚀")
    print("=" * 60)
    
    # Load config
    print(f"📖 Loading configuration from: {cli_args.config}")
    config = load_config(cli_args.config)
    
    # Override config with command-line arguments if provided
    if cli_args.model_path:
        config['model']['path'] = cli_args.model_path
    if cli_args.output_dir:
        config['output']['output_dir'] = cli_args.output_dir
    if cli_args.batch_size:
        config['training']['batch_size'] = cli_args.batch_size
    if cli_args.learning_rate:
        config['training']['learning_rate'] = cli_args.learning_rate
    
    # Create output directory
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the final configuration to the output directory
    config_save_path = os.path.join(output_dir, "training_config.yaml")
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    print(f"📄 Configuration saved to: {config_save_path}")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Setup datasets
    train_dataset, valid_dataset, test_dataset = setup_datasets(config, tokenizer)
    
    # Setup optimizer
    optimizer = setup_optimizer(model, config)
    
    # Setup training arguments
    training_args = setup_training_args(config)
    
    # Setup training callback
    log_dir = config['logging']['log_dir']
    callback = SentimentTrainingCallback(
        log_dir=log_dir,
        save_steps=training_args.steps_per_save
    )
    
    print(f"\n🎯 Starting training...")
    print(f"   Output directory: {output_dir}")
    print(f"   Log directory: {log_dir}")
    
    try:
        # Start training (uses default cross_entropy loss)
        train(
            model=model,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=valid_dataset,
            args=training_args,
            training_callback=callback
        )
        
        print(f"\n✅ Training complete!")
        
        # Save the final model
        print(f"💾 Saving final model to: {output_dir}")
        
        # Save the full model and tokenizer
        save_model(output_dir, model, tokenizer)
        
        # Save a final model config
        model_config = config['model'].copy()
        model_config.update({
            "fine_tuned": True,
            "fine_tune_type": "full",
            "task_type": "sentiment_analysis",
            "training_config": config
        })
        
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(model_config, f, ensure_ascii=False, indent=2)
        
        print(f"🎉 Full-parameter fine-tuning finished successfully!")
        print(f"   Model saved in: {output_dir}")
        print(f"   Logs saved in: {log_dir}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user.")
        print(f"💾 Saving current model state...")
        
        # Save model state upon interruption
        interrupt_dir = os.path.join(output_dir, "interrupted")
        os.makedirs(interrupt_dir, exist_ok=True)
        
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(
            os.path.join(interrupt_dir, "adapters.safetensors"), 
            adapter_weights
        )
        
        print(f"✅ Model state saved to: {interrupt_dir}")
        
    except Exception as e:
        print(f"\n❌ An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
