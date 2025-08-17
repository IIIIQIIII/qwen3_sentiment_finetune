#!/usr/bin/env python3
"""
Qwen3-0.5B Sentiment Analysis Model Evaluation Script

This script evaluates the performance of the fine-tuned model on a test set.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    classification_report,
    confusion_matrix
)
from tqdm import tqdm

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


def load_test_data(test_file: str) -> List[Dict]:
    """Loads test data from a JSONL file."""
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        sys.exit(1)
    
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    
    print(f"✅ Loaded test data: {len(test_data)} records")
    return test_data


def predict_sentiment(model, tokenizer, text: str, sampler) -> int:
    """
    Predicts the sentiment of a single text.
    
    Args:
        model: The loaded model.
        tokenizer: The tokenizer.
        text: The input text.
        sampler: The sampler for generation.
    
    Returns:
        The predicted sentiment (0=negative, 1=positive).
    """
    # This prompt is in Chinese as the fine-tuned task is Chinese sentiment analysis.
    prompt = f"请判断以下文本的情感倾向，正面回复'1'，负面回复'0'。\n文本：{text}\n情感："
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template
    chat_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    # Generate response
    output = generate(
        model,
        tokenizer,
        prompt=chat_input,
        max_tokens=4,  # Only need a single digit
        sampler=sampler
    ).strip()
    
    # Extract prediction
    if "1" in output:
        return 1
    elif "0" in output:
        return 0
    else:
        # Default to 0 to avoid errors
        return 0


def extract_text_from_prompt(prompt: str) -> str:
    """Extracts the original text from the prompt string."""
    # Prompt format: "请判断以下文本的情感倾向，正面回复'1'，负面回复'0'。\n文本：{text}\n情感："
    if "文本：" in prompt and "\n情感：" in prompt:
        start = prompt.find("文本：") + 3
        end = prompt.find("\n情感：")
        return prompt[start:end].strip()
    return prompt


def evaluate_model(model, tokenizer, test_data: List[Dict], sampler) -> Dict:
    """Evaluates the model's performance."""
    predictions = []
    true_labels = []
    texts = []
    
    print("🔄 Starting model evaluation...")
    
    for item in tqdm(test_data, desc="Predicting"):
        # Extract true label
        true_label = int(item['completion'])
        true_labels.append(true_label)
        
        # Extract text
        text = extract_text_from_prompt(item['prompt'])
        texts.append(text)
        
        # Predict
        pred = predict_sentiment(model, tokenizer, text, sampler)
        predictions.append(pred)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average=None, labels=[0, 1]
    )
    
    # Calculate macro and micro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro'
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        true_labels, predictions, average='micro'
    )
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
    
    results = {
        "accuracy": accuracy,
        "precision": {
            "negative": precision[0],
            "positive": precision[1],
            "macro": precision_macro,
            "micro": precision_micro
        },
        "recall": {
            "negative": recall[0],
            "positive": recall[1],
            "macro": recall_macro,
            "micro": recall_micro
        },
        "f1": {
            "negative": f1[0],
            "positive": f1[1],
            "macro": f1_macro,
            "micro": f1_micro
        },
        "support": {
            "negative": int(support[0]),
            "positive": int(support[1])
        },
        "confusion_matrix": cm.tolist(),
        "predictions": predictions,
        "true_labels": true_labels,
        "texts": texts
    }
    
    return results


def print_evaluation_results(results: Dict):
    """Prints the evaluation results."""
    print("\n" + "="*60)
    print("📊 Model Evaluation Results")
    print("="*60)
    
    print(f"\n🎯 Overall Performance:")
    print(f"   Accuracy: {results['accuracy']:.4f}")
    print(f"   F1-Score (Macro):   {results['f1']['macro']:.4f}")
    print(f"   F1-Score (Micro):   {results['f1']['micro']:.4f}")
    
    print(f"\n📈 Per-Class Performance:")
    print(f"   Negative Sentiment (0):")
    print(f"     Precision: {results['precision']['negative']:.4f}")
    print(f"     Recall:    {results['recall']['negative']:.4f}")
    print(f"     F1-Score:  {results['f1']['negative']:.4f}")
    print(f"     Support:   {results['support']['negative']}")
    
    print(f"\n   Positive Sentiment (1):")
    print(f"     Precision: {results['precision']['positive']:.4f}")
    print(f"     Recall:    {results['recall']['positive']:.4f}")
    print(f"     F1-Score:  {results['f1']['positive']:.4f}")
    print(f"     Support:   {results['support']['positive']}")
    
    print(f"\n🔍 Confusion Matrix:")
    cm = results['confusion_matrix']
    print(f"                Predicted")
    print(f"              Neg(0) Pos(1)")
    print(f"    Actual 0  {cm[0][0]:6d} {cm[0][1]:6d}")
    print(f"           1  {cm[1][0]:6d} {cm[1][1]:6d}")


def save_detailed_results(results: Dict, output_file: str, model_path: str):
    """Saves detailed evaluation results to a JSON file."""
    # Create detailed results dictionary
    detailed_results = {
        "model_path": model_path,
        "evaluation_summary": {
            "total_samples": len(results['true_labels']),
            "accuracy": results['accuracy'],
            "precision_macro": results['precision']['macro'],
            "recall_macro": results['recall']['macro'],
            "f1_macro": results['f1']['macro'],
            "precision_micro": results['precision']['micro'],
            "recall_micro": results['recall']['micro'],
            "f1_micro": results['f1']['micro']
        },
        "class_metrics": {
            "negative": {
                "precision": results['precision']['negative'],
                "recall": results['recall']['negative'],
                "f1": results['f1']['negative'],
                "support": results['support']['negative']
            },
            "positive": {
                "precision": results['precision']['positive'],
                "recall": results['recall']['positive'],
                "f1": results['f1']['positive'],
                "support": results['support']['positive']
            }
        },
        "confusion_matrix": results['confusion_matrix'],
        "predictions": results['predictions'],
        "true_labels": results['true_labels']
    }
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Detailed results saved to: {output_file}")


def save_error_analysis(results: Dict, output_file: str):
    """Saves an error analysis report."""
    predictions = results['predictions']
    true_labels = results['true_labels']
    texts = results['texts']
    
    errors = []
    for i, (pred, true, text) in enumerate(zip(predictions, true_labels, texts)):
        if pred != true:
            errors.append({
                "index": i,
                "text": text,
                "true_label": int(true),
                "predicted_label": int(pred),
                "error_type": "false_positive" if pred == 1 and true == 0 else "false_negative"
            })
    
    error_analysis = {
        "total_errors": len(errors),
        "error_rate": len(errors) / len(predictions) if predictions else 0,
        "false_positives": len([e for e in errors if e["error_type"] == "false_positive"]),
        "false_negatives": len([e for e in errors if e["error_type"] == "false_negative"]),
        "errors": errors
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(error_analysis, f, ensure_ascii=False, indent=2)
    
    print(f"🔍 Error analysis saved to: {output_file}")
    print(f"   Total errors: {len(errors)} / {len(predictions)} ({error_analysis['error_rate']*100:.1f}%)")
    print(f"   False Positives: {error_analysis['false_positives']}")
    print(f"   False Negatives: {error_analysis['false_negatives']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Qwen3 sentiment analysis model.")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the fine-tuned model directory.")
    parser.add_argument("--test_data", type=str, 
                       default="data/processed/test.jsonl",
                       help="Path to the test data file (JSONL format).")
    parser.add_argument("--output_dir", type=str, 
                       default="evaluation_results",
                       help="Directory to save evaluation results.")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (currently only 1 is supported).")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📊 Qwen3-0.5B Sentiment Analysis Model Evaluation 📊")
    print("=" * 60)
    
    # Check model path
    if not os.path.exists(args.model_path):
        print(f"❌ Model path not found: {args.model_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    print(f"🔄 Loading model from: {args.model_path}")
    try:
        model, tokenizer = load(
            args.model_path,
            tokenizer_config={"trust_remote_code": True}
        )
        model.eval()
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Setup sampler (use consistent parameters)
    sampler = make_sampler(
        temp=0.7,   # Official recommendation for Qwen3
        top_p=0.8,  # Official recommendation for Qwen3
        top_k=20    # Official recommendation for Qwen3
    )
    print(f"✅ Sampler configured successfully")
    
    # Load test data
    test_data = load_test_data(args.test_data)
    
    # Evaluate model
    results = evaluate_model(model, tokenizer, test_data, sampler)
    
    # Print results
    print_evaluation_results(results)
    
    # Save detailed results
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    save_detailed_results(results, results_file, args.model_path)
    
    # Save error analysis
    error_file = os.path.join(args.output_dir, "error_analysis.json")
    save_error_analysis(results, error_file)
    
    # Save classification report (sklearn format)
    report_file = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        report = classification_report(
            results['true_labels'], 
            results['predictions'],
            target_names=['Negative', 'Positive'], # Use English labels
            digits=4
        )
        f.write(report)
    print(f"📄 Classification report saved to: {report_file}")
    
    print(f"\n🎉 Evaluation complete!")
    print(f"   Results saved in: {args.output_dir}")
    print(f"   Overall Accuracy: {results['accuracy']:.4f}")
    print(f"   F1-Score (Macro): {results['f1']['macro']:.4f}")


if __name__ == "__main__":
    main()
