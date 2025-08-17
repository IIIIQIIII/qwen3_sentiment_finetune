#!/usr/bin/env python3
"""
Qwen3 0.6B Sentiment Analysis Inference Script

This script runs single-text, batch, or interactive inference for sentiment analysis
using the fine-tuned model.
"""

import argparse
import json
import os
import sys
from typing import List, Dict

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


def predict_sentiment(model, tokenizer, text: str, sampler, verbose: bool = False) -> Dict:
    """
    Predicts the sentiment of a single text.
    
    Args:
        model: The loaded model.
        tokenizer: The tokenizer.
        text: The input text.
        sampler: The sampler for generation.
        verbose: Whether to print detailed information.
    
    Returns:
        A dictionary containing the prediction results.
    """
    # This prompt is in Chinese as the fine-tuned task is Chinese sentiment analysis.
    prompt = f"请判断以下文本的情感倾向，正面回复\'1\'，负面回复\'0\'。\n文本：{text}\n情感："
    
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
    
    if verbose:
        print(f"📝 Input Prompt: {chat_input}")
    
    # Generate response
    output = generate(
        model,
        tokenizer,
        prompt=chat_input,
        max_tokens=10,  # A bit more to catch potential explanations from the model
        sampler=sampler,
        verbose=verbose
    ).strip()
    
    if verbose:
        print(f"🤖 Model Output: {output}")
    
    # Parse the prediction
    prediction = None
    confidence = "high"
    sentiment_en = ""
    
    if "1" in output and "0" not in output:
        prediction = 1
        sentiment_en = "Positive"
    elif "0" in output and "1" not in output:
        prediction = 0
        sentiment_en = "Negative"
    elif "1" in output and "0" in output:
        # If both 0 and 1 are present, check which comes first
        idx_0 = output.find("0")
        idx_1 = output.find("1")
        if idx_1 < idx_0:
            prediction = 1
            sentiment_en = "Positive"
        else:
            prediction = 0
            sentiment_en = "Negative"
        confidence = "low"
    else:
        # Default case
        prediction = 0
        sentiment_en = "Negative"
        confidence = "very_low"
    
    return {
        "text": text,
        "prediction": prediction,
        "sentiment": sentiment_en,
        "confidence": confidence,
        "raw_output": output,
        "prompt": prompt
    }


def batch_predict(model, tokenizer, texts: List[str], sampler, verbose: bool = False) -> List[Dict]:
    """
    Performs batch prediction on a list of texts.
    
    Args:
        model: The loaded model.
        tokenizer: The tokenizer.
        texts: A list of texts.
        sampler: The sampler for generation.
        verbose: Whether to print detailed information.
    
    Returns:
        A list of prediction results.
    """
    results = []
    
    print(f"🔄 Starting batch prediction for {len(texts)} texts...")
    
    for i, text in enumerate(texts):
        if verbose:
            print(f"\n--- Processing text {i+1}/{len(texts)} ---")
        
        result = predict_sentiment(model, tokenizer, text, sampler, verbose)
        results.append(result)
        
        if not verbose:
            # Show brief progress
            sentiment_emoji = "😊" if result['prediction'] == 1 else "😞"
            print(f"[{i+1}/{len(texts)}] {sentiment_emoji} {result['sentiment']}: {text[:50]}{'...' if len(text) > 50 else ''}")
    
    return results


def load_texts_from_file(file_path: str) -> List[str]:
    """Loads a list of texts from a file (one text per line)."""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    
    print(f"✅ Loaded {len(texts)} texts from file.")
    return texts


def save_results(results: List[Dict], output_file: str):
    """Saves the prediction results to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Results saved to: {output_file}")


def print_summary(results: List[Dict]):
    """Prints a summary of the prediction results."""
    if not results:
        return
    
    total = len(results)
    positive = sum(1 for r in results if r['prediction'] == 1)
    negative = total - positive
    
    print(f"\n📊 Prediction Summary:")
    print(f"   Total texts: {total}")
    print(f"   Positive: {positive} ({positive/total*100:.1f}%) ")
    print(f"   Negative: {negative} ({negative/total*100:.1f}%)")
    
    # Confidence statistics
    high_conf = sum(1 for r in results if r['confidence'] == 'high')
    low_conf = sum(1 for r in results if r['confidence'] == 'low')
    very_low_conf = sum(1 for r in results if r['confidence'] == 'very_low')
    
    print(f"\n🎯 Confidence Distribution:")
    print(f"   High Confidence: {high_conf} ({high_conf/total*100:.1f}%)")
    if low_conf > 0:
        print(f"   Low Confidence: {low_conf} ({low_conf/total*100:.1f}%)")
    if very_low_conf > 0:
        print(f"   Very Low Confidence: {very_low_conf} ({very_low_conf/total*100:.1f}%)")


def interactive_mode(model, tokenizer, sampler):
    """Runs the interactive inference mode."""
    print("\n🎮 Entering interactive mode (type 'quit' or 'exit' to stop)")
    print("=" * 50)
    
    while True:
        try:
            text = input("\nEnter text to analyze: ").strip()
            
            if text.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            if not text:
                print("⚠️ Please enter non-empty text.")
                continue
            
            result = predict_sentiment(model, tokenizer, text, sampler, verbose=False)
            
            # Display result
            sentiment_emoji = "😊" if result['prediction'] == 1 else "😞"
            print(f"\n{sentiment_emoji} Sentiment Analysis Result:")
            print(f"   Text: {text}")
            print(f"   Prediction: {result['sentiment']} ({result['prediction']})")
            print(f"   Confidence: {result['confidence']}")
            if result['confidence'] != 'high':
                print(f"   Raw Output: {result['raw_output']}")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3 Sentiment Analysis Inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the fine-tuned model directory.")
    parser.add_argument("--text", type=str,
                       help="A single text to analyze.")
    parser.add_argument("--input_file", type=str,
                       help="Path to an input file with one text per line.")
    parser.add_argument("--output_file", type=str,
                       help="Path to save the output results (JSON format).")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode.")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed information during inference.")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔮 Qwen3 0.6B Sentiment Analysis Inference 🔮")
    print("=" * 60)
    
    # Check model path
    if not os.path.exists(args.model_path):
        print(f"❌ Model path not found: {args.model_path}")
        sys.exit(1)
    
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
    
    # Choose mode based on arguments
    if args.interactive:
        # Interactive mode
        interactive_mode(model, tokenizer, sampler)
    
    elif args.text:
        # Single text mode
        print(f"\n🔍 Analyzing single text:")
        result = predict_sentiment(model, tokenizer, args.text, sampler, args.verbose)
        
        sentiment_emoji = "😊" if result['prediction'] == 1 else "😞"
        print(f"\n{sentiment_emoji} Analysis Result:")
        print(f"   Text: {result['text']}")
        print(f"   Prediction: {result['sentiment']} ({result['prediction']})")
        print(f"   Confidence: {result['confidence']}")
        
        if args.verbose:
            print(f"   Raw Output: {result['raw_output']}")
            print(f"   Prompt Used: {result['prompt']}")
        
        # Save result
        if args.output_file:
            save_results([result], args.output_file)
    
    elif args.input_file:
        # Batch file mode
        texts = load_texts_from_file(args.input_file)
        results = batch_predict(model, tokenizer, texts, sampler, args.verbose)
        
        # Print summary
        print_summary(results)
        
        # Save results
        if args.output_file:
            save_results(results, args.output_file)
        else:
            # Default save to current directory
            default_output = "inference_results.json"
            save_results(results, default_output)
    
    else:
        # No input specified, show help and start interactive mode
        print("⚠️ No input text specified. Starting interactive mode...")
        interactive_mode(model, tokenizer, sampler)


if __name__ == "__main__":
    main()