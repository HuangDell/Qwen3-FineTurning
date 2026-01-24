import os
import sys
import re
from typing import List, Dict, Tuple
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

# =========================
# Editable parameters
# =========================
MODEL_TYPE = "lora"  # "base", "lora", or "grpo"
BASE_MODEL_PATH = "model/qwen"  # Base model path
ADAPTER_PATH = "out/qwen3-4b-lora-medical"  # Path to trained adapter (used for lora/grpo)

# Dataset parameters
DATASET_NAME = "clinical_medicine"
BENCHMARK_PATH = "data/benchmark/ceval-exma"

# Output parameters
OUTPUT_DIR = "out/benchmark"

# Generation parameters
MAX_NEW_TOKENS = 10  # Only need A/B/C/D response
TEMPERATURE = 0.0  # 0 for greedy, 0.7 for sampling
TOP_P = 0.9
TOP_K = 50
DO_SAMPLE = False  # Set to True if TEMPERATURE > 0

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer():
    """Load model and tokenizer based on MODEL_TYPE."""
    print(f"Loading model with type: {MODEL_TYPE}")
    print(f"Base model path: {BASE_MODEL_PATH}")
    
    if MODEL_TYPE == "base":
        # Load base model only
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_PATH,
            use_fast=True,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
    elif MODEL_TYPE in ["lora", "grpo"]:
        # Load base model with adapter
        print(f"Adapter path: {ADAPTER_PATH}")
        
        # Load tokenizer from adapter path (includes saved tokenizer)
        tokenizer = AutoTokenizer.from_pretrained(
            ADAPTER_PATH,
            use_fast=True,
            trust_remote_code=True
        )
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load adapter
        print(f"Loading {MODEL_TYPE.upper()} adapter...")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model = model.merge_and_unload()  # Merge weights for faster inference
        
    else:
        raise ValueError(f"Invalid MODEL_TYPE: {MODEL_TYPE}. Must be 'base', 'lora', or 'grpo'.")
    
    model.eval()
    print(f"Model loaded successfully on {DEVICE}")
    return model, tokenizer


def load_dataset_splits():
    """Load dev and test splits from local parquet files."""
    print(f"\nLoading dataset: {DATASET_NAME}")
    print(f"Dataset path: {BENCHMARK_PATH}")
    
    # Load dataset from local directory
    dataset = load_dataset(
        BENCHMARK_PATH,
        name=DATASET_NAME,
    )
    
    dev_data = dataset['dev']
    test_data = dataset['test']
    
    print(f"Dev examples: {len(dev_data)}")
    print(f"Test examples: {len(test_data)}")
    
    return dev_data, test_data


def build_few_shot_prompt(dev_examples: List[Dict], test_example: Dict) -> str:
    """
    Construct a few-shot prompt with dev examples in Chinese format.
    
    Args:
        dev_examples: List of dev examples for few-shot learning
        test_example: The test question to answer
        
    Returns:
        Formatted prompt string
    """
    prompt = "以下是中国临床医学考试的单选题，请选出正确答案。你只需要回答A,B,C,D。\n\n"
    
    # Add few-shot examples
    for i, example in enumerate(dev_examples, 1):
        prompt += f"问题{i}: {example['question']}\n"
        prompt += f"A. {example['A']}\n"
        prompt += f"B. {example['B']}\n"
        prompt += f"C. {example['C']}\n"
        prompt += f"D. {example['D']}\n"
        prompt += f"答案: {example['answer']}\n\n"
    
    # Add test question
    prompt += f"问题{len(dev_examples) + 1}: {test_example['question']}\n"
    prompt += f"A. {test_example['A']}\n"
    prompt += f"B. {test_example['B']}\n"
    prompt += f"C. {test_example['C']}\n"
    prompt += f"D. {test_example['D']}\n"
    prompt += "答案:"
    
    return prompt


def extract_answer(response: str) -> str:
    """
    Extract A/B/C/D answer from model response.
    
    Args:
        response: Raw model response string
        
    Returns:
        Extracted answer ('A', 'B', 'C', 'D') or 'UNKNOWN' if not found
    """
    # Remove whitespace
    response = response.strip()
    
    # Try to find answer in various formats
    # Pattern 1: Direct answer (A, B, C, D at start)
    if response and response[0].upper() in ['A', 'B', 'C', 'D']:
        return response[0].upper()
    
    # Pattern 2: Answer with Chinese text (答案是A, 选A, etc.)
    pattern = r'[答选择是：:]\s*([A-D])'
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 3: Just find first occurrence of A, B, C, or D
    pattern = r'\b([A-D])\b'
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 4: Check if answer appears in the response
    for answer in ['A', 'B', 'C', 'D']:
        if answer in response.upper():
            return answer
    
    return 'UNKNOWN'


def generate_response(model, tokenizer, prompt: str) -> str:
    """
    Generate response from model for the given prompt.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: Input prompt string
        
    Returns:
        Generated response string
    """
    # Format as a chat message
    messages = [{"role": "user", "content": prompt}]
    
    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE if DO_SAMPLE else 1.0,
            top_p=TOP_P,
            top_k=TOP_K,
            do_sample=DO_SAMPLE,
            pad_token_id=model.generation_config.pad_token_id,
            eos_token_id=model.generation_config.eos_token_id,
        )
    
    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    return response.strip()


def evaluate_model(model, tokenizer, dev_data, test_data) -> List[Dict]:
    """
    Evaluate model on test data using few-shot prompting.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        dev_data: Dev examples for few-shot
        test_data: Test examples to evaluate
        
    Returns:
        List of result dictionaries
    """
    print("\n" + "="*60)
    print("Starting evaluation...")
    print("="*60)
    
    # Convert dev_data to list for few-shot examples
    dev_examples = [example for example in dev_data]
    
    results = []
    correct_count = 0
    total_count = len(test_data)
    
    for idx, test_example in enumerate(test_data):
        # Build prompt with few-shot examples
        prompt = build_few_shot_prompt(dev_examples, test_example)
        
        # Generate response
        response = generate_response(model, tokenizer, prompt)
        
        # Extract answer
        predicted_answer = extract_answer(response)
        correct_answer = test_example['answer']
        is_correct = (predicted_answer == correct_answer)
        
        if is_correct:
            correct_count += 1
        
        # Store result
        result = {
            'id': test_example['id'],
            'question': test_example['question'][:50] + "..." if len(test_example['question']) > 50 else test_example['question'],
            'predicted': predicted_answer,
            'correct': correct_answer,
            'is_correct': is_correct,
            'raw_response': response
        }
        results.append(result)
        
        # Print progress
        if (idx + 1) % 10 == 0:
            current_acc = (correct_count / (idx + 1)) * 100
            print(f"Progress: {idx + 1}/{total_count} | Accuracy: {current_acc:.2f}%")
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print("="*60)
    
    return results


def save_results(results: List[Dict], output_path: str):
    """
    Save evaluation results to a text file.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save the results
    """
    # Calculate metrics
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    incorrect = total - correct
    accuracy = (correct / total * 100) if total > 0 else 0
    unknown_count = sum(1 for r in results if r['predicted'] == 'UNKNOWN')
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write results to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("C-Eval Clinical Medicine Evaluation Results\n")
        f.write("="*70 + "\n\n")
        
        # Model information
        f.write("Model Configuration:\n")
        f.write(f"  Model Type: {MODEL_TYPE}\n")
        f.write(f"  Base Model: {BASE_MODEL_PATH}\n")
        if MODEL_TYPE in ["lora", "grpo"]:
            f.write(f"  Adapter Path: {ADAPTER_PATH}\n")
        f.write(f"  Dataset: {DATASET_NAME}\n")
        f.write(f"\n")
        
        # Generation parameters
        f.write("Generation Parameters:\n")
        f.write(f"  Temperature: {TEMPERATURE}\n")
        f.write(f"  Do Sample: {DO_SAMPLE}\n")
        f.write(f"  Max New Tokens: {MAX_NEW_TOKENS}\n")
        f.write(f"\n")
        
        # Evaluation info
        f.write("Evaluation Setup:\n")
        f.write(f"  Test Split: {total} questions\n")
        f.write(f"  Few-shot: 5 examples\n")
        f.write(f"\n")
        
        # Summary metrics
        f.write("="*70 + "\n")
        f.write("Summary Results:\n")
        f.write("="*70 + "\n")
        f.write(f"  Total Questions: {total}\n")
        f.write(f"  Correct: {correct}\n")
        f.write(f"  Incorrect: {incorrect}\n")
        f.write(f"  Failed Extractions: {unknown_count}\n")
        f.write(f"  Accuracy: {accuracy:.2f}%\n")
        f.write("\n")
        
        # Detailed results
        f.write("="*70 + "\n")
        f.write("Detailed Results:\n")
        f.write("="*70 + "\n")
        f.write(f"{'ID':<6} | {'Question':<40} | {'Pred':<5} | {'True':<5} | {'Status'}\n")
        f.write("-"*70 + "\n")
        
        for result in results:
            status = "✓" if result['is_correct'] else "✗"
            f.write(f"{result['id']:<6} | {result['question']:<40} | {result['predicted']:<5} | {result['correct']:<5} | {status}\n")
        
        f.write("\n")
        f.write("="*70 + "\n")
        f.write(f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n")
    
    print(f"\nResults saved to: {output_path}")
    print(f"\nFinal Accuracy: {accuracy:.2f}%")


def main():
    """Main evaluation function."""
    print("="*70)
    print("C-Eval Clinical Medicine Benchmark Evaluation")
    print("="*70)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Load dataset
    dev_data, test_data = load_dataset_splits()
    
    # Evaluate model
    results = evaluate_model(model, tokenizer, dev_data, test_data)
    
    # Determine model identifier for output filename
    if MODEL_TYPE == "base":
        model_id = "base"
    else:
        # Extract model name from adapter path
        adapter_name = os.path.basename(ADAPTER_PATH)
        model_id = f"{MODEL_TYPE}_{adapter_name}"
    
    # Save results
    output_path = os.path.join(
        OUTPUT_DIR,
        "ceval-exma",
        DATASET_NAME,
        f"{model_id}_results.txt"
    )
    save_results(results, output_path)
    
    print("\n" + "="*70)
    print("Evaluation finished successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
