import os
import sys
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================
# Editable parameters
# =========================
BASE_MODEL_PATH = "model/qwen"  # Base model path

# Generation parameters
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50
DO_SAMPLE = True

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer():
    """Load base model and tokenizer."""
    print(f"Loading base model from: {BASE_MODEL_PATH}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        use_fast=True,
        trust_remote_code=True
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.eval()
    
    print(f"Model loaded successfully on {DEVICE}")
    return model, tokenizer


def chat(model, tokenizer, messages: List[Dict[str, str]]) -> str:
    """Generate response for the conversation."""
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
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


def main():
    """Main interactive chat loop."""
    print("="*60)
    print("Base Qwen3 Model Interactive Chat")
    print("="*60)
    print("Commands:")
    print("  - Type your message to chat")
    print("  - Type 'clear' to reset conversation")
    print("  - Type 'quit' or 'exit' to quit")
    print("="*60)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Initialize conversation history
    messages = []
    
    print("\nReady to chat! Type your message:\n")
    
    while True:
        # Get user input
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break
        
        # Handle commands
        if user_input.lower() in ["quit", "exit"]:
            print("\nGoodbye!")
            break
        
        if user_input.lower() == "clear":
            messages = []
            print("\n[Conversation cleared]\n")
            continue
        
        if not user_input:
            continue
        
        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        # Generate response
        try:
            response = chat(model, tokenizer, messages)
            print(f"Assistant: {response}\n")
            
            # Add assistant response to history
            messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            print(f"\n[Error generating response: {e}]\n")
            # Remove the last user message if generation failed
            messages.pop()


if __name__ == "__main__":
    main()
