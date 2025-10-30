"""
Script for interactive text generation.

Usage:
    python scripts/generate.py --model checkpoints/final/model.pt --tokenizer tokenizers/my_tokenizer

Interactive generation with a trained model.
"""

import argparse
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.llm import ContinualLLM, ModelConfig
from src.tokenization.bpe_tokenizer import BPETokenizer


def generate_text(model, tokenizer, prompt, device, **generation_kwargs):
    """Generate text from prompt."""
    model.eval()

    # Encode prompt
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=True)],
        dtype=torch.long,
        device=device
    )

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(input_ids, **generation_kwargs)

    # Decode
    generated_text = tokenizer.decode(
        generated_ids[0].tolist(),
        skip_special_tokens=True
    )

    return generated_text


def interactive_mode(model, tokenizer, device, generation_kwargs):
    """Interactive generation loop."""
    print("\n" + "="*60)
    print("Interactive Generation Mode")
    print("="*60)
    print("Type your prompts below. Commands:")
    print("  /quit - Exit")
    print("  /temp <value> - Set temperature (e.g., /temp 0.8)")
    print("  /tokens <n> - Set max tokens (e.g., /tokens 100)")
    print("  /topk <n> - Set top-k (e.g., /topk 40)")
    print("  /topp <p> - Set top-p (e.g., /topp 0.9)")
    print("="*60 + "\n")

    while True:
        try:
            prompt = input(">>> ").strip()

            if not prompt:
                continue

            # Handle commands
            if prompt == "/quit":
                print("Goodbye!")
                break

            elif prompt.startswith("/temp "):
                try:
                    temp = float(prompt.split()[1])
                    generation_kwargs['temperature'] = temp
                    print(f"✓ Temperature set to {temp}")
                except:
                    print("✗ Invalid temperature")
                continue

            elif prompt.startswith("/tokens "):
                try:
                    tokens = int(prompt.split()[1])
                    generation_kwargs['max_new_tokens'] = tokens
                    print(f"✓ Max tokens set to {tokens}")
                except:
                    print("✗ Invalid token count")
                continue

            elif prompt.startswith("/topk "):
                try:
                    topk = int(prompt.split()[1])
                    generation_kwargs['top_k'] = topk
                    print(f"✓ Top-k set to {topk}")
                except:
                    print("✗ Invalid top-k")
                continue

            elif prompt.startswith("/topp "):
                try:
                    topp = float(prompt.split()[1])
                    generation_kwargs['top_p'] = topp
                    print(f"✓ Top-p set to {topp}")
                except:
                    print("✗ Invalid top-p")
                continue

            # Generate
            print("\nGenerating...")
            generated = generate_text(
                model, tokenizer, prompt, device, **generation_kwargs
            )

            print(f"\n{generated}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate text with LLM")

    # Model
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True,
                       help="Path to tokenizer")

    # Generation
    parser.add_argument("--prompt", type=str,
                       help="Prompt for generation (if not provided, enters interactive mode)")
    parser.add_argument("--max_tokens", type=int, default=100,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40,
                       help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p (nucleus) sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                       help="Repetition penalty")

    # Device
    parser.add_argument("--device", type=str, default="mps",
                       help="Device (cpu, cuda, mps)")

    # Mode
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive mode")

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("="*60)
    print("Text Generation")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Device: {device}")
    print("="*60 + "\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BPETokenizer.load(args.tokenizer)
    print(f"✓ Loaded: {len(tokenizer)} tokens\n")

    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.model, map_location='cpu')

    config = ModelConfig(**checkpoint['config'])
    model = ContinualLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✓ Loaded: {model.get_num_params():,} parameters\n")

    # Generation kwargs
    generation_kwargs = {
        'max_new_tokens': args.max_tokens,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'repetition_penalty': args.repetition_penalty
    }

    print("Generation settings:")
    for key, value in generation_kwargs.items():
        print(f"  {key}: {value}")
    print()

    # Interactive or single generation
    if args.interactive or args.prompt is None:
        interactive_mode(model, tokenizer, device, generation_kwargs)
    else:
        # Single generation
        print(f"Prompt: {args.prompt}\n")
        print("Generating...\n")

        generated = generate_text(
            model, tokenizer, args.prompt, device, **generation_kwargs
        )

        print(generated)


if __name__ == "__main__":
    main()
