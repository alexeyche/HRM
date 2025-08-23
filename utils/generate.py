import argparse
import random
import torch
from typing import List

from dataset.grammar import sample_programs, parse_program_with_ast, get_cfg, realize_program
from models.generation_head import GrammarAwareGenerationHead


def generate_with_ml_head(n: int, max_steps: int, hidden_dim: int, seed: int | None = None) -> List[str]:
    """Generate programs using the ML-based grammar-aware generation head."""
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    cfg = get_cfg()
    gen_head = GrammarAwareGenerationHead(hidden_dim, cfg)

    programs = []
    for i in range(n):
        # Use different random context for each program
        context_embeddings = torch.randn(1, 1, hidden_dim)

        # Generate program tokens
        program_tokens_batch = gen_head.generate_program(context_embeddings, max_steps=max_steps)
        program_tokens = program_tokens_batch[0]

        # Convert tokens to code
        try:
            program_code = realize_program(program_tokens)
            programs.append(program_code)
        except Exception as e:
            programs.append(f"# Error realizing tokens: {e}\n# Tokens: {program_tokens}")

    return programs


def main():
    parser = argparse.ArgumentParser(description="Generate sample programs from the CFG")
    parser.add_argument("--n", type=int, default=20, help="Number of programs to generate")
    parser.add_argument("--max_depth", type=int, default=10, help="Max derivation depth for generation (grammar-based only)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # ML-based generation options
    parser.add_argument("--ml", action="store_true", help="Use ML-based grammar-aware generation head instead of grammar sampling")
    parser.add_argument("--max_steps", type=int, default=50, help="Max expansion steps for ML generation")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for ML generation head")

    # Output options
    parser.add_argument("--show_tokens", action="store_true", help="Show token sequence for ML generation")
    parser.add_argument("--only_valid", action="store_true", help="Only show syntactically valid programs")
    parser.add_argument("--stats", action="store_true", help="Show generation statistics")

    args = parser.parse_args()

    cfg = get_cfg()

    if args.ml:
        print(f"🤖 Generating {args.n} programs using ML-based grammar-aware head")
        print(f"   Hidden dim: {args.hidden_dim}, Max steps: {args.max_steps}")
        if args.seed:
            print(f"   Seed: {args.seed}")
        print("=" * 60)

        programs = generate_with_ml_head(args.n, args.max_steps, args.hidden_dim, args.seed)
    else:
        print(f"📝 Generating {args.n} programs using grammar-based sampling")
        print(f"   Max depth: {args.max_depth}")
        if args.seed:
            print(f"   Seed: {args.seed}")
        print("=" * 60)

        programs = sample_programs(cfg, args.n, max_depth=args.max_depth, seed=args.seed)

    # Statistics tracking
    valid_count = 0
    total_tokens = 0

    shown_count = 0
    for i, program in enumerate(programs, start=1):
        is_valid = parse_program_with_ast(program)
        if is_valid:
            valid_count += 1

        # Skip invalid programs if --only_valid is set
        if args.only_valid and not is_valid:
            continue

        status = "✅" if is_valid else "❌"
        print(f"===== Program {i}, compiled: {status} =====")

        # Show tokens for ML generation if requested
        if args.ml and args.show_tokens:
            # Re-generate to get tokens (not efficient but good for debugging)
            torch.manual_seed(args.seed + i if args.seed else i)
            context = torch.randn(1, 1, args.hidden_dim)
            gen_head = GrammarAwareGenerationHead(args.hidden_dim, cfg)
            tokens = gen_head.generate_program(context, max_steps=args.max_steps)[0]
            print(f"Tokens ({len(tokens)}): {tokens}")
            total_tokens += len(tokens)
            print()

        print(program)
        print()
        shown_count += 1

    # Show statistics if requested
    if args.stats:
        print("=" * 60)
        print("📊 Generation Statistics:")
        print(f"   Total programs: {len(programs)}")
        print(f"   Valid programs: {valid_count}/{len(programs)} ({valid_count/len(programs)*100:.1f}%)")
        print(f"   Programs shown: {shown_count}")

        if args.ml and args.show_tokens and total_tokens > 0:
            avg_tokens = total_tokens / shown_count if shown_count > 0 else 0
            print(f"   Average tokens per program: {avg_tokens:.1f}")

        if args.ml:
            print(f"   ML Generation - Hidden dim: {args.hidden_dim}, Max steps: {args.max_steps}")
        else:
            print(f"   Grammar Generation - Max depth: {args.max_depth}")


if __name__ == "__main__":
    main()


