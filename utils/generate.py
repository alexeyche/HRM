import argparse
import random
from typing import List

from dataset.grammar import sample_programs, parse_program, get_cfg


def main():
    parser = argparse.ArgumentParser(description="Generate sample programs from the CFG")
    parser.add_argument("--n", type=int, default=100, help="Number of programs to generate")
    parser.add_argument("--max_depth", type=int, default=10, help="Max derivation depth for generation")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    cfg = get_cfg()
    programs: List[str] = sample_programs(cfg, args.n, max_depth=args.max_depth, seed=args.seed)

    for i, program in enumerate(programs, start=1):
        status = "✅" if parse_program(program) else "❌"
        print(f"===== Program {i}, compiled: {status} =====")
        print(program)
        print()


if __name__ == "__main__":
    main()


