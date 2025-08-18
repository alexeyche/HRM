import argparse
import random
from typing import List

from dataset.grammar import sample_programs


def main():
    parser = argparse.ArgumentParser(description="Generate sample programs from the CFG")
    parser.add_argument("--level", default="ALL", help="Start symbol: ALL, LEVEL1.1, LEVEL1.2, LEVEL2.1, LEVEL2.2, LEVEL3.1, LEVEL3.2, LEVEL4.1")
    parser.add_argument("--n", type=int, default=5, help="Number of programs to generate")
    parser.add_argument("--max_depth", type=int, default=30, help="Max derivation depth for generation")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    programs: List[str] = sample_programs(n=args.n, level=args.level, max_depth=args.max_depth, seed=args.seed)

    for i, program in enumerate(programs, start=1):
        print(f"===== Program {i} =====")
        print(program)
        print()


if __name__ == "__main__":
    main()


