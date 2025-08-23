#!/usr/bin/env python3
"""Debug script to analyze generation issues"""

import torch
from model.generation_head import GrammarAwareGenerationHead
from dataset.grammar import get_cfg, realize_program, parse_program_with_ast

def debug_generation():
    """Debug the generation process step by step"""
    print("🔍 Debugging Program Generation")
    print("=" * 40)
    
    grammar = get_cfg()
    gen_head = GrammarAwareGenerationHead(64, grammar)
    context_embeddings = torch.randn(1, 1, 64)
    
    # Generate with very detailed output
    torch.manual_seed(42)
    programs = gen_head.generate_program(context_embeddings, max_steps=100)
    tokens = programs[0]
    
    print(f"Generated {len(tokens)} tokens:")
    print(f"Tokens: {tokens}")
    print()
    
    # Try to realize it
    try:
        code = realize_program(tokens)
        print("Realized code:")
        print(code)
        print()
        
        # Check if parseable
        is_valid = parse_program_with_ast(code)
        print(f"Valid Python: {is_valid}")
        
        if not is_valid:
            print("Issues likely:")
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if line.strip():
                    print(f"Line {i}: '{line}'")
                    
    except Exception as e:
        print(f"Failed to realize: {e}")

if __name__ == "__main__":
    debug_generation()