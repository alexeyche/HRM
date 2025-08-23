#!/usr/bin/env python3
"""Test script to find valid generated programs"""

import torch
from model.generation_head import GrammarAwareGenerationHead
from dataset.grammar import get_cfg, realize_program, parse_program_with_ast

def test_valid_generation():
    """Generate multiple programs and show the valid ones"""
    print("🎯 Testing for Valid Generated Programs")
    print("=" * 50)
    
    grammar = get_cfg()
    gen_head = GrammarAwareGenerationHead(64, grammar)
    
    valid_programs = []
    total_tries = 20
    
    for i in range(total_tries):
        # Use different seeds and contexts
        torch.manual_seed(i * 10)
        context_embeddings = torch.randn(1, 1, 64)
        
        # Try different step counts
        for max_steps in [20, 40, 80]:
            try:
                programs = gen_head.generate_program(context_embeddings, max_steps=max_steps)
                tokens = programs[0]
                
                code = realize_program(tokens)
                
                if parse_program_with_ast(code):
                    valid_programs.append({
                        'seed': i * 10,
                        'max_steps': max_steps,
                        'tokens': len(tokens),
                        'code': code
                    })
                    print(f"✅ Valid program {len(valid_programs)} (seed={i*10}, steps={max_steps}):")
                    print(code)
                    print("-" * 30)
                    
                    if len(valid_programs) >= 5:  # Stop after finding 5 valid programs
                        break
                        
            except Exception as e:
                continue
                
        if len(valid_programs) >= 5:
            break
    
    print(f"\n📊 Results: Found {len(valid_programs)}/{total_tries*3} valid programs")
    
    if valid_programs:
        print("\n🎉 Valid Program Examples:")
        for prog in valid_programs[:3]:  # Show first 3
            print(f"Seed {prog['seed']}, {prog['max_steps']} steps, {prog['tokens']} tokens:")
            print(prog['code'])
            print("=" * 40)
    
    return len(valid_programs)

if __name__ == "__main__":
    valid_count = test_valid_generation()
    print(f"\nGenerated {valid_count} valid programs! 🎉")