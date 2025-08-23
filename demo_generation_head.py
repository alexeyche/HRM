#!/usr/bin/env python3
"""
Demo script for the Grammar-Aware Generation Head

This script demonstrates the generation head functionality by generating
sample program tokens using the NLTK grammar.
"""

import torch
from model.generation_head import GrammarAwareGenerationHead
from dataset.grammar import get_cfg, realize_program

def demo_generation_head():
    """Demonstrate the generation head functionality."""
    print("🔧 Grammar-Aware Generation Head Demo")
    print("=" * 50)
    
    # Load the grammar
    grammar = get_cfg()
    print(f"📝 Loaded grammar with {len(list(grammar.productions()))} productions")
    
    # Initialize generation head
    hidden_dim = 128
    gen_head = GrammarAwareGenerationHead(hidden_dim, grammar)
    print(f"🧠 Initialized generation head with hidden_dim={hidden_dim}")
    
    # Create dummy context embeddings
    batch_size = 3
    seq_len = 1
    context_embeddings = torch.randn(batch_size, seq_len, hidden_dim)
    print(f"📊 Generated context embeddings: {context_embeddings.shape}")
    
    print("\n🎯 Production Head Test:")
    print("-" * 30)
    
    # Test production head with start symbol
    start_nt = grammar.start()
    hidden_state = context_embeddings[0:1, -1, :]
    production_logits = gen_head.production_head(hidden_state, start_nt)
    print(f"Start symbol: {start_nt}")
    print(f"Production logits shape: {production_logits.shape}")
    
    # Show some valid productions for start symbol
    valid_productions = gen_head.production_head.nonterminal_to_productions.get(start_nt, [])
    print(f"Number of valid productions for {start_nt}: {len(valid_productions)}")
    
    print("\n🎯 Identifier Head Test:")
    print("-" * 30)
    
    # Test identifier head
    context_ids = ['a', 'x', 'result']
    id_output = gen_head.identifier_head(hidden_state, context_ids)
    print(f"Available identifiers: {context_ids}")
    print(f"Generation logits shape: {id_output['generation'].shape}")
    print(f"Copy attention shape: {id_output['copy_attention'].shape}")
    
    # Sample an identifier
    sampled_ids = gen_head.identifier_head.sample_identifier(id_output)
    print(f"Sampled identifier: {sampled_ids[0]}")
    
    print("\n🎯 Literal Head Test:")
    print("-" * 30)
    
    # Test literal head
    lit_output = gen_head.literal_head(hidden_state)
    print(f"Literal type logits shape: {lit_output['type'].shape}")
    print(f"Int value logits shape: {lit_output['int_value'].shape}")
    
    # Sample a literal
    sampled_lits = gen_head.literal_head.sample_literal(lit_output)
    print(f"Sampled literal: {sampled_lits[0]}")
    
    print("\n🎯 Full Program Generation Test:")
    print("-" * 40)
    
    # Generate programs
    programs = gen_head.generate_program(context_embeddings, max_steps=20)
    
    for i, program_tokens in enumerate(programs):
        print(f"\n📝 Program {i+1}:")
        print(f"Tokens: {program_tokens[:10]}...")  # Show first 10 tokens
        
        # Try to realize the program
        try:
            program_code = realize_program(program_tokens)
            print("Generated code:")
            print(program_code[:200] + "..." if len(program_code) > 200 else program_code)
        except Exception as e:
            print(f"Could not realize program: {e}")
    
    print("\n✅ Demo completed successfully!")

if __name__ == "__main__":
    demo_generation_head()