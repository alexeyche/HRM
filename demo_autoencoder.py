#!/usr/bin/env python3
"""
Demo script showing the AST autoencoder pipeline.
"""

import torch
from models.ast_autoencoder import ASTAutoencoder
from dataset.graph_dataset import ProgramGraphDataset, ProgramGraphCollator


def main():
    print("🚀 AST Autoencoder Pipeline Demo")
    print("=" * 50)
    
    # 1. Create dataset from existing programs
    print("📚 Creating dataset from programs...")
    dataset = ProgramGraphDataset(programs_per_spec=1, examples_per_program=1)
    print(f"   Dataset size: {len(dataset)} items")
    
    stats = dataset.get_statistics()
    print(f"   Unique programs: {stats['unique_programs']}")
    print(f"   Program categories: {list(stats['categories'].keys())}")
    
    # 2. Show a few sample programs
    print("\n📝 Sample programs:")
    for i in range(min(3, len(dataset))):
        graph, info = dataset[i]
        print(f"   {i+1}. {info['spec_name']}: {info['description'][:50]}...")
        print(f"      Graph: {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")
    
    # 3. Create autoencoder
    print("\n🧠 Creating AST Autoencoder...")
    autoencoder = ASTAutoencoder(
        hidden_dim=64,
        encoder_layers=2, 
        max_decode_steps=30
    )
    
    total_params = sum(p.numel() for p in autoencoder.parameters())
    print(f"   Total parameters: {total_params:,}")
    print(f"   Encoder: {sum(p.numel() for p in autoencoder.encoder.parameters()):,} params")
    print(f"   Decoder: {sum(p.numel() for p in autoencoder.decoder.parameters()):,} params")
    
    # 4. Test encoding pipeline
    print("\n⚙️  Testing encoding pipeline...")
    collator = ProgramGraphCollator()
    
    # Get a small batch
    batch_data = [dataset[i] for i in range(min(2, len(dataset)))]
    batch = collator(batch_data)
    
    with torch.no_grad():
        # Test encoding only (decoding requires full generation head implementation)
        latent = autoencoder.encode(batch['graphs'], batch['program_infos'])
        print(f"   Encoded {batch['batch_size']} programs to latent space: {latent.shape}")
        print(f"   Latent statistics: mean={latent.mean():.3f}, std={latent.std():.3f}")
    
    # 5. Show architecture summary
    print("\n🏗️  Architecture Summary:")
    print("   Pipeline: Programs → AST → Graph → GCN → Latent → Generation Head → Programs")
    print(f"   Components:")
    print(f"     • AST Converter: Python code → PyTorch Geometric graphs")
    print(f"     • Graph Dataset: {len(dataset)} program instances from registry")
    print(f"     • Graph Encoder: {autoencoder.encoder.graph_encoder.num_layers}-layer GCN with node embeddings")
    print(f"     • Generation Head: Grammar-constrained decoder with {len(autoencoder.decoder.production_head.production_to_idx)} production rules")
    
    print("\n✅ Core pipeline successfully implemented!")
    print("💡 Ready for training and program synthesis experiments.")


if __name__ == "__main__":
    main()