"""
AST Autoencoder for Program Synthesis

Combines Graph Neural Network encoder with grammar-constrained generation head
to create an autoencoder that can encode programs into latent representations
and decode them back to programs.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from models.graph_encoder import ProgramGraphEncoder
from models.generation_head import GrammarAwareGenerationHead
from dataset.grammar import get_cfg
from dataset.ast_converter import program_to_graph
from dataset.grammar import realize_program, parse_program_with_ast

class ASTAutoencoder(nn.Module):
    """
    Complete autoencoder for program synthesis.

    Architecture:
    1. Graph encoder: Converts AST graphs to latent embeddings
    2. Generation head: Decodes latent embeddings to grammar-constrained programs

    The model can be trained to reconstruct programs, learning useful
    representations for program synthesis tasks.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        encoder_layers: int = 3,
        encoder_dropout: float = 0.1,
        encoder_pooling: str = "mean",
        max_decode_steps: int = 100,
        **encoder_kwargs
    ):
        """
        Initialize the AST autoencoder.

        Args:
            hidden_dim: Hidden dimension for both encoder and decoder
            encoder_layers: Number of GCN layers in encoder
            encoder_dropout: Dropout rate for encoder
            encoder_pooling: Pooling method for graph encoder
            max_decode_steps: Maximum steps for program generation
            **encoder_kwargs: Additional arguments for ProgramGraphEncoder
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_decode_steps = max_decode_steps

        # Graph encoder
        self.encoder = ProgramGraphEncoder(
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            dropout=encoder_dropout,
            pooling=encoder_pooling,
            **encoder_kwargs
        )

        # Grammar-constrained generation head
        self.decoder = GrammarAwareGenerationHead(
            hidden_dim=hidden_dim,
            grammar=get_cfg()
        )

        # Bridge layer to convert graph embeddings to decoder format
        self.encoder_to_decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def encode(self, data: Batch, program_infos: Optional[List[Dict[str, Any]]] = None) -> torch.Tensor:
        """
        Encode graphs to latent representations.

        Args:
            data: Batch of PyTorch Geometric graphs
            program_infos: Optional program information for context

        Returns:
            Latent embeddings [batch_size, hidden_dim]
        """
        # Get graph embeddings
        graph_embeddings = self.encoder(data, program_infos)

        # Transform to decoder input format
        latent = self.encoder_to_decoder(graph_embeddings)

        return latent

    def decode(
        self,
        latent: torch.Tensor,
        max_steps: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Decode latent representations to programs.

        Args:
            latent: Latent embeddings [batch_size, hidden_dim]
            max_steps: Maximum generation steps (default: self.max_decode_steps)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter

        Returns:
            Dictionary containing generated programs and metadata
        """
        max_steps = max_steps or self.max_decode_steps
        batch_size = latent.shape[0]

        # Initialize generation state
        generated_programs = []
        generation_info = []

        result = self.decoder.generate_program(
            context_embeddings=latent,
            max_steps=max_steps
        )

        for tokens in result:
            try:
                program = realize_program(tokens)
                generated_programs.append(program)
                generation_info.append({
                    'success': parse_program_with_ast(program),
                    'steps': len(tokens),
                    'error': None,
                    'tokens': tokens
                })

            except Exception as e:
                # Handle generation failures gracefully
                generated_programs.append('')
                generation_info.append({
                    'success': False,
                    'steps': 0,
                    'error': str(e),
                    'tokens': []
                })

        return {
            'programs': generated_programs,
            'generation_info': generation_info,
            'latent': latent
        }

    def forward(
        self,
        data: Batch,
        program_infos: Optional[List[Dict[str, Any]]] = None,
        decode: bool = True,
        **decode_kwargs
    ) -> Dict[str, Any]:
        """
        Full forward pass: encode and optionally decode.

        Args:
            data: Batch of PyTorch Geometric graphs
            program_infos: Optional program information for context
            decode: Whether to perform decoding step
            **decode_kwargs: Arguments for decode method

        Returns:
            Dictionary containing latent embeddings and optionally decoded programs
        """
        # Encode
        latent = self.encode(data, program_infos)

        result = {'latent': latent}

        # Decode if requested
        if decode:
            decode_result = self.decode(latent, **decode_kwargs)
            result.update(decode_result)

        return result

    def reconstruct(self, program_codes: List[str], **decode_kwargs) -> Dict[str, Any]:
        """
        Reconstruct programs through encode-decode cycle.

        Args:
            program_codes: List of Python program strings
            **decode_kwargs: Arguments for decode method

        Returns:
            Dictionary containing original programs, reconstructed programs, and metrics
        """
        from torch_geometric.data import Batch

        # Convert programs to graphs
        graphs = []
        valid_indices = []

        for i, code in enumerate(program_codes):
            try:
                graph = program_to_graph(code)
                graphs.append(graph)
                valid_indices.append(i)
            except Exception as e:
                print(f"Warning: Failed to convert program {i}: {e}")

        if not graphs:
            return {
                'original_programs': program_codes,
                'reconstructed_programs': [''] * len(program_codes),
                'success_rate': 0.0,
                'valid_conversions': 0,
                'reconstruction_errors': ['No valid graphs'] * len(program_codes)
            }

        # Batch graphs
        try:
            batched_graphs = Batch.from_data_list(graphs)
        except Exception as e:
            return {
                'original_programs': program_codes,
                'reconstructed_programs': [''] * len(program_codes),
                'success_rate': 0.0,
                'valid_conversions': len(graphs),
                'reconstruction_errors': [f'Batching failed: {e}'] * len(program_codes)
            }

        # Forward pass
        with torch.no_grad():
            result = self.forward(batched_graphs, decode=True, **decode_kwargs)

        # Map results back to original indices
        reconstructed = [''] * len(program_codes)
        errors = ['Not processed'] * len(program_codes)

        for i, orig_idx in enumerate(valid_indices):
            if i < len(result['programs']):
                reconstructed[orig_idx] = result['programs'][i]
                info = result['generation_info'][i]
                if info['success']:
                    errors[orig_idx] = None
                else:
                    errors[orig_idx] = info.get('error', 'Generation failed')

        # Calculate metrics
        successful_reconstructions = sum(1 for err in errors if err is None)
        success_rate = successful_reconstructions / len(program_codes)

        return {
            'original_programs': program_codes,
            'reconstructed_programs': reconstructed,
            'success_rate': success_rate,
            'valid_conversions': len(graphs),
            'reconstruction_errors': errors,
            'generation_info': result.get('generation_info', []),
            'latent_embeddings': result.get('latent')
        }


class ASTAutoencoderTrainer:
    """
    Training utilities for the AST autoencoder.

    Provides methods for reconstruction loss, program similarity metrics,
    and training loop helpers.
    """

    def __init__(self, model: ASTAutoencoder, device: torch.device = None):
        """
        Initialize trainer.

        Args:
            model: ASTAutoencoder model to train
            device: Device to use for training
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def reconstruction_loss(
        self,
        original_programs: List[str],
        reconstructed_programs: List[str],
        latent: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate reconstruction loss.

        Args:
            original_programs: Original program strings
            reconstructed_programs: Reconstructed program strings
            latent: Latent embeddings

        Returns:
            Dictionary of loss components
        """
        batch_size = len(original_programs)
        device = latent.device

        # Program-level reconstruction loss (exact match)
        exact_matches = []
        for orig, recon in zip(original_programs, reconstructed_programs):
            exact_matches.append(1.0 if orig.strip() == recon.strip() else 0.0)

        exact_match_rate = torch.tensor(exact_matches, device=device).mean()

        # Semantic similarity loss (placeholder - could use AST comparison)
        # For now, use simple string similarity
        similarity_scores = []
        for orig, recon in zip(original_programs, reconstructed_programs):
            if not orig.strip() or not recon.strip():
                similarity_scores.append(0.0)
            else:
                # Simple token-based similarity
                orig_tokens = set(orig.split())
                recon_tokens = set(recon.split())
                if not orig_tokens:
                    similarity_scores.append(0.0)
                else:
                    jaccard = len(orig_tokens & recon_tokens) / len(orig_tokens | recon_tokens)
                    similarity_scores.append(jaccard)

        similarity_score = torch.tensor(similarity_scores, device=device).mean()

        # Regularization loss on latent space
        latent_reg = torch.mean(latent ** 2)

        # Combined loss
        reconstruction_loss = 1.0 - similarity_score
        total_loss = reconstruction_loss + 0.01 * latent_reg

        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'similarity_score': similarity_score,
            'exact_match_rate': exact_match_rate,
            'latent_regularization': latent_reg
        }


def test_autoencoder():
    """Test function for the AST autoencoder."""
    print("Testing ASTAutoencoder...")

    # Sample program codes
    test_programs = [
        "def add(a, b):\n    return a + b",
        "def max_of_list(lst):\n    result = lst[0]\n    for x in lst:\n        if x > result:\n            result = x\n    return result",
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
    ]

    # Create model
    model = ASTAutoencoder(hidden_dim=64, max_decode_steps=50)

    # Test reconstruction
    print(f"Testing reconstruction of {len(test_programs)} programs...")

    with torch.no_grad():
        result = model.reconstruct(test_programs, temperature=0.1)

    print(f"Success rate: {result['success_rate']:.2%}")
    print(f"Valid conversions: {result['valid_conversions']}")

    for i, (orig, recon, error) in enumerate(zip(
        result['original_programs'],
        result['reconstructed_programs'],
        result['reconstruction_errors']
    )):
        print(f"\nProgram {i}:")
        print(f"Original: {repr(orig)}")
        print(f"Reconstructed: {repr(recon)}")
        if error:
            print(f"Error: {error}")
        else:
            print("✅ Success!")

    print("✅ Test completed!")


if __name__ == "__main__":
    test_autoencoder()