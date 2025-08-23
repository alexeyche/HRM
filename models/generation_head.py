"""
Grammar-Aware Generation Head for Program Synthesis

This module implements a grammar-constrained generation head that produces
syntactically valid Python code by expanding NLTK CFG productions.
"""

from typing import Dict, List, Optional, Tuple, Union, Set
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk import CFG, Nonterminal
from dataset.grammar import get_cfg, get_token_patterns


class ProductionHead(nn.Module):
    """
    Core component that selects grammar production rules for non-terminals.
    
    Handles rule-level expansion by masking invalid productions and selecting
    from valid alternatives for the current non-terminal.
    """
    
    def __init__(self, hidden_dim: int, grammar: CFG):
        super().__init__()
        self.grammar = grammar
        self.hidden_dim = hidden_dim
        
        # Create mappings for productions
        self.production_to_idx = {}
        self.idx_to_production = {}
        self.nonterminal_to_productions = {}
        
        self._build_production_mappings()
        
        # Linear layer to project hidden state to production logits
        self.production_proj = nn.Linear(hidden_dim, len(self.production_to_idx))
    
    def _build_production_mappings(self):
        """Build mappings between productions and indices for efficient lookup."""
        productions = list(self.grammar.productions())
        
        # Create bidirectional mappings
        for i, prod in enumerate(productions):
            self.production_to_idx[prod] = i
            self.idx_to_production[i] = prod
            
            # Group productions by LHS non-terminal
            lhs = prod.lhs()
            if lhs not in self.nonterminal_to_productions:
                self.nonterminal_to_productions[lhs] = []
            self.nonterminal_to_productions[lhs].append(i)
    
    def forward(self, hidden_state: torch.Tensor, current_nonterminal: Optional[Nonterminal] = None) -> torch.Tensor:
        """
        Select a production rule for the current non-terminal.
        
        Args:
            hidden_state: Context embeddings from transformer (batch_size, hidden_dim)
            current_nonterminal: Current non-terminal to expand
            
        Returns:
            Masked logits over valid productions
        """
        # Get raw production logits
        production_logits = self.production_proj(hidden_state)
        
        # Apply masking if we have a specific non-terminal
        if current_nonterminal is not None:
            mask = self._create_production_mask(current_nonterminal, production_logits.device)
            production_logits = production_logits + mask
        
        return production_logits
    
    def _create_production_mask(self, nonterminal: Nonterminal, device: torch.device) -> torch.Tensor:
        """
        Create mask for valid productions of a given non-terminal.
        
        Args:
            nonterminal: Non-terminal to create mask for
            device: Device to create tensor on
            
        Returns:
            Mask tensor where valid productions have 0, invalid have -inf
        """
        mask = torch.full((len(self.production_to_idx),), float('-inf'), device=device)
        
        if nonterminal in self.nonterminal_to_productions:
            valid_indices = self.nonterminal_to_productions[nonterminal]
            mask[valid_indices] = 0.0
        
        return mask


class IdentifierHead(nn.Module):
    """
    Specialized head for handling identifier tokens.
    
    Supports both generation of new identifiers and copy mechanism
    to reuse existing identifiers in scope.
    """
    
    def __init__(self, hidden_dim: int, vocab_size: int = 26, max_identifiers: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size  # a-z for simple identifiers
        self.max_identifiers = max_identifiers
        
        # Generation head for new identifiers
        self.gen_proj = nn.Linear(hidden_dim, vocab_size)
        
        # Copy mechanism components
        self.copy_gate = nn.Linear(hidden_dim, 1)  # decide whether to copy or generate
        self.copy_attention = nn.Linear(hidden_dim, hidden_dim)  # attention over available identifiers
    
    def forward(self, hidden_state: torch.Tensor, context_identifiers: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Predict identifier - either generate new or copy existing.
        
        Args:
            hidden_state: Context embeddings (batch_size, hidden_dim)
            context_identifiers: List of identifiers currently in scope
            
        Returns:
            Dictionary with generation, copy gate, and copy attention logits
        """
        batch_size = hidden_state.size(0)
        
        # Generation logits for new identifiers
        gen_logits = self.gen_proj(hidden_state)  # (batch_size, vocab_size)
        
        # Copy gate - sigmoid to get probability of copying vs generating
        copy_gate_logits = self.copy_gate(hidden_state)  # (batch_size, 1)
        
        # Copy attention logits
        if context_identifiers and len(context_identifiers) > 0:
            # Create simple embeddings for identifiers (map a->0, b->1, etc.)
            identifier_embeddings = self._create_identifier_embeddings(context_identifiers, hidden_state.device)
            
            # Compute attention scores
            query = self.copy_attention(hidden_state)  # (batch_size, hidden_dim)
            
            # Simple dot-product attention
            copy_attention_logits = torch.matmul(query.unsqueeze(1), identifier_embeddings.T)  # (batch_size, 1, num_ids)
            copy_attention_logits = copy_attention_logits.squeeze(1)  # (batch_size, num_ids)
        else:
            # No identifiers to copy from
            copy_attention_logits = torch.zeros(batch_size, 0, device=hidden_state.device)
        
        return {
            "generation": gen_logits,
            "copy_gate": copy_gate_logits,
            "copy_attention": copy_attention_logits,
            "available_identifiers": context_identifiers or []
        }
    
    def _create_identifier_embeddings(self, identifiers: List[str], device: torch.device) -> torch.Tensor:
        """
        Create simple embeddings for available identifiers.
        
        Args:
            identifiers: List of identifier strings
            device: Device to create tensor on
            
        Returns:
            Embedding tensor (num_identifiers, hidden_dim)
        """
        # Simple mapping: convert single-letter identifiers to indices
        embeddings = []
        for identifier in identifiers:
            if len(identifier) == 1 and identifier.islower():
                # Map a->0, b->1, ..., z->25
                idx = ord(identifier) - ord('a')
                if 0 <= idx < self.vocab_size:
                    # Create one-hot embedding
                    embedding = torch.zeros(self.hidden_dim, device=device)
                    embedding[idx % self.hidden_dim] = 1.0
                    embeddings.append(embedding)
        
        if embeddings:
            return torch.stack(embeddings)
        else:
            # Return empty tensor if no valid identifiers
            return torch.zeros(0, self.hidden_dim, device=device)
    
    def sample_identifier(self, logits_dict: Dict[str, torch.Tensor], temperature: float = 1.0) -> List[str]:
        """
        Sample an identifier from the logits.
        
        Args:
            logits_dict: Output from forward pass
            temperature: Sampling temperature
            
        Returns:
            List of sampled identifier strings (one per batch item)
        """
        batch_size = logits_dict["generation"].size(0)
        results = []
        
        for i in range(batch_size):
            copy_gate_prob = torch.sigmoid(logits_dict["copy_gate"][i])
            
            # Decide whether to copy or generate
            if (copy_gate_prob > 0.5 and 
                len(logits_dict["available_identifiers"]) > 0 and 
                logits_dict["copy_attention"].size(1) > 0):
                
                # Copy from available identifiers
                copy_probs = F.softmax(logits_dict["copy_attention"][i] / temperature, dim=0)
                copy_idx = torch.multinomial(copy_probs, 1).item()
                results.append(logits_dict["available_identifiers"][copy_idx])
            else:
                # Generate new identifier
                gen_probs = F.softmax(logits_dict["generation"][i] / temperature, dim=0)
                gen_idx = torch.multinomial(gen_probs, 1).item()
                results.append(chr(ord('a') + gen_idx))
        
        return results


class LiteralHead(nn.Module):
    """
    Specialized head for handling literal values (numbers, strings, booleans).
    
    Uses factorized prediction: first predicts literal type, then literal value.
    """
    
    def __init__(self, hidden_dim: int, max_int: int = 20, str_vocab_size: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_int = max_int
        self.str_vocab_size = str_vocab_size
        
        # Literal type prediction (int, str, bool)
        self.type_proj = nn.Linear(hidden_dim, 3)
        
        # Value predictions for each type
        self.int_proj = nn.Linear(hidden_dim, max_int + 1)  # integers 0 to max_int
        self.bool_proj = nn.Linear(hidden_dim, 2)  # True/False
        self.str_proj = nn.Linear(hidden_dim, str_vocab_size)  # character vocabulary
        
        # String length prediction (for multi-character strings)
        self.str_len_proj = nn.Linear(hidden_dim, 4)  # lengths 0-3
    
    def forward(self, hidden_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict literal type and value.
        
        Args:
            hidden_state: Context embeddings (batch_size, hidden_dim)
            
        Returns:
            Dictionary with type and value predictions
        """
        # Predict literal type
        type_logits = self.type_proj(hidden_state)
        
        # Predict values for each type
        int_logits = self.int_proj(hidden_state)
        bool_logits = self.bool_proj(hidden_state)
        str_char_logits = self.str_proj(hidden_state)
        str_len_logits = self.str_len_proj(hidden_state)
        
        return {
            "type": type_logits,  # (batch_size, 3) - [int, str, bool]
            "int_value": int_logits,  # (batch_size, max_int+1)
            "bool_value": bool_logits,  # (batch_size, 2)
            "str_char": str_char_logits,  # (batch_size, str_vocab_size) 
            "str_length": str_len_logits  # (batch_size, 4)
        }
    
    def sample_literal(self, logits_dict: Dict[str, torch.Tensor], temperature: float = 1.0) -> List[str]:
        """
        Sample a literal value from the logits.
        
        Args:
            logits_dict: Output from forward pass
            temperature: Sampling temperature
            
        Returns:
            List of sampled literal strings (one per batch item)
        """
        batch_size = logits_dict["type"].size(0)
        results = []
        
        for i in range(batch_size):
            # Sample literal type
            type_probs = F.softmax(logits_dict["type"][i] / temperature, dim=0)
            type_idx = torch.multinomial(type_probs, 1).item()
            
            if type_idx == 0:  # integer
                int_probs = F.softmax(logits_dict["int_value"][i] / temperature, dim=0)
                int_val = torch.multinomial(int_probs, 1).item()
                results.append(str(int_val))
                
            elif type_idx == 1:  # string
                # Sample string length
                len_probs = F.softmax(logits_dict["str_length"][i] / temperature, dim=0)
                str_len = torch.multinomial(len_probs, 1).item()
                
                if str_len == 0:
                    # Empty string
                    results.append('""')
                else:
                    # Generate string characters
                    chars = []
                    for _ in range(str_len):
                        char_probs = F.softmax(logits_dict["str_char"][i] / temperature, dim=0)
                        char_idx = torch.multinomial(char_probs, 1).item()
                        # Map to printable characters (simplified)
                        if char_idx < 26:
                            chars.append(chr(ord('a') + char_idx))
                        elif char_idx < 52:
                            chars.append(chr(ord('A') + (char_idx - 26)))
                        elif char_idx < 62:
                            chars.append(str(char_idx - 52))
                        else:
                            chars.append('_')
                    
                    char_str = "".join(chars)
                    results.append(f'"{char_str}"')
                    
            else:  # boolean
                bool_probs = F.softmax(logits_dict["bool_value"][i] / temperature, dim=0)
                bool_val = torch.multinomial(bool_probs, 1).item()
                results.append("True" if bool_val == 1 else "False")
        
        return results
    
    def _get_literal_type_name(self, type_idx: int) -> str:
        """Get human readable name for literal type index."""
        type_names = ["int", "str", "bool"]
        return type_names[type_idx] if 0 <= type_idx < len(type_names) else "unknown"


class FunctionCallHead(nn.Module):
    """
    Specialized head for function call constructs.
    
    Predicts function name, argument count, and delegates argument
    expansion back to the main production head.
    """
    
    def __init__(self, hidden_dim: int, max_args: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_args = max_args
        
        # Function name prediction (built-ins + user-defined)
        self.func_proj = nn.Linear(hidden_dim, 20)  # approximate number of built-in functions
        
        # Argument count prediction
        self.arg_count_proj = nn.Linear(hidden_dim, max_args + 1)  # 0 to max_args
    
    def forward(self, hidden_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict function call components.
        
        Args:
            hidden_state: Context embeddings
            
        Returns:
            Dictionary with function and argument predictions
        """
        func_logits = self.func_proj(hidden_state)
        arg_count_logits = self.arg_count_proj(hidden_state)
        
        return {
            "function": func_logits,
            "arg_count": arg_count_logits
        }


class ControlFlowHead(nn.Module):
    """
    Specialized head for control flow constructs (if, while, for).
    
    Predicts control keywords and handles expansion of sub-blocks.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Control flow keyword prediction
        self.control_proj = nn.Linear(hidden_dim, 6)  # if, elif, else, while, for, break, continue
    
    def forward(self, hidden_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict control flow constructs.
        
        Args:
            hidden_state: Context embeddings
            
        Returns:
            Dictionary with control flow predictions
        """
        control_logits = self.control_proj(hidden_state)
        
        return {
            "control": control_logits
        }


class GrammarAwareGenerationHead(nn.Module):
    """
    Main grammar-aware generation head that orchestrates program synthesis.
    
    Integrates ProductionHead with specialized value heads to generate
    syntactically valid Python code through grammar-guided expansion.
    """
    
    def __init__(self, hidden_dim: int, grammar: Optional[CFG] = None):
        super().__init__()
        
        if grammar is None:
            grammar = get_cfg()
        
        self.grammar = grammar
        self.hidden_dim = hidden_dim
        
        # Load terminal token patterns from grammar
        self.token_patterns = get_token_patterns()
        self.terminal_to_token_map = self._build_terminal_to_token_map()
        
        # Core production head
        self.production_head = ProductionHead(hidden_dim, grammar)
        
        # Specialized value heads
        self.identifier_head = IdentifierHead(hidden_dim)
        self.literal_head = LiteralHead(hidden_dim)
        self.function_call_head = FunctionCallHead(hidden_dim)
        self.control_flow_head = ControlFlowHead(hidden_dim)
        
        # Expansion stack for managing non-terminals
        self.expansion_stack = []
    
    def forward(self, context_embeddings: torch.Tensor, 
                current_nonterminal: Optional[Nonterminal] = None) -> Dict[str, torch.Tensor]:
        """
        Generate next tokens or production rules based on current context.
        
        Args:
            context_embeddings: Hidden states from transformer backbone
            current_nonterminal: Current non-terminal at top of expansion stack
            
        Returns:
            Dictionary with predictions from relevant heads
        """
        batch_size = context_embeddings.size(0)
        
        # Get production predictions
        production_logits = self.production_head(context_embeddings, current_nonterminal)
        
        output = {
            "production": production_logits
        }
        
        # TODO: Add logic to route to specialized heads based on production selection
        # For now, return predictions from all heads
        output.update({
            "identifier": self.identifier_head(context_embeddings),
            "literal": self.literal_head(context_embeddings),
            "function_call": self.function_call_head(context_embeddings),
            "control_flow": self.control_flow_head(context_embeddings)
        })
        
        return output
    
    def expand_production(self, production_idx: int) -> List[Union[str, Nonterminal]]:
        """
        Expand a production rule and return resulting symbols.
        
        Args:
            production_idx: Index of production rule to expand
            
        Returns:
            List of terminals and non-terminals from expansion
        """
        if production_idx in self.production_head.idx_to_production:
            production = self.production_head.idx_to_production[production_idx]
            return list(production.rhs())
        else:
            return []
    
    def generate_program(self, context_embeddings: torch.Tensor, max_steps: int = 100) -> List[List[str]]:
        """
        Generate complete program by expanding grammar rules.
        
        Args:
            context_embeddings: Context from transformer backbone (batch_size, seq_len, hidden_dim)
            max_steps: Maximum expansion steps
            
        Returns:
            List of token lists, one per batch item
        """
        batch_size = context_embeddings.size(0)
        
        # Initialize expansion state for each batch item
        batch_results = []
        
        for batch_idx in range(batch_size):
            # Get hidden state for this batch item (use last position)
            hidden_state = context_embeddings[batch_idx:batch_idx+1, -1, :]
            
            # Initialize expansion stack with start symbol
            expansion_stack = [self.grammar.start()]
            terminals = []
            context_identifiers = []
            
            steps = 0
            while expansion_stack and steps < max_steps:
                steps += 1
                
                # Get current symbol to process
                current_symbol = expansion_stack.pop()
                
                # If it's a terminal, handle it appropriately
                if not isinstance(current_symbol, Nonterminal):
                    terminal_str = str(current_symbol)
                    
                    if terminal_str == "VARIABLE":
                        # Use identifier head
                        id_output = self.identifier_head(hidden_state, context_identifiers)
                        sampled_ids = self.identifier_head.sample_identifier(id_output)
                        terminal_token = sampled_ids[0]
                        terminals.append(terminal_token)
                        
                        # Add to context for future reference
                        if terminal_token not in context_identifiers:
                            context_identifiers.append(terminal_token)
                            
                    elif terminal_str in ["DIGIT", "STRING", "TRUE", "FALSE"]:
                        # Use literal head
                        lit_output = self.literal_head(hidden_state)
                        sampled_lits = self.literal_head.sample_literal(lit_output)
                        terminals.append(sampled_lits[0])
                        
                    else:
                        # Regular terminal - map to actual token
                        actual_token = self._map_terminal_to_token(terminal_str)
                        terminals.append(actual_token)
                    
                    continue
                
                # Get production predictions for non-terminal
                production_logits = self.production_head(hidden_state, current_symbol)
                
                # Sample a production
                production_probs = F.softmax(production_logits, dim=-1)
                production_idx = int(torch.multinomial(production_probs[0], 1).item())
                
                # Expand the production
                expansion = self.expand_production(production_idx)
                
                # Add all symbols from expansion to stack in reverse order
                # This ensures proper left-to-right processing
                for symbol in reversed(expansion):
                    expansion_stack.append(symbol)
            
            batch_results.append(terminals)
        
        return batch_results
    
    def _build_terminal_to_token_map(self) -> Dict[str, str]:
        """Build mapping from terminal symbols to tokens using grammar token patterns."""
        terminal_to_token = {}
        
        for terminal, token_list in self.token_patterns.items():
            if len(token_list) == 1:
                # Single token mapping (most common)
                terminal_to_token[terminal] = token_list[0]
            elif len(token_list) > 1:
                # Multiple options - use first one as default
                # For operators like ADDOP: ["+", "-"], we'll use "+"
                # For comparisons like BINARY_CMP: ["<", ">", ...], we'll use "<"
                terminal_to_token[terminal] = token_list[0]
        
        return terminal_to_token
    
    def _map_terminal_to_token(self, terminal: str) -> str:
        """Map grammar terminal symbols to actual code tokens using grammar patterns."""
        return self.terminal_to_token_map.get(terminal, terminal)