"""
Grammar-Aware Generation Head for Program Synthesis

This module implements a grammar-constrained generation head that produces
syntactically valid Python code by expanding NLTK CFG productions.
"""

from typing import Dict, List, Optional, Tuple, Union, Set, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk import CFG, Nonterminal
from dataset.grammar import get_cfg, get_token_patterns
from dataset.grammar import parse_tokens_to_productions



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

        # Simplified copy mechanism: single unified approach
        # Instead of separate copy gate + copy attention + generation, 
        # use a single classifier over all possible choices
        
        # The choices are: 26 generation options (a-z) + max_identifiers copy options
        total_choices = vocab_size + max_identifiers  # 26 chars + copy slots
        self.unified_classifier = nn.Linear(hidden_dim, total_choices)
        
        # Simple embedding for context identifiers (for attention if needed)
        self.identifier_embedding = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, hidden_state: torch.Tensor, context_identifiers: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Simplified identifier prediction using unified classifier.

        Args:
            hidden_state: Context embeddings (batch_size, hidden_dim)
            context_identifiers: List of identifiers currently in scope

        Returns:
            Dictionary with unified logits and choice mapping information
        """
        batch_size = hidden_state.size(0)
        device = hidden_state.device

        # Get unified logits for all choices
        all_logits = self.unified_classifier(hidden_state)  # (batch_size, total_choices)
        
        # Split into generation and copy parts
        generation_logits = all_logits[:, :self.vocab_size]  # First 26 are a-z
        copy_logits = all_logits[:, self.vocab_size:]  # Rest are copy slots
        
        # Mask unavailable copy slots
        num_available = len(context_identifiers) if context_identifiers else 0
        if num_available > 0:
            # Only keep logits for available identifiers
            available_copy_logits = copy_logits[:, :num_available]
            # Mask unused slots with -inf
            if num_available < self.max_identifiers:
                mask = torch.full((batch_size, self.max_identifiers - num_available), 
                                float('-inf'), device=device)
                copy_logits = torch.cat([available_copy_logits, mask], dim=1)
            else:
                copy_logits = available_copy_logits
        else:
            # No identifiers available - mask all copy logits
            copy_logits = torch.full((batch_size, self.max_identifiers), 
                                   float('-inf'), device=device)

        return {
            "generation": generation_logits,
            "copy": copy_logits, 
            "unified": all_logits,
            "available_identifiers": context_identifiers or [],
            "num_available": num_available
        }

    # Removed old embedding method - now using learnable embeddings

    def sample_identifier(self, logits_dict: Dict[str, torch.Tensor], temperature: float = 1.0) -> List[str]:
        """
        Sample an identifier using simplified unified approach.

        Args:
            logits_dict: Output from forward pass
            temperature: Sampling temperature

        Returns:
            List of sampled identifier strings (one per batch item)
        """
        batch_size = logits_dict["unified"].size(0)
        results = []
        available_identifiers = logits_dict["available_identifiers"]
        num_available = logits_dict["num_available"]

        for i in range(batch_size):
            # Sample from unified distribution
            unified_probs = F.softmax(logits_dict["unified"][i] / temperature, dim=0)
            choice_idx = torch.multinomial(unified_probs, 1).item()

            if choice_idx < self.vocab_size:
                # Generate new identifier (a-z)
                results.append(chr(ord('a') + choice_idx))
            else:
                # Copy existing identifier
                copy_idx = choice_idx - self.vocab_size
                if copy_idx < num_available:
                    results.append(available_identifiers[copy_idx])
                else:
                    # Fallback - generate 'a' if invalid copy index
                    results.append('a')

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

        # Route to specialized heads based on context and current non-terminal
        # Only call heads that are likely to be needed
        if current_nonterminal is not None:
            # Analyze which productions are possible for this non-terminal
            if current_nonterminal in self.production_head.nonterminal_to_productions:
                possible_productions = self.production_head.nonterminal_to_productions[current_nonterminal]

                # Check if any possible production needs specialized heads
                needs_identifier = False
                needs_literal = False
                needs_function = False
                needs_control = False

                for prod_idx in possible_productions:
                    requirements = self._analyze_production_requirements(prod_idx)
                    needs_identifier |= requirements["needs_identifier"]
                    needs_literal |= requirements["needs_literal"]
                    needs_function |= requirements["needs_function"]
                    needs_control |= requirements["needs_control"]

                # Only call heads that are actually needed
                if needs_identifier:
                    output["identifier"] = self.identifier_head(context_embeddings)
                if needs_literal:
                    output["literal"] = self.literal_head(context_embeddings)
                if needs_function:
                    output["function_call"] = self.function_call_head(context_embeddings)
                if needs_control:
                    output["control_flow"] = self.control_flow_head(context_embeddings)
        else:
            # If no specific non-terminal, call all heads (fallback for training)
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

    def _analyze_production_requirements(self, production_idx: int) -> Dict[str, bool]:
        """
        Analyze a production to determine which specialized heads are needed.

        Args:
            production_idx: Index of production rule to analyze

        Returns:
            Dictionary indicating which heads are required for this production
        """
        if production_idx not in self.production_head.idx_to_production:
            return {"needs_identifier": False, "needs_literal": False, "needs_function": False, "needs_control": False}

        production = self.production_head.idx_to_production[production_idx]
        rhs = production.rhs()

        needs_identifier = False
        needs_literal = False
        needs_function = False
        needs_control = False

        # Analyze the production's right-hand side for terminal requirements
        for symbol in rhs:
            symbol_str = str(symbol)

            # Check for identifier requirements
            if symbol_str == "VARIABLE":
                needs_identifier = True

            # Check for literal requirements
            elif symbol_str in ["DIGIT", "STRING", "TRUE", "FALSE"]:
                needs_literal = True

            # Check for function call requirements
            elif symbol_str in ["SUM", "LEN", "MIN", "MAX", "RANGE", "ABS", "SORTED", "SET", "STR", "INT", "LIST"]:
                needs_function = True

            # Check for control flow requirements
            elif symbol_str in ["IF", "ELIF", "ELSE", "WHILE", "FOR", "BREAK", "CONTINUE"]:
                needs_control = True

        return {
            "needs_identifier": needs_identifier,
            "needs_literal": needs_literal,
            "needs_function": needs_function,
            "needs_control": needs_control
        }

    def _apply_smart_production_selection(self, production_logits: torch.Tensor,
                                        current_nonterminal: Nonterminal,
                                        current_steps: int, max_steps: int) -> torch.Tensor:
        """
        Apply smart heuristics for production selection to avoid infinite recursion.

        Args:
            production_logits: Raw logits from production head
            current_nonterminal: The non-terminal being expanded
            current_steps: Current number of expansion steps
            max_steps: Maximum allowed steps

        Returns:
            Adjusted probability distribution for production selection
        """
        # Start with softmax of original logits
        probs = F.softmax(production_logits, dim=-1)

        # Get valid productions for this non-terminal
        if current_nonterminal not in self.production_head.nonterminal_to_productions:
            return probs

        valid_indices = self.production_head.nonterminal_to_productions[current_nonterminal]

        # Analyze each valid production
        terminal_production_indices = []
        recursive_production_indices = []

        for idx in valid_indices:
            if idx >= len(probs) or idx not in self.production_head.idx_to_production:
                continue

            production = self.production_head.idx_to_production[idx]
            rhs = production.rhs()

            # Check if production contains the same non-terminal (recursive)
            is_recursive = current_nonterminal in rhs

            # Check if production leads to terminals (no non-terminals in RHS)
            has_nonterminals = any(isinstance(symbol, Nonterminal) for symbol in rhs)

            if is_recursive:
                recursive_production_indices.append(idx)
            elif not has_nonterminals:
                terminal_production_indices.append(idx)

        # Apply bias based on current step count and available options
        bias_factor = min(current_steps / max_steps, 0.9)  # Increase bias as we approach max_steps

        # Create adjusted probabilities
        adjusted_probs = probs.clone()

        if len(terminal_production_indices) > 0:
            # Boost terminal productions when we're deep in expansion
            boost_factor = 1.0 + bias_factor * 3.0  # Up to 4x boost for terminal productions
            for idx in terminal_production_indices:
                if idx < len(adjusted_probs):
                    adjusted_probs[idx] *= boost_factor

        if len(recursive_production_indices) > 0:
            # Penalize recursive productions when we're deep in expansion
            penalty_factor = max(0.1, 1.0 - bias_factor * 2.0)  # Down to 0.1x for recursive productions
            for idx in recursive_production_indices:
                if idx < len(adjusted_probs):
                    adjusted_probs[idx] *= penalty_factor

        # Renormalize
        adjusted_probs = adjusted_probs / adjusted_probs.sum()

        return adjusted_probs

    def _process_terminal(self, terminal_symbol: Union[str, Nonterminal],
                         hidden_state: torch.Tensor,
                         context_identifiers: List[str]) -> Optional[str]:
        """
        Standardized terminal symbol processing using appropriate specialized heads.

        Args:
            terminal_symbol: Terminal symbol to process
            hidden_state: Current hidden state
            context_identifiers: Available identifiers for copy mechanism

        Returns:
            Generated token string, or None if processing failed
        """
        terminal_str = str(terminal_symbol)

        try:
            # Route to appropriate head based on terminal type
            if terminal_str == "VARIABLE":
                # Use identifier head with copy mechanism
                id_output = self.identifier_head(hidden_state, context_identifiers)
                sampled_ids = self.identifier_head.sample_identifier(id_output)
                return sampled_ids[0] if sampled_ids else "a"  # Fallback to 'a'

            elif terminal_str in ["DIGIT", "STRING", "TRUE", "FALSE"]:
                # Use literal head with type-aware generation
                lit_output = self.literal_head(hidden_state)

                # Override the type selection based on the specific terminal requested
                if terminal_str == "DIGIT":
                    # Force integer type
                    lit_output['type'] = torch.tensor([[10.0, -10.0, -10.0]] * hidden_state.size(0))
                elif terminal_str == "STRING":
                    # Force string type
                    lit_output['type'] = torch.tensor([[-10.0, 10.0, -10.0]] * hidden_state.size(0))
                elif terminal_str in ["TRUE", "FALSE"]:
                    # Force boolean type
                    lit_output['type'] = torch.tensor([[-10.0, -10.0, 10.0]] * hidden_state.size(0))

                sampled_lits = self.literal_head.sample_literal(lit_output)
                return sampled_lits[0] if sampled_lits else "0"  # Fallback to '0'

            elif terminal_str in ["SUM", "LEN", "MIN", "MAX", "RANGE", "ABS", "SORTED", "SET", "STR", "INT", "LIST"]:
                # Use function call head for built-in functions
                # For now, just use the mapped token - could be enhanced later
                return self._map_terminal_to_token(terminal_str)

            elif terminal_str in ["IF", "ELIF", "ELSE", "WHILE", "FOR", "BREAK", "CONTINUE"]:
                # Use control flow head for control keywords
                # For now, just use the mapped token - could be enhanced later
                return self._map_terminal_to_token(terminal_str)

            else:
                # Regular terminal - use token pattern mapping
                return self._map_terminal_to_token(terminal_str)

        except Exception:
            # Fallback to token mapping if specialized head fails
            return self._map_terminal_to_token(terminal_str)

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
            # Continue until expansion stack is empty OR we hit max_steps
            # Priority: Complete program generation over step limit
            while expansion_stack and steps < max_steps:
                steps += 1

                # Get current symbol to process
                current_symbol = expansion_stack.pop()

                # If it's a terminal, handle it using standardized processing
                if not isinstance(current_symbol, Nonterminal):
                    terminal_token = self._process_terminal(current_symbol, hidden_state, context_identifiers)
                    if terminal_token is not None:
                        terminals.append(terminal_token)

                        # Update context for identifiers
                        if str(current_symbol) == "VARIABLE" and terminal_token not in context_identifiers:
                            context_identifiers.append(terminal_token)

                    continue

                # Handle non-terminal expansion
                # Get production predictions for non-terminal
                production_logits = self.production_head(hidden_state, current_symbol)

                # Apply smart production selection to avoid infinite recursion
                production_probs = self._apply_smart_production_selection(
                    production_logits[0], current_symbol, steps, max_steps
                )
                production_idx = int(torch.multinomial(production_probs, 1).item())

                # Validate that this production is actually for this non-terminal
                if production_idx not in self.production_head.idx_to_production:
                    # Fallback: find any valid production for this non-terminal
                    if current_symbol in self.production_head.nonterminal_to_productions:
                        valid_productions = self.production_head.nonterminal_to_productions[current_symbol]
                        production_idx = valid_productions[0]  # Use first valid production
                    else:
                        # Skip this non-terminal if no valid productions (shouldn't happen)
                        continue

                # Double-check the production is for the current non-terminal
                selected_production = self.production_head.idx_to_production[production_idx]
                if selected_production.lhs() != current_symbol:
                    # Find correct production for this non-terminal
                    if current_symbol in self.production_head.nonterminal_to_productions:
                        valid_productions = self.production_head.nonterminal_to_productions[current_symbol]
                        production_idx = valid_productions[0]  # Use first valid production

                # Expand the production
                expansion = self.expand_production(production_idx)

                # Add all symbols from expansion to stack in reverse order
                # This ensures proper left-to-right processing
                for symbol in reversed(expansion):
                    expansion_stack.append(symbol)

            # If we still have non-terminals in the stack but hit max_steps,
            # try to complete the program by using epsilon productions or minimal expansions
            if expansion_stack and steps >= max_steps:
                # Try to complete with minimal expansions for remaining non-terminals
                remaining_steps = min(20, len(expansion_stack))  # Give a few more steps to complete
                completion_steps = 0

                while expansion_stack and completion_steps < remaining_steps:
                    completion_steps += 1
                    current_symbol = expansion_stack.pop()

                    if not isinstance(current_symbol, Nonterminal):
                        # It's a terminal - use standardized processing
                        terminal_token = self._process_terminal(current_symbol, hidden_state, context_identifiers)
                        if terminal_token is not None:
                            terminals.append(terminal_token)

                            # Update context for identifiers
                            if str(current_symbol) == "VARIABLE" and terminal_token not in context_identifiers:
                                context_identifiers.append(terminal_token)
                        continue

                    # Try to find a minimal expansion for this non-terminal
                    if current_symbol in self.production_head.nonterminal_to_productions:
                        valid_productions = self.production_head.nonterminal_to_productions[current_symbol]
                        # Use first production (often the simplest)
                        production_idx = valid_productions[0]
                        expansion = self.expand_production(production_idx)

                        # Add symbols in reverse order
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

    def compute_sequence_loss_single(
        self,
        hidden_state: torch.Tensor,
        tokens: List[str],
        temperature: float = 1.0
    ) -> Tuple[float, Tuple[int, int, int, int], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        device = hidden_state.device
        if not tokens:
            return (
                0.0,
                (0, 0, 0, 0),  # (total_steps, production_steps, identifier_steps, literal_steps)
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
                {}  # Empty debug info
            )

        production_loss = torch.tensor(0.0, device=device)
        identifier_loss = torch.tensor(0.0, device=device)
        literal_loss = torch.tensor(0.0, device=device)

        # try:
        # Parse tokens to grammar productions and required terminals
        production_sequence, terminal_requirements = parse_tokens_to_productions(tokens, self.grammar)

        step_loss = torch.tensor(0.0, device=device)
        production_steps = 0
        identifier_steps = 0
        literal_steps = 0
        
        # Debug info tracking
        debug_info = {
            "copy_decisions": [],  # List of (should_copy, target_value, context_size)
            "copy_gate_losses": [],  # List of copy gate loss values
            "copy_attention_losses": [],  # List of copy attention loss values
            "generation_losses": [],  # List of generation loss values
            "identifiers_processed": 0,
            "copy_attempts": 0,
            "generation_attempts": 0,
        }

        # Compute loss for each production decision
        for prod_idx, (nonterminal, target_prod_idx) in enumerate(production_sequence):
            # Get production logits for current nonterminal
            prod_logits = self.production_head(hidden_state, nonterminal)

            # Apply temperature
            prod_logits = prod_logits / temperature

            # Cross-entropy loss for production selection
            target_tensor = torch.tensor([target_prod_idx], device=device)
            prod_loss = F.cross_entropy(prod_logits, target_tensor)

            step_loss = torch.add(step_loss, prod_loss)
            production_loss = torch.add(production_loss, prod_loss)
            production_steps += 1

        # Compute loss for terminal value predictions using proper context from parsing
        for terminal_requirement in terminal_requirements:
            # Handle both old format (terminal_type, target_value) and new format (terminal_type, target_value, context)
            if len(terminal_requirement) == 3:
                terminal_type, target_value, context_identifiers = terminal_requirement
                # Use the context from parsing, or empty list if None
                context_identifiers = context_identifiers or []
            else:
                # Fallback to old format
                terminal_type, target_value = terminal_requirement
                context_identifiers = []
            if terminal_type == "VARIABLE":
                # Identifier loss with proper copy mechanism training
                id_output = self.identifier_head(hidden_state, context_identifiers)

                if target_value and len(target_value) == 1 and target_value.islower():
                    # Simplified unified approach - single loss calculation
                    should_copy = target_value in context_identifiers
                    debug_info["identifiers_processed"] += 1
                    debug_info["copy_decisions"].append((should_copy, target_value, len(context_identifiers)))

                    # Calculate target index in unified classifier
                    if should_copy and len(context_identifiers) > 0:
                        # Target is a copy operation
                        try:
                            copy_target_idx = context_identifiers.index(target_value)
                            # Index in unified classifier = vocab_size + copy_index
                            unified_target_idx = self.identifier_head.vocab_size + copy_target_idx
                            debug_info["copy_attempts"] += 1
                        except (ValueError, IndexError):
                            # Fallback to generation if target not found in context
                            target_char_idx = ord(target_value.lower()) - ord('a')
                            unified_target_idx = target_char_idx
                            debug_info["generation_attempts"] += 1
                    else:
                        # Target is a generation operation (a-z)
                        target_char_idx = ord(target_value.lower()) - ord('a')
                        if 0 <= target_char_idx < 26:
                            unified_target_idx = target_char_idx
                            debug_info["generation_attempts"] += 1
                        else:
                            # Invalid character, skip
                            continue

                    # Single unified loss calculation
                    target_tensor = torch.tensor([unified_target_idx], device=device)
                    unified_loss = F.cross_entropy(
                        id_output["unified"] / temperature, target_tensor
                    )
                    
                    # Record loss for debugging
                    if should_copy:
                        debug_info["copy_attention_losses"].append(unified_loss.item())
                    else:
                        debug_info["generation_losses"].append(unified_loss.item())
                    
                    step_loss = torch.add(step_loss, unified_loss)
                    identifier_loss = torch.add(identifier_loss, unified_loss)
                    identifier_steps += 1

            elif terminal_type in ["DIGIT", "STRING", "TRUE", "FALSE"]:
                # Literal loss
                lit_output = self.literal_head(hidden_state)

                # Type loss
                if terminal_type == "DIGIT":
                    target_type = torch.tensor([0], device=device)  # int type
                    type_loss = F.cross_entropy(lit_output["type"] / temperature, target_type)
                    step_loss = torch.add(step_loss, type_loss)
                    literal_loss = torch.add(literal_loss, type_loss)
                    literal_steps += 1

                    # Value loss for integers
                    try:
                        int_val = int(target_value)
                        if 0 <= int_val <= 20:  # Within our int range
                            target_int = torch.tensor([int_val], device=device)
                            int_loss = F.cross_entropy(lit_output["int_value"] / temperature, target_int)
                            step_loss = torch.add(step_loss, int_loss)
                            literal_loss = torch.add(literal_loss, int_loss)
                            literal_steps += 1
                    except ValueError:
                        pass

                elif terminal_type == "STRING":
                    target_type = torch.tensor([1], device=device)  # string type
                    type_loss = F.cross_entropy(lit_output["type"] / temperature, target_type)
                    step_loss = torch.add(step_loss, type_loss)
                    literal_loss = torch.add(literal_loss, type_loss)
                    literal_steps += 1

                elif terminal_type in ["TRUE", "FALSE"]:
                    target_type = torch.tensor([2], device=device)  # bool type
                    type_loss = F.cross_entropy(lit_output["type"] / temperature, target_type)
                    step_loss = torch.add(step_loss, type_loss)
                    literal_loss = torch.add(literal_loss, type_loss)
                    literal_steps += 1

                    # Value loss for booleans
                    bool_val = 1 if target_value == "True" else 0
                    target_bool = torch.tensor([bool_val], device=device)
                    bool_loss = F.cross_entropy(lit_output["bool_value"] / temperature, target_bool)
                    step_loss = torch.add(step_loss, bool_loss)
                    literal_loss = torch.add(literal_loss, bool_loss)
                    literal_steps += 1

        total_steps = production_steps + identifier_steps + literal_steps
        loss = 0.0
        if isinstance(step_loss, torch.Tensor):
            loss = step_loss.item() if total_steps > 0 else 0.0
        else:
            loss = float(step_loss) if total_steps > 0 else 0.0
        return loss, (total_steps, production_steps, identifier_steps, literal_steps), production_loss, identifier_loss, literal_loss, debug_info

    def compute_sequence_loss(
        self,
        context_embeddings: torch.Tensor,
        target_tokens: List[List[str]],
        temperature: float = 1.0,
        use_batch: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute cross-entropy loss for target token sequences using grammar-constrained generation.

        Args:
            context_embeddings: Context from transformer backbone (batch_size, seq_len, hidden_dim)
            target_tokens: Target token sequences for each batch item
            temperature: Temperature for softmax computations

        Returns:
            Dictionary containing loss components and metrics
        """
        batch_size = len(target_tokens)
        device = context_embeddings.device

        total_production_loss = torch.tensor(0.0, device=device)
        total_identifier_loss = torch.tensor(0.0, device=device)
        total_literal_loss = torch.tensor(0.0, device=device)
        total_steps = 0
        production_step_count = 0
        identifier_step_count = 0
        literal_step_count = 0
        
        # Aggregate debug info across batch
        batch_debug_info = {
            "total_identifiers_processed": 0,
            "total_copy_attempts": 0,
            "total_generation_attempts": 0,
            "copy_decisions_summary": {"copy": 0, "generate": 0},
            "avg_copy_gate_loss": 0.0,
            "avg_copy_attention_loss": 0.0, 
            "avg_generation_loss": 0.0,
            "context_size_distribution": [],
        }
        all_debug_infos = []  # Collect all debug info from samples

        if use_batch:
            raise NotImplementedError("Batch loss computation is not implemented yet")

        else:
            batch_losses = []

            for batch_idx in range(batch_size):
                hidden_state = context_embeddings[batch_idx:batch_idx+1, -1, :]
                tokens = target_tokens[batch_idx]

                loss, step_counts, production_loss, identifier_loss, literal_loss, single_debug_info = self.compute_sequence_loss_single(
                    hidden_state,
                    tokens,
                    temperature
                )
                batch_losses.append(loss)
                total_production_loss = torch.add(total_production_loss, production_loss)
                total_identifier_loss = torch.add(total_identifier_loss, identifier_loss)
                total_literal_loss = torch.add(total_literal_loss, literal_loss)
                total_steps += step_counts[0]  # Total steps
                production_step_count += step_counts[1]  # Production steps
                identifier_step_count += step_counts[2]  # Identifier steps  
                literal_step_count += step_counts[3]  # Literal steps
                
                # Collect all debug info
                all_debug_infos.append(single_debug_info)
                
                # Aggregate debug info
                batch_debug_info["total_identifiers_processed"] += single_debug_info["identifiers_processed"]
                batch_debug_info["total_copy_attempts"] += single_debug_info["copy_attempts"] 
                batch_debug_info["total_generation_attempts"] += single_debug_info["generation_attempts"]
                
                # Count copy vs generate decisions
                for should_copy, _, context_size in single_debug_info["copy_decisions"]:
                    if should_copy:
                        batch_debug_info["copy_decisions_summary"]["copy"] += 1
                    else:
                        batch_debug_info["copy_decisions_summary"]["generate"] += 1
                    batch_debug_info["context_size_distribution"].append(context_size)

        # Fixed averaging: use proper step counts for each loss type
        if production_step_count > 0:
            avg_production_loss = total_production_loss / production_step_count
        else:
            avg_production_loss = torch.tensor(0.0, device=device)
            
        if identifier_step_count > 0:
            avg_identifier_loss = total_identifier_loss / identifier_step_count
        else:
            avg_identifier_loss = torch.tensor(0.0, device=device)
            
        if literal_step_count > 0:
            avg_literal_loss = total_literal_loss / literal_step_count
        else:
            avg_literal_loss = torch.tensor(0.0, device=device)

        # Loss balancing to prevent any component from dominating
        # Apply adaptive weighting based on relative magnitudes
        loss_weights = [1.0, 1.0, 1.0]  # Default equal weights
        
        # Calculate adaptive weights based on current loss magnitudes
        # This helps prevent any one loss from dominating training
        if production_step_count > 0 and identifier_step_count > 0:
            # Production vs Identifier balancing
            prod_magnitude = avg_production_loss.item()
            id_magnitude = avg_identifier_loss.item()
            
            # If identifier loss is much smaller, increase its weight
            if prod_magnitude > 0 and id_magnitude > 0:
                ratio = prod_magnitude / id_magnitude
                if ratio > 2.0:  # Production loss is much larger
                    loss_weights[1] = min(3.0, ratio / 2.0)  # Boost identifier loss weight
                elif ratio < 0.5:  # Identifier loss is much larger
                    loss_weights[0] = min(3.0, (1.0 / ratio) / 2.0)  # Boost production loss weight
        
        # Apply gradient clipping per component to prevent instability
        max_component_loss = 5.0  # Maximum allowed loss for any component
        clipped_production_loss = torch.clamp(avg_production_loss, max=max_component_loss)
        clipped_identifier_loss = torch.clamp(avg_identifier_loss, max=max_component_loss)
        clipped_literal_loss = torch.clamp(avg_literal_loss, max=max_component_loss)
        
        # Weighted and clipped combined loss
        total_loss = (loss_weights[0] * clipped_production_loss + 
                     loss_weights[1] * clipped_identifier_loss + 
                     loss_weights[2] * clipped_literal_loss)
        
        # Calculate average debug losses for reporting
        all_copy_gate_losses = []
        all_copy_attention_losses = []
        all_generation_losses = []
        
        for single_debug in all_debug_infos:  # Use all collected debug info
            all_copy_gate_losses.extend(single_debug.get("copy_gate_losses", []))
            all_copy_attention_losses.extend(single_debug.get("copy_attention_losses", []))
            all_generation_losses.extend(single_debug.get("generation_losses", []))
        
        batch_debug_info["avg_copy_gate_loss"] = sum(all_copy_gate_losses) / max(1, len(all_copy_gate_losses))
        batch_debug_info["avg_copy_attention_loss"] = sum(all_copy_attention_losses) / max(1, len(all_copy_attention_losses))
        batch_debug_info["avg_generation_loss"] = sum(all_generation_losses) / max(1, len(all_generation_losses))

        return {
            "total_loss": total_loss,
            "production_loss": avg_production_loss,
            "identifier_loss": avg_identifier_loss,
            "literal_loss": avg_literal_loss,
            "clipped_production_loss": clipped_production_loss,
            "clipped_identifier_loss": clipped_identifier_loss,
            "clipped_literal_loss": clipped_literal_loss,
            "production_weight": torch.tensor(loss_weights[0], device=device),
            "identifier_weight": torch.tensor(loss_weights[1], device=device),
            "literal_weight": torch.tensor(loss_weights[2], device=device),
            "batch_losses": torch.tensor(batch_losses, device=device),
            "total_steps": torch.tensor(total_steps, device=device),
            "production_steps": torch.tensor(production_step_count, device=device),
            "identifier_steps": torch.tensor(identifier_step_count, device=device),
            "literal_steps": torch.tensor(literal_step_count, device=device),
            "avg_loss_per_step": total_loss / max(1, total_steps) if total_steps > 0 else torch.tensor(0.0, device=device),
            "debug_info": batch_debug_info
        }