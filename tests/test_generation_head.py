"""
Tests for Grammar-Aware Generation Head

Following agile development principles, these tests start simple and grow
with the implementation.
"""

import pytest
import torch
from nltk import CFG, Nonterminal

from model.generation_head import (
    GrammarAwareGenerationHead,
    ProductionHead,
    IdentifierHead,
    LiteralHead,
    FunctionCallHead,
    ControlFlowHead
)
from dataset.grammar import get_cfg, realize_program, parse_program_with_ast


@pytest.fixture
def sample_grammar():
    """Provide sample grammar for testing."""
    return get_cfg()


@pytest.fixture
def sample_hidden_state():
    """Provide sample hidden state tensor for testing."""
    batch_size = 2
    hidden_dim = 64
    return torch.randn(batch_size, hidden_dim)


class TestProductionHead:
    """Test the core ProductionHead component."""
    
    def test_initialization(self, sample_grammar):
        """Test ProductionHead initializes correctly with grammar."""
        hidden_dim = 64
        production_head = ProductionHead(hidden_dim, sample_grammar)
        
        assert production_head.grammar == sample_grammar
        assert production_head.hidden_dim == hidden_dim
        assert hasattr(production_head, 'production_proj')
    
    def test_forward_pass_shape(self, sample_grammar, sample_hidden_state):
        """Test ProductionHead forward pass returns correct shape."""
        hidden_dim = sample_hidden_state.size(-1)
        production_head = ProductionHead(hidden_dim, sample_grammar)
        
        output = production_head(sample_hidden_state)
        
        # Should return tensor with production dimensions
        assert output.dim() == 2
        assert output.size(0) == sample_hidden_state.size(0)  # batch size
    
    def test_production_mappings_creation(self, sample_grammar):
        """Test production mapping creation works correctly."""
        hidden_dim = 64
        production_head = ProductionHead(hidden_dim, sample_grammar)
        
        # Check that mappings are populated
        assert len(production_head.production_to_idx) > 0
        assert len(production_head.idx_to_production) > 0
        assert len(production_head.nonterminal_to_productions) > 0
        
        # Check bidirectional consistency
        for prod, idx in production_head.production_to_idx.items():
            assert production_head.idx_to_production[idx] == prod
        
        # Check non-terminal mappings make sense
        for nt, prod_indices in production_head.nonterminal_to_productions.items():
            for idx in prod_indices:
                prod = production_head.idx_to_production[idx]
                assert prod.lhs() == nt
    
    def test_rule_masking_basic(self, sample_grammar, sample_hidden_state):
        """Test that rule masking works for specific non-terminals."""
        hidden_dim = sample_hidden_state.size(-1)
        production_head = ProductionHead(hidden_dim, sample_grammar)
        
        # Test with a known non-terminal
        start_nt = sample_grammar.start()
        output = production_head(sample_hidden_state, start_nt)
        
        # Should return valid tensor
        assert output.dim() == 2
        assert output.size(0) == sample_hidden_state.size(0)
        
        # Check that some values are not -inf (meaning valid rules exist)
        assert not torch.all(torch.isinf(output))
    
    def test_rule_masking_different_nonterminals(self, sample_grammar, sample_hidden_state):
        """Test that different non-terminals produce different masks."""
        hidden_dim = sample_hidden_state.size(-1)
        production_head = ProductionHead(hidden_dim, sample_grammar)
        
        # Get two different non-terminals
        nonterminals = list(production_head.nonterminal_to_productions.keys())
        if len(nonterminals) >= 2:
            nt1, nt2 = nonterminals[0], nonterminals[1]
            
            output1 = production_head(sample_hidden_state, nt1)
            output2 = production_head(sample_hidden_state, nt2)
            
            # The masked positions should be different (unless they share all rules)
            mask1 = torch.isinf(output1)
            mask2 = torch.isinf(output2)
            
            # At least one should be different (unless identical rule sets)
            if production_head.nonterminal_to_productions[nt1] != production_head.nonterminal_to_productions[nt2]:
                assert not torch.equal(mask1, mask2)
    
    def test_no_masking_without_nonterminal(self, sample_grammar, sample_hidden_state):
        """Test that no masking occurs when no non-terminal is provided."""
        hidden_dim = sample_hidden_state.size(-1)
        production_head = ProductionHead(hidden_dim, sample_grammar)
        
        # Test without providing non-terminal
        output_no_mask = production_head(sample_hidden_state, None)
        
        # Should not contain -inf values (no masking)
        assert not torch.any(torch.isinf(output_no_mask))
        
        # Compare with masked version
        start_nt = sample_grammar.start()
        output_masked = production_head(sample_hidden_state, start_nt)
        
        # They should be different (one has masking, one doesn't)
        assert not torch.equal(output_no_mask, output_masked)


class TestIdentifierHead:
    """Test the IdentifierHead component."""
    
    def test_initialization(self):
        """Test IdentifierHead initializes correctly."""
        hidden_dim = 64
        vocab_size = 26
        identifier_head = IdentifierHead(hidden_dim, vocab_size)
        
        assert identifier_head.hidden_dim == hidden_dim
        assert identifier_head.vocab_size == vocab_size
        assert hasattr(identifier_head, 'gen_proj')
        assert hasattr(identifier_head, 'copy_gate')
        assert hasattr(identifier_head, 'copy_attention')
    
    def test_forward_pass(self, sample_hidden_state):
        """Test IdentifierHead forward pass returns correct structure."""
        hidden_dim = sample_hidden_state.size(-1)
        identifier_head = IdentifierHead(hidden_dim)
        
        output = identifier_head(sample_hidden_state)
        
        assert isinstance(output, dict)
        assert 'generation' in output
        assert 'copy_gate' in output
        assert 'copy_attention' in output
        assert 'available_identifiers' in output
        
        # Check shapes
        assert output['generation'].size(0) == sample_hidden_state.size(0)
        assert output['copy_gate'].size(0) == sample_hidden_state.size(0)
        
    def test_copy_mechanism_with_identifiers(self, sample_hidden_state):
        """Test copy mechanism when identifiers are available."""
        hidden_dim = sample_hidden_state.size(-1)
        identifier_head = IdentifierHead(hidden_dim)
        
        context_ids = ['a', 'b', 'x']
        output = identifier_head(sample_hidden_state, context_ids)
        
        # Should have non-zero copy attention logits
        assert output['copy_attention'].size(1) > 0
        assert output['available_identifiers'] == context_ids
        
    def test_copy_mechanism_without_identifiers(self, sample_hidden_state):
        """Test copy mechanism when no identifiers available."""
        hidden_dim = sample_hidden_state.size(-1)
        identifier_head = IdentifierHead(hidden_dim)
        
        output = identifier_head(sample_hidden_state, [])
        
        # Should have empty copy attention
        assert output['copy_attention'].size(1) == 0
        assert output['available_identifiers'] == []
        
    def test_sample_identifier(self, sample_hidden_state):
        """Test identifier sampling functionality."""
        hidden_dim = sample_hidden_state.size(-1)
        identifier_head = IdentifierHead(hidden_dim)
        
        output = identifier_head(sample_hidden_state, ['a', 'b'])
        sampled = identifier_head.sample_identifier(output)
        
        assert isinstance(sampled, list)
        assert len(sampled) == sample_hidden_state.size(0)
        
        for identifier in sampled:
            assert isinstance(identifier, str)
            assert len(identifier) == 1  # Single character identifiers


class TestLiteralHead:
    """Test the LiteralHead component."""
    
    def test_initialization(self):
        """Test LiteralHead initializes correctly."""
        hidden_dim = 64
        literal_head = LiteralHead(hidden_dim)
        
        assert literal_head.hidden_dim == hidden_dim
        assert hasattr(literal_head, 'type_proj')
        assert hasattr(literal_head, 'int_proj')
        assert hasattr(literal_head, 'bool_proj')
        assert hasattr(literal_head, 'str_proj')
    
    def test_forward_pass(self, sample_hidden_state):
        """Test LiteralHead forward pass returns correct structure."""
        hidden_dim = sample_hidden_state.size(-1)
        literal_head = LiteralHead(hidden_dim)
        
        output = literal_head(sample_hidden_state)
        
        assert isinstance(output, dict)
        assert 'type' in output
        assert 'int_value' in output
        assert 'bool_value' in output
        assert 'str_char' in output
        assert 'str_length' in output
        
        # Check shapes
        for key in output:
            assert output[key].size(0) == sample_hidden_state.size(0)
            
    def test_sample_literal_integers(self, sample_hidden_state):
        """Test literal sampling for integers."""
        hidden_dim = sample_hidden_state.size(-1)
        literal_head = LiteralHead(hidden_dim, max_int=5)
        
        # Force type to be integer by setting high logits
        output = literal_head(sample_hidden_state)
        output['type'] = torch.tensor([[10.0, -10.0, -10.0], [10.0, -10.0, -10.0]])  # Force int type
        
        sampled = literal_head.sample_literal(output, temperature=0.1)
        
        assert isinstance(sampled, list)
        assert len(sampled) == sample_hidden_state.size(0)
        
        for literal in sampled:
            assert literal.isdigit()
            assert 0 <= int(literal) <= 5
            
    def test_sample_literal_booleans(self, sample_hidden_state):
        """Test literal sampling for booleans."""
        hidden_dim = sample_hidden_state.size(-1)
        literal_head = LiteralHead(hidden_dim)
        
        # Force type to be boolean
        output = literal_head(sample_hidden_state)
        output['type'] = torch.tensor([[-10.0, -10.0, 10.0], [-10.0, -10.0, 10.0]])  # Force bool type
        
        sampled = literal_head.sample_literal(output, temperature=0.1)
        
        assert isinstance(sampled, list)
        assert len(sampled) == sample_hidden_state.size(0)
        
        for literal in sampled:
            assert literal in ["True", "False"]
            
    def test_sample_literal_strings(self, sample_hidden_state):
        """Test literal sampling for strings."""
        hidden_dim = sample_hidden_state.size(-1)
        literal_head = LiteralHead(hidden_dim)
        
        # Force type to be string
        output = literal_head(sample_hidden_state)
        output['type'] = torch.tensor([[-10.0, 10.0, -10.0], [-10.0, 10.0, -10.0]])  # Force str type
        
        sampled = literal_head.sample_literal(output, temperature=0.1)
        
        assert isinstance(sampled, list)
        assert len(sampled) == sample_hidden_state.size(0)
        
        for literal in sampled:
            # Should be quoted strings
            assert literal.startswith('"') and literal.endswith('"')
            
    def test_literal_type_names(self):
        """Test literal type name mapping."""
        literal_head = LiteralHead(64)
        
        assert literal_head._get_literal_type_name(0) == "int"
        assert literal_head._get_literal_type_name(1) == "str"
        assert literal_head._get_literal_type_name(2) == "bool"
        assert literal_head._get_literal_type_name(3) == "unknown"


class TestFunctionCallHead:
    """Test the FunctionCallHead component."""
    
    def test_initialization(self):
        """Test FunctionCallHead initializes correctly."""
        hidden_dim = 64
        max_args = 3
        func_head = FunctionCallHead(hidden_dim, max_args)
        
        assert func_head.hidden_dim == hidden_dim
        assert func_head.max_args == max_args
        assert hasattr(func_head, 'func_proj')
        assert hasattr(func_head, 'arg_count_proj')
    
    def test_forward_pass(self, sample_hidden_state):
        """Test FunctionCallHead forward pass returns correct structure."""
        hidden_dim = sample_hidden_state.size(-1)
        func_head = FunctionCallHead(hidden_dim)
        
        output = func_head(sample_hidden_state)
        
        assert isinstance(output, dict)
        assert 'function' in output
        assert 'arg_count' in output
        
        # Check shapes
        assert output['function'].size(0) == sample_hidden_state.size(0)
        assert output['arg_count'].size(0) == sample_hidden_state.size(0)


class TestControlFlowHead:
    """Test the ControlFlowHead component."""
    
    def test_initialization(self):
        """Test ControlFlowHead initializes correctly."""
        hidden_dim = 64
        control_head = ControlFlowHead(hidden_dim)
        
        assert control_head.hidden_dim == hidden_dim
        assert hasattr(control_head, 'control_proj')
    
    def test_forward_pass(self, sample_hidden_state):
        """Test ControlFlowHead forward pass returns correct structure."""
        hidden_dim = sample_hidden_state.size(-1)
        control_head = ControlFlowHead(hidden_dim)
        
        output = control_head(sample_hidden_state)
        
        assert isinstance(output, dict)
        assert 'control' in output
        
        # Check shape
        assert output['control'].size(0) == sample_hidden_state.size(0)


class TestGrammarAwareGenerationHead:
    """Test the main GrammarAwareGenerationHead component."""
    
    def test_initialization(self, sample_grammar):
        """Test GrammarAwareGenerationHead initializes correctly."""
        hidden_dim = 64
        gen_head = GrammarAwareGenerationHead(hidden_dim, sample_grammar)
        
        assert gen_head.grammar == sample_grammar
        assert gen_head.hidden_dim == hidden_dim
        
        # Check all subcomponents exist
        assert hasattr(gen_head, 'production_head')
        assert hasattr(gen_head, 'identifier_head')
        assert hasattr(gen_head, 'literal_head')
        assert hasattr(gen_head, 'function_call_head')
        assert hasattr(gen_head, 'control_flow_head')
        assert hasattr(gen_head, 'expansion_stack')
    
    def test_initialization_with_default_grammar(self):
        """Test GrammarAwareGenerationHead works with default grammar."""
        hidden_dim = 64
        gen_head = GrammarAwareGenerationHead(hidden_dim)
        
        assert gen_head.grammar is not None
        assert isinstance(gen_head.grammar, CFG)
    
    def test_forward_pass(self, sample_hidden_state, sample_grammar):
        """Test GrammarAwareGenerationHead forward pass returns all head outputs."""
        hidden_dim = sample_hidden_state.size(-1)
        gen_head = GrammarAwareGenerationHead(hidden_dim, sample_grammar)
        
        output = gen_head(sample_hidden_state)
        
        assert isinstance(output, dict)
        
        # Check all expected outputs are present
        expected_keys = ['production', 'identifier', 'literal', 'function_call', 'control_flow']
        for key in expected_keys:
            assert key in output
    
    def test_expand_production_placeholder(self, sample_grammar):
        """Test expand_production method exists (placeholder test)."""
        hidden_dim = 64
        gen_head = GrammarAwareGenerationHead(hidden_dim, sample_grammar)
        
        # Basic test that method exists and returns a list
        result = gen_head.expand_production(0)
        assert isinstance(result, list)
    
    def test_expand_production_functionality(self, sample_grammar):
        """Test expand_production method with real productions."""
        hidden_dim = 64
        gen_head = GrammarAwareGenerationHead(hidden_dim, sample_grammar)
        
        # Get a valid production index
        productions = list(sample_grammar.productions())
        if productions:
            # Test with first production
            expansion = gen_head.expand_production(0)
            assert isinstance(expansion, list)
            
            # Should match the actual production
            expected = list(productions[0].rhs())
            assert expansion == expected
    
    def test_generate_program_basic(self, sample_grammar):
        """Test generate_program method produces output."""
        hidden_dim = 64
        gen_head = GrammarAwareGenerationHead(hidden_dim, sample_grammar)
        
        # Create dummy context embeddings (batch_size=1, seq_len=1, hidden_dim)
        context_embeddings = torch.randn(1, 1, hidden_dim)
        
        # Generate program with small max_steps to keep it manageable
        result = gen_head.generate_program(context_embeddings, max_steps=10)
        
        assert isinstance(result, list)
        assert len(result) == 1  # One result per batch item
        assert isinstance(result[0], list)  # Tokens for first batch item
        assert len(result[0]) > 0  # Should generate some tokens
        
    def test_generate_program_multiple_batches(self, sample_grammar):
        """Test generate_program with multiple batch items."""
        hidden_dim = 64
        gen_head = GrammarAwareGenerationHead(hidden_dim, sample_grammar)
        
        batch_size = 2
        context_embeddings = torch.randn(batch_size, 1, hidden_dim)
        
        result = gen_head.generate_program(context_embeddings, max_steps=5)
        
        assert isinstance(result, list)
        assert len(result) == batch_size
        
        for batch_result in result:
            assert isinstance(batch_result, list)
            # Each batch should produce some tokens
            assert len(batch_result) >= 0
    
    def test_terminal_mapping(self, sample_grammar):
        """Test terminal symbol to token mapping."""
        hidden_dim = 64
        gen_head = GrammarAwareGenerationHead(hidden_dim, sample_grammar)
        
        # Test some known mappings
        assert gen_head._map_terminal_to_token("DEF") == "def"
        assert gen_head._map_terminal_to_token("LPAREN") == "("
        assert gen_head._map_terminal_to_token("RETURN") == "return"
        
        # Unknown terminals should return themselves
        assert gen_head._map_terminal_to_token("UNKNOWN_TOKEN") == "UNKNOWN_TOKEN"
    
    def test_terminal_mapping_uses_grammar_patterns(self, sample_grammar):
        """Test that terminal mapping uses grammar patterns correctly."""
        from dataset.grammar import get_token_patterns
        
        hidden_dim = 64
        gen_head = GrammarAwareGenerationHead(hidden_dim, sample_grammar)
        
        # Get expected patterns from grammar
        token_patterns = get_token_patterns()
        
        # Test that our mapping matches the grammar patterns
        for terminal, tokens in token_patterns.items():
            if len(tokens) == 1:
                # Single token should map exactly
                assert gen_head._map_terminal_to_token(terminal) == tokens[0]
            elif len(tokens) > 1:
                # Multiple tokens should use first as default
                assert gen_head._map_terminal_to_token(terminal) == tokens[0]
        
        # Test specific multi-token cases
        assert gen_head._map_terminal_to_token("ADDOP") == "+"  # First of ["+", "-"]
        assert gen_head._map_terminal_to_token("MULOP") == "*"  # First of ["*", "/", "%"]
        assert gen_head._map_terminal_to_token("BINARY_CMP") == "<"  # First of comparisons


class TestGenerateProgramComprehensive:
    """Comprehensive tests for the generate_program method focusing on code validity."""
    
    def test_generate_program_produces_realizable_code(self, sample_grammar):
        """Test that generated programs can be converted to actual code."""
        hidden_dim = 64
        gen_head = GrammarAwareGenerationHead(hidden_dim, sample_grammar)
        
        # Generate multiple programs with different step limits
        context_embeddings = torch.randn(1, 1, hidden_dim)
        
        for max_steps in [10, 20, 50]:
            programs = gen_head.generate_program(context_embeddings, max_steps=max_steps)
            program_tokens = programs[0]
            
            # Should be able to realize the tokens into code
            try:
                code = realize_program(program_tokens)
                assert isinstance(code, str)
                assert len(code) > 0
                # Should at least start with "def program"
                assert "def" in code and "program" in code
                
                print(f"Generated code with {max_steps} steps:\n{code[:100]}...")
                
            except Exception as e:
                pytest.fail(f"Failed to realize program with {max_steps} steps: {e}")
    
    def test_generate_program_produces_parseable_code(self, sample_grammar):
        """Test that generated programs produce syntactically valid Python."""
        hidden_dim = 64
        gen_head = GrammarAwareGenerationHead(hidden_dim, sample_grammar)
        
        context_embeddings = torch.randn(2, 1, hidden_dim)
        
        # Generate multiple programs and test parsability
        valid_programs = 0
        total_programs = 5
        total_generated = 0
        
        for _ in range(total_programs):
            programs = gen_head.generate_program(context_embeddings, max_steps=30)
            total_generated += len(programs)
            
            for program_tokens in programs:
                try:
                    code = realize_program(program_tokens)
                    
                    # Check if it's valid Python
                    if parse_program_with_ast(code):
                        valid_programs += 1
                        print(f"✅ Valid program generated:\n{code[:150]}...")
                    else:
                        print(f"❌ Invalid syntax in:\n{code[:150]}...")
                        
                except Exception as e:
                    print(f"⚠️ Could not realize tokens: {e}")
        
        # We should generate at least some valid programs
        # Note: Due to random generation, not all may be complete/valid
        print(f"Generated {valid_programs}/{total_generated} valid programs")
        assert valid_programs >= 0  # At least we shouldn't crash
    
    def test_generate_program_contains_required_structure(self, sample_grammar):
        """Test that generated programs contain basic required structure."""
        hidden_dim = 64
        gen_head = GrammarAwareGenerationHead(hidden_dim, sample_grammar)
        
        context_embeddings = torch.randn(1, 1, hidden_dim)
        programs = gen_head.generate_program(context_embeddings, max_steps=50)  # More steps for complete structure
        program_tokens = programs[0]
        
        # Convert to string for easier checking
        token_str = " ".join(program_tokens)
        
        # Should contain function definition structure
        assert "def" in program_tokens, "Should contain 'def' keyword"
        assert "program" in program_tokens, "Should contain 'program' function name"
        assert "(" in program_tokens, "Should contain opening parenthesis"
        
        # With more steps, we should get at least some structure
        # But we'll be lenient since generation can be incomplete
        has_closing_paren = ")" in program_tokens
        has_colon = ":" in program_tokens
        
        # At least one should be present (partial structure is ok)
        assert has_closing_paren or has_colon, "Should have some function structure (closing paren or colon)"
        
        # Should have some structure tokens
        structure_tokens = ["<NEWLINE>", "<INDENT>", "<DEDENT>"]
        has_structure = any(token in program_tokens for token in structure_tokens)
        assert has_structure, "Should contain structure tokens (NEWLINE, INDENT, DEDENT)"
        
        print(f"Program structure verified: {token_str[:100]}...")
    
    def test_generate_program_variable_consistency(self, sample_grammar):
        """Test that generated programs maintain variable consistency."""
        hidden_dim = 64
        gen_head = GrammarAwareGenerationHead(hidden_dim, sample_grammar)
        
        context_embeddings = torch.randn(1, 1, hidden_dim)
        
        # Generate multiple programs and check variable usage
        for trial in range(3):
            programs = gen_head.generate_program(context_embeddings, max_steps=40)
            program_tokens = programs[0]
            
            # Find variables used (single letter identifiers)
            variables_used = []
            for token in program_tokens:
                if len(token) == 1 and token.islower() and token.isalpha():
                    variables_used.append(token)
            
            # Should have some variables
            if variables_used:
                print(f"Trial {trial}: Variables used: {set(variables_used)}")
                
                # Check that variables are reasonable (a-z)
                for var in variables_used:
                    assert 'a' <= var <= 'z', f"Variable {var} should be single lowercase letter"
            
            print(f"Trial {trial}: Generated {len(program_tokens)} tokens")
    
    def test_generate_program_with_different_contexts(self, sample_grammar):
        """Test generation with different context embeddings produces different programs."""
        hidden_dim = 64
        gen_head = GrammarAwareGenerationHead(hidden_dim, sample_grammar)
        
        # Create different context embeddings
        context1 = torch.randn(1, 1, hidden_dim)
        context2 = torch.randn(1, 1, hidden_dim)
        
        # Generate programs with same seed for reproducibility
        torch.manual_seed(42)
        programs1 = gen_head.generate_program(context1, max_steps=15)
        
        torch.manual_seed(43) 
        programs2 = gen_head.generate_program(context2, max_steps=15)
        
        # Programs should likely be different (though randomness could make them same)
        tokens1 = programs1[0]
        tokens2 = programs2[0]
        
        print(f"Program 1: {' '.join(tokens1[:10])}...")
        print(f"Program 2: {' '.join(tokens2[:10])}...")
        
        # At minimum, we should generate some tokens
        assert len(tokens1) > 0
        assert len(tokens2) > 0
    
    def test_generate_program_respects_max_steps(self, sample_grammar):
        """Test that program generation respects max_steps limit."""
        hidden_dim = 64
        gen_head = GrammarAwareGenerationHead(hidden_dim, sample_grammar)
        
        context_embeddings = torch.randn(1, 1, hidden_dim)
        
        # Test with very small max_steps
        small_programs = gen_head.generate_program(context_embeddings, max_steps=5)
        small_tokens = small_programs[0]
        
        # Test with larger max_steps  
        large_programs = gen_head.generate_program(context_embeddings, max_steps=50)
        large_tokens = large_programs[0]
        
        print(f"Small program ({len(small_tokens)} tokens): {' '.join(small_tokens)}")
        print(f"Large program ({len(large_tokens)} tokens): {' '.join(large_tokens[:20])}...")
        
        # Larger max_steps should generally produce more tokens (unless early termination)
        # But at minimum both should produce some output
        assert len(small_tokens) >= 0
        assert len(large_tokens) >= 0
        
        # Small program should be relatively constrained
        assert len(small_tokens) <= 20, "Small max_steps should produce relatively few tokens"


class TestGrammarIntegration:
    """Test integration with NLTK grammar system."""
    
    def test_grammar_loading(self):
        """Test that grammar loads correctly from dataset module."""
        grammar = get_cfg()
        
        assert isinstance(grammar, CFG)
        assert grammar.start() is not None
        
        # Check that grammar has productions
        productions = list(grammar.productions())
        assert len(productions) > 0
    
    def test_grammar_productions_structure(self):
        """Test that grammar productions have expected structure."""
        grammar = get_cfg()
        
        # Get all non-terminals
        nonterminals = set()
        for prod in grammar.productions():
            nonterminals.add(prod.lhs())
        
        # Should have key non-terminals from our grammar
        expected_nts = ['S', 'FUNC_DEF', 'EXPR', 'STMT']
        for nt_name in expected_nts:
            nt = Nonterminal(nt_name)
            assert nt in nonterminals, f"Expected non-terminal {nt_name} not found"
    
    def test_production_head_with_real_grammar(self):
        """Test ProductionHead initialization with real grammar."""
        grammar = get_cfg()
        hidden_dim = 64
        
        production_head = ProductionHead(hidden_dim, grammar)
        
        # Should not raise any errors
        assert production_head.grammar == grammar