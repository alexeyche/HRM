"""
Tests for dataset/build_program_dataset.py

Tests round-trip serialization/deserialization of all programs in the registry
to ensure data integrity and safety.
"""
import pytest
import torch
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any

from torch_geometric.data import Data

from dataset.programs import get_program_registry, ProgramRegistry, Example
from dataset.graph_dataset import ProgramGraphDataset
from dataset.build_program_dataset import (
    _data_to_dict, 
    _dict_to_data,
    _make_info_safe,
    generate_program_dataset,
    load_sample
)


class TestDataSerialization:
    """Test Data object serialization/deserialization."""
    
    def test_data_to_dict_round_trip(self):
        """Test that Data objects can be converted to dict and back without loss."""
        # Create a sample Data object with various attributes
        original_data = Data(
            x=torch.randn(5, 3),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
            edge_attr=torch.randn(4, 2),
            y=torch.tensor([1, 0, 1]),
            num_nodes=5
        )
        
        # Add some custom attributes that Data objects can have
        original_data.spec_name = "test_program"
        original_data.description = "A test program"
        original_data.program_code = "def program(x): return x + 1"
        
        # Convert to dict and back
        data_dict = _data_to_dict(original_data)
        reconstructed_data = _dict_to_data(data_dict)
        
        # Verify all tensors are preserved
        assert torch.equal(original_data.x, reconstructed_data.x)
        assert torch.equal(original_data.edge_index, reconstructed_data.edge_index)
        assert torch.equal(original_data.edge_attr, reconstructed_data.edge_attr)
        assert torch.equal(original_data.y, reconstructed_data.y)
        assert original_data.num_nodes == reconstructed_data.num_nodes
        
        # Verify custom attributes are preserved
        assert original_data.spec_name == reconstructed_data.spec_name
        assert original_data.description == reconstructed_data.description
        assert original_data.program_code == reconstructed_data.program_code
        
    def test_empty_data_object(self):
        """Test serialization of empty Data objects."""
        original_data = Data()
        
        data_dict = _data_to_dict(original_data)
        reconstructed_data = _dict_to_data(data_dict)
        
        # Should be able to handle empty Data objects
        assert isinstance(reconstructed_data, Data)
        
    def test_data_dict_contains_only_safe_types(self):
        """Test that data_dict contains only safe types for torch.load."""
        original_data = Data(
            x=torch.randn(3, 2),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            spec_name="test",
            description="test desc"
        )
        
        data_dict = _data_to_dict(original_data)
        
        # Verify all values are safe types
        for key, value in data_dict.items():
            assert isinstance(value, (torch.Tensor, str, int, float, bool, type(None))), \
                f"Unsafe type {type(value)} for key {key}"


class TestInfoSafety:
    """Test program info dictionary safety."""
    
    def test_make_info_safe_with_examples(self):
        """Test that Example objects are converted to safe dictionaries."""
        # Create sample info with Example objects
        examples = [
            Example(input=5, output=10),
            Example(input=[1, 2, 3], output=6),
            Example(input="hello", output="HELLO")
        ]
        
        info = {
            'spec_name': 'test_program',
            'description': 'A test program',
            'examples': examples,
            'input_types': ['int'],
            'output_types': ['int'],
            'program_code': 'def program(x): return x * 2',
            'instance_idx': 0
        }
        
        safe_info = _make_info_safe(info)
        
        # Verify examples are converted to dicts
        assert len(safe_info['examples']) == 3
        for example in safe_info['examples']:
            assert isinstance(example, dict)
            assert 'input' in example
            assert 'output' in example
            assert not isinstance(example, Example)
        
        # Verify example data is preserved
        assert safe_info['examples'][0]['input'] == 5
        assert safe_info['examples'][0]['output'] == 10
        assert safe_info['examples'][1]['input'] == [1, 2, 3]
        assert safe_info['examples'][1]['output'] == 6
        
        # Verify other fields are unchanged
        assert safe_info['spec_name'] == 'test_program'
        assert safe_info['description'] == 'A test program'
        
    def test_make_info_safe_without_examples(self):
        """Test info dictionary without Example objects."""
        info = {
            'spec_name': 'test_program',
            'description': 'A test program',
            'input_types': ['int'],
            'output_types': ['int']
        }
        
        safe_info = _make_info_safe(info)
        
        # Should preserve all fields
        assert safe_info == info
        
    def test_make_info_safe_with_mixed_examples(self):
        """Test info with mix of Example objects and plain dicts."""
        examples = [
            Example(input=1, output=2),
            {'input': 3, 'output': 4},  # Already a dict
            Example(input=5, output=6)
        ]
        
        info = {'examples': examples}
        safe_info = _make_info_safe(info)
        
        # All should be converted to dicts
        assert len(safe_info['examples']) == 3
        for example in safe_info['examples']:
            assert isinstance(example, dict)
            assert 'input' in example
            assert 'output' in example


class TestRegistryRoundTrip:
    """Test round-trip serialization for all programs in the registry."""
    
    def test_all_programs_round_trip(self):
        """Test that all programs in the registry can be serialized and loaded safely."""
        registry = get_program_registry()
        program_names = registry.list_names()
        
        assert len(program_names) > 0, "Registry should contain programs"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a small dataset with one instance of each program
            dataset = ProgramGraphDataset(
                registry=registry,
                programs_per_spec=1,
                examples_per_program=2,
                seed=42,
                cache_dir=None
            )
            
            print(f"Testing {len(dataset)} program instances")
            
            # Test each program in the dataset
            for idx in range(len(dataset)):
                graph, info = dataset[idx]
                
                # Verify we got valid data
                assert isinstance(graph, Data), f"Expected Data object for program {idx}"
                assert isinstance(info, dict), f"Expected dict for info {idx}"
                assert 'spec_name' in info, f"Missing spec_name for program {idx}"
                assert 'program_code' in info, f"Missing program_code for program {idx}"
                
                # Test serialization round-trip
                temp_file = Path(temp_dir) / f"test_{idx:06d}.pt"
                
                # Convert to safe format
                graph_dict = _data_to_dict(graph)
                safe_info = _make_info_safe(info)
                
                # Save with safe serialization
                torch.save({
                    'graph_dict': graph_dict,
                    'info': safe_info,
                    'index': idx
                }, temp_file)
                
                # Load back with safe loading
                loaded_graph, loaded_info, loaded_idx = load_sample(str(temp_file))
                
                # Verify loaded data matches original
                self._assert_graphs_equal(graph, loaded_graph, info['spec_name'])
                self._assert_infos_equal(info, loaded_info, info['spec_name'])
                assert idx == loaded_idx, f"Index mismatch for {info['spec_name']}"
                
                print(f"✅ Program {info['spec_name']} passed round-trip test")
    
    def test_safe_loading_with_weights_only(self):
        """Test that all serialized files can be loaded with weights_only=True."""
        registry = get_program_registry()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate a small dataset
            generate_program_dataset(temp_dir, num_samples=5, seed=123)
            
            # Try to load each file with weights_only=True
            graphs_dir = Path(temp_dir) / "graphs"
            for pt_file in graphs_dir.glob("*.pt"):
                # This should not raise any security warnings
                data = torch.load(pt_file, weights_only=True)
                
                # Verify structure
                assert 'graph_dict' in data
                assert 'info' in data
                assert 'index' in data
                
                # Verify all data types are safe
                assert isinstance(data['graph_dict'], dict)
                assert isinstance(data['info'], dict)
                assert isinstance(data['index'], int)
                
                # Verify examples are safe dicts, not Example objects
                if 'examples' in data['info']:
                    for example in data['info']['examples']:
                        assert isinstance(example, dict)
                        assert not isinstance(example, Example)
                
                print(f"✅ Safe loading verified for {pt_file.name}")
    
    def _assert_graphs_equal(self, original: Data, loaded: Data, program_name: str):
        """Assert that two Data objects are equivalent."""
        # Check all tensor attributes that exist in original
        for key, orig_tensor in original.items():
            if isinstance(orig_tensor, torch.Tensor):
                loaded_tensor = getattr(loaded, key, None)
                
                assert loaded_tensor is not None, \
                    f"Missing tensor {key} in loaded graph for {program_name}"
                assert torch.equal(orig_tensor, loaded_tensor), \
                    f"Tensor {key} mismatch for {program_name}"
        
        # Check scalar attributes that were added by the graph dataset
        for key in ['spec_name', 'description', 'program_code']:
            if hasattr(original, key):
                orig_value = getattr(original, key)
                loaded_value = getattr(loaded, key, None)
                
                assert loaded_value == orig_value, \
                    f"Attribute {key} mismatch for {program_name}: {orig_value} vs {loaded_value}"
    
    def _assert_infos_equal(self, original: Dict[str, Any], loaded: Dict[str, Any], program_name: str):
        """Assert that two info dictionaries are equivalent."""
        # Check all keys except examples (handled separately)
        for key in original:
            if key == 'examples':
                continue
                
            assert key in loaded, f"Missing key {key} in loaded info for {program_name}"
            assert original[key] == loaded[key], \
                f"Info key {key} mismatch for {program_name}: {original[key]} vs {loaded[key]}"
        
        # Check examples separately (they get converted from Example objects to dicts)
        if 'examples' in original:
            assert 'examples' in loaded, f"Missing examples in loaded info for {program_name}"
            assert len(original['examples']) == len(loaded['examples']), \
                f"Example count mismatch for {program_name}"
            
            for i, (orig_ex, loaded_ex) in enumerate(zip(original['examples'], loaded['examples'])):
                # Original might be Example object, loaded should be dict
                if isinstance(orig_ex, Example):
                    assert loaded_ex['input'] == orig_ex.input, \
                        f"Example {i} input mismatch for {program_name}"
                    assert loaded_ex['output'] == orig_ex.output, \
                        f"Example {i} output mismatch for {program_name}"
                else:
                    assert loaded_ex == orig_ex, \
                        f"Example {i} mismatch for {program_name}"


class TestDatasetGeneration:
    """Test full dataset generation process."""
    
    def test_generate_program_dataset(self):
        """Test full dataset generation with all programs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate dataset
            generate_program_dataset(temp_dir, num_samples=10, seed=456)
            
            # Verify output structure
            assert (Path(temp_dir) / "metadata.json").exists()
            graphs_dir = Path(temp_dir) / "graphs"
            assert graphs_dir.exists()
            
            # Verify all files can be loaded safely
            pt_files = list(graphs_dir.glob("*.pt"))
            assert len(pt_files) == 10
            
            for pt_file in pt_files:
                graph, info, index = load_sample(str(pt_file))
                
                # Basic validity checks
                assert isinstance(graph, Data)
                assert isinstance(info, dict)
                assert isinstance(index, int)
                assert 'spec_name' in info
                assert 'program_code' in info
                
                print(f"✅ Generated dataset file {pt_file.name} verified")


if __name__ == "__main__":
    # Run the tests
    test_data = TestDataSerialization()
    test_data.test_data_to_dict_round_trip()
    test_data.test_empty_data_object()
    test_data.test_data_dict_contains_only_safe_types()
    print("✅ Data serialization tests passed")
    
    test_info = TestInfoSafety()
    test_info.test_make_info_safe_with_examples()
    test_info.test_make_info_safe_without_examples()
    test_info.test_make_info_safe_with_mixed_examples()
    print("✅ Info safety tests passed")
    
    test_roundtrip = TestRegistryRoundTrip()
    test_roundtrip.test_all_programs_round_trip()
    test_roundtrip.test_safe_loading_with_weights_only()
    print("✅ Registry round-trip tests passed")
    
    test_dataset = TestDatasetGeneration()
    test_dataset.test_generate_program_dataset()
    print("✅ Dataset generation tests passed")
    
    print("\n🎉 All tests passed successfully!")