#!/usr/bin/env python3
"""
Test script for program synthesis dataset.
Loads all Python programs from a dataset directory and tests them against their YAML specifications.
"""

import os
import sys
import yaml
import json
import traceback
from typing import Dict, Any, List, Tuple
import argparse
from dataclasses import dataclass


@dataclass
class TestResult:
    program_id: str
    program_name: str
    total_examples: int
    passed_examples: int
    failed_examples: int
    errors: List[str]
    success_rate: float


def load_program_spec(yaml_path: str) -> Dict[str, Any]:
    """Load program specification from YAML file"""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def load_python_program(py_path: str) -> str:
    """Load Python program code"""
    with open(py_path, 'r') as f:
        return f.read()


def execute_program_on_example(code: str, example: Dict[str, Any]) -> Tuple[bool, Any, str]:
    """
    Execute Python program on a single example.
    Returns (success, actual_output, error_message)
    """
    try:
        # Create a local namespace for execution
        local_namespace = {}
        
        # Execute the program code to define the function
        exec(code, local_namespace)
        
        # Get the program function (should be named 'program')
        if 'program' not in local_namespace:
            return False, None, "No 'program' function found in code"
        
        program_func = local_namespace['program']
        
        # Prepare input arguments
        program_input = example['input']
        expected_output = example['output']
        
        # Call the program function with appropriate arguments
        if isinstance(program_input, list):
            # Multiple arguments
            actual_output = program_func(*program_input)
        else:
            # Single argument
            actual_output = program_func(program_input)
        
        # Compare outputs
        if actual_output == expected_output:
            return True, actual_output, ""
        else:
            return False, actual_output, f"Expected {expected_output}, got {actual_output}"
    
    except Exception as e:
        return False, None, f"Runtime error: {str(e)}"


def test_single_program(program_base_path: str) -> TestResult:
    """Test a single program against its specification"""
    program_id = os.path.basename(program_base_path)
    yaml_path = f"{program_base_path}.yaml"
    py_path = f"{program_base_path}.py"
    
    # Load specification
    try:
        spec = load_program_spec(yaml_path)
    except Exception as e:
        return TestResult(
            program_id=program_id,
            program_name="unknown",
            total_examples=0,
            passed_examples=0,
            failed_examples=0,
            errors=[f"Failed to load YAML spec: {str(e)}"],
            success_rate=0.0
        )
    
    # Load Python code
    try:
        code = load_python_program(py_path)
    except Exception as e:
        return TestResult(
            program_id=program_id,
            program_name=spec.get('description', 'unknown'),
            total_examples=0,
            passed_examples=0,
            failed_examples=0,
            errors=[f"Failed to load Python code: {str(e)}"],
            success_rate=0.0
        )
    
    # Test each example
    examples = spec.get('examples', [])
    total_examples = len(examples)
    passed_examples = 0
    errors = []
    
    for i, example in enumerate(examples):
        success, actual_output, error_msg = execute_program_on_example(code, example)
        
        if success:
            passed_examples += 1
        else:
            # More detailed error information including input/output details
            input_str = str(example['input'])
            expected_str = str(example['output'])
            actual_str = str(actual_output) if actual_output is not None else "None"
            
            detailed_error = f"Example {i+1}/{total_examples}: Input={input_str}, Expected={expected_str}, Got={actual_str} | {error_msg}"
            errors.append(detailed_error)
    
    failed_examples = total_examples - passed_examples
    success_rate = passed_examples / total_examples if total_examples > 0 else 0.0
    
    return TestResult(
        program_id=program_id,
        program_name=spec.get('description', 'unknown'),
        total_examples=total_examples,
        passed_examples=passed_examples,
        failed_examples=failed_examples,
        errors=errors,
        success_rate=success_rate
    )


def find_program_files(dataset_dir: str) -> List[str]:
    """Find all program files in dataset directory"""
    program_files = []
    
    # Look in both train and test directories
    for split_dir in ['train', 'test']:
        split_path = os.path.join(dataset_dir, split_dir)
        if not os.path.exists(split_path):
            continue
        
        # Find all .yaml files and extract base names
        for file in os.listdir(split_path):
            if file.endswith('.yaml') and file.startswith('program'):
                base_name = file[:-5]  # Remove .yaml extension
                base_path = os.path.join(split_path, base_name)
                
                # Check that corresponding .py file exists
                if os.path.exists(f"{base_path}.py"):
                    program_files.append(base_path)
    
    return sorted(program_files)


def print_test_summary(results: List[TestResult]):
    """Print summary of test results"""
    total_programs = len(results)
    total_examples = sum(r.total_examples for r in results)
    total_passed = sum(r.passed_examples for r in results)
    total_failed = sum(r.failed_examples for r in results)
    
    programs_with_all_passed = sum(1 for r in results if r.success_rate == 1.0)
    programs_with_some_failed = sum(1 for r in results if 0 < r.success_rate < 1.0)
    programs_with_all_failed = sum(1 for r in results if r.success_rate == 0.0)
    
    overall_success_rate = total_passed / total_examples if total_examples > 0 else 0.0
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total programs tested: {total_programs}")
    print(f"Total examples tested: {total_examples}")
    print(f"Total examples passed: {total_passed}")
    print(f"Total examples failed: {total_failed}")
    print(f"Overall success rate: {overall_success_rate:.2%}")
    print()
    print(f"Programs with all examples passed: {programs_with_all_passed}")
    print(f"Programs with some examples failed: {programs_with_some_failed}")
    print(f"Programs with all examples failed: {programs_with_all_failed}")
    print("="*80)


def print_detailed_results(results: List[TestResult], show_errors: bool = False):
    """Print detailed results for each program"""
    print("\nDETAILED RESULTS:")
    print("-" * 80)
    
    for result in results:
        status = "PASS" if result.success_rate == 1.0 else "FAIL" if result.success_rate == 0.0 else "PARTIAL"
        print(f"[{status}] {result.program_id}: {result.program_name}")
        print(f"    Examples: {result.passed_examples}/{result.total_examples} passed ({result.success_rate:.1%})")
        
        if show_errors and result.errors:
            print("    Errors:")
            for error in result.errors:  # Show all errors, not just first 3
                print(f"      - {error}")
        print()


def print_failure_analysis(results: List[TestResult]):
    """Print detailed analysis of failed programs"""
    failed_programs = [r for r in results if r.success_rate < 1.0]
    
    if not failed_programs:
        return
    
    print(f"\nFAILURE ANALYSIS ({len(failed_programs)} programs with failures):")
    print("=" * 80)
    
    # Categorize failures
    complete_failures = [r for r in failed_programs if r.success_rate == 0.0]
    partial_failures = [r for r in failed_programs if 0 < r.success_rate < 1.0]
    
    if complete_failures:
        print(f"\nCOMPLETE FAILURES ({len(complete_failures)} programs - all examples failed):")
        print("-" * 60)
        for result in complete_failures:
            print(f"• {result.program_id}: {result.program_name}")
            print(f"  Failed all {result.total_examples} examples")
            if result.errors:
                print(f"  Sample error: {result.errors[0]}")
            print()
    
    if partial_failures:
        print(f"\nPARTIAL FAILURES ({len(partial_failures)} programs - some examples failed):")
        print("-" * 60)
        for result in partial_failures:
            print(f"• {result.program_id}: {result.program_name}")
            print(f"  Passed {result.passed_examples}/{result.total_examples} examples ({result.success_rate:.1%})")
            print(f"  Failed examples: {result.failed_examples}")
            if result.errors:
                print("  Failed example details:")
                for error in result.errors:
                    print(f"    - {error}")
            print()
    
    # Summary by failure patterns
    print("\nFAILURE PATTERNS:")
    print("-" * 40)
    
    # Group by error type
    error_patterns = {}
    for result in failed_programs:
        for error in result.errors:
            # Extract error type (before the colon)
            if "Runtime error:" in error:
                error_type = "Runtime Error"
            elif "Expected" in error and "got" in error:
                error_type = "Wrong Output"
            elif "No 'program' function found" in error:
                error_type = "Missing Function"
            else:
                error_type = "Other"
            
            if error_type not in error_patterns:
                error_patterns[error_type] = []
            error_patterns[error_type].append(result.program_id)
    
    for error_type, program_ids in error_patterns.items():
        unique_programs = list(set(program_ids))
        print(f"• {error_type}: {len(unique_programs)} programs affected")
        if len(unique_programs) <= 5:
            print(f"  Programs: {', '.join(unique_programs)}")
        else:
            print(f"  Programs: {', '.join(unique_programs[:5])} ... and {len(unique_programs)-5} more")


def save_results_to_json(results: List[TestResult], output_path: str):
    """Save test results to JSON file"""
    results_data = []
    for result in results:
        results_data.append({
            'program_id': result.program_id,
            'program_name': result.program_name,
            'total_examples': result.total_examples,
            'passed_examples': result.passed_examples,
            'failed_examples': result.failed_examples,
            'success_rate': result.success_rate,
            'errors': result.errors
        })
    
    summary = {
        'total_programs': len(results),
        'total_examples': sum(r.total_examples for r in results),
        'total_passed': sum(r.passed_examples for r in results),
        'total_failed': sum(r.failed_examples for r in results),
        'overall_success_rate': sum(r.passed_examples for r in results) / sum(r.total_examples for r in results) if sum(r.total_examples for r in results) > 0 else 0.0,
        'programs_all_passed': sum(1 for r in results if r.success_rate == 1.0),
        'programs_some_failed': sum(1 for r in results if 0 < r.success_rate < 1.0),
        'programs_all_failed': sum(1 for r in results if r.success_rate == 0.0)
    }
    
    output_data = {
        'summary': summary,
        'results': results_data
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test program synthesis dataset")
    parser.add_argument("dataset_path", help="Path to the dataset directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--show-errors", action="store_true", help="Show error messages for failed examples")
    parser.add_argument("--brief", action="store_true", help="Show brief summary only (skip failure analysis)")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    parser.add_argument("--limit", type=int, help="Limit number of programs to test (for debugging)")
    
    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory '{dataset_path}' does not exist")
        sys.exit(1)
    
    print(f"Testing program synthesis dataset: {dataset_path}")
    
    # Find all program files
    program_files = find_program_files(dataset_path)
    
    if not program_files:
        print("No program files found in dataset directory")
        sys.exit(1)
    
    if args.limit:
        program_files = program_files[:args.limit]
        print(f"Limited to first {args.limit} programs")
    
    print(f"Found {len(program_files)} programs to test")
    
    # Test each program
    results = []
    failed_programs = []
    
    for i, program_path in enumerate(program_files):
        if args.verbose:
            print(f"Testing {i+1}/{len(program_files)}: {os.path.basename(program_path)}")
        
        try:
            result = test_single_program(program_path)
            results.append(result)
            
            if result.success_rate < 1.0:
                failed_programs.append(result)
                
        except Exception as e:
            print(f"Error testing {program_path}: {str(e)}")
            traceback.print_exc()
    
    # Print results
    print_test_summary(results)
    
    if args.verbose:
        print_detailed_results(results, show_errors=args.show_errors)
    
    # Show failure analysis unless brief mode is requested
    if not args.brief:
        print_failure_analysis(results)
    
    # Save results if requested
    if args.output:
        save_results_to_json(results, args.output)
    
    # Exit with appropriate code
    overall_success_rate = sum(r.passed_examples for r in results) / sum(r.total_examples for r in results) if results else 0.0
    if overall_success_rate == 1.0:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print(f"\nSome tests failed (overall success rate: {overall_success_rate:.1%})")
        sys.exit(1)


if __name__ == "__main__":
    main()