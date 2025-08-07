from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel, Field
import random
import re
from enum import Enum


class ParameterType(str, Enum):
    """Supported parameter types for program inputs/outputs"""
    INT = "int"
    FLOAT = "float"
    STR = "str"
    BOOL = "bool"
    LIST_INT = "List[int]"
    LIST_FLOAT = "List[float]"
    LIST_STR = "List[str]"
    LIST_BOOL = "List[bool]"
    DICT_STR_INT = "Dict[str,int]"
    DICT_STR_STR = "Dict[str,str]"
    DICT_INT_STR = "Dict[int,str]"
    SET_INT = "Set[int]"
    SET_STR = "Set[str]"
    SET_FLOAT = "Set[float]"


class AugmentationSpec(BaseModel):
    """Specification for parameter augmentation"""
    type: str = Field(..., description="Type of augmentation (RandInt, Choice, etc.)")
    parameters: List[Union[int, float, str]] = Field(..., description="Parameters for the augmentation")
    description: str = Field(..., description="Description of what this augmentation does")

    def generate_value(self) -> Any:
        """Generate a value based on the augmentation specification"""
        if self.type == "RandInt":
            if len(self.parameters) == 2:
                return random.randint(int(self.parameters[0]), int(self.parameters[1]))
            elif len(self.parameters) == 3:
                return random.randint(int(self.parameters[0]), int(self.parameters[1]))
        elif self.type == "Choice":
            return random.choice(self.parameters)
        elif self.type == "RandFloat":
            if len(self.parameters) == 2:
                return random.uniform(float(self.parameters[0]), float(self.parameters[1]))
        elif self.type == "RandArray":
            if len(self.parameters) >= 3:
                min_len, max_len, min_val, max_val = int(self.parameters[0]), int(self.parameters[1]), int(self.parameters[2]), int(self.parameters[3])
                length = random.randint(min_len, max_len)
                return [random.randint(min_val, max_val) for _ in range(length)]
        elif self.type == "RandString":
            if len(self.parameters) >= 2:
                min_len, max_len = int(self.parameters[0]), int(self.parameters[1])
                length = random.randint(min_len, max_len)
                chars = "abcdefghijklmnopqrstuvwxyz"
                return ''.join(random.choice(chars) for _ in range(length))
        elif self.type == "RandBool":
            return random.choice([True, False])

        # Default fallback
        return self.parameters[0] if self.parameters else None


class ParameterSpec(BaseModel):
    """Specification for a program parameter (input or output)"""
    type: ParameterType = Field(..., description="Type of the parameter")
    description: str = Field(..., description="Description of the parameter")
    augmentation: Optional[AugmentationSpec] = Field(None, description="Augmentation specification for generating examples")

    def __str__(self) -> str:
        """String representation for the parameter"""
        if self.augmentation:
            params_str = ", ".join(str(p) for p in self.augmentation.parameters)
            return f"{self.augmentation.type}({params_str})"
        return self.type.value


class Example(BaseModel):
    """A single input-output example"""
    input: Union[int, float, str, bool, List, Dict] = Field(..., description="Input value(s)")
    output: Union[int, float, str, bool, List, Dict] = Field(..., description="Expected output value(s)")


class ProgramSpecification(BaseModel):
    """Complete specification for a program template"""
    name: str = Field(..., description="Unique name for the program")
    description: str = Field(..., description="Description of what the program does")
    inputs: List[ParameterSpec] = Field(..., description="Input parameter specifications")
    outputs: List[ParameterSpec] = Field(..., description="Output parameter specifications")
    base_examples: List[Example] = Field(..., description="Base examples for the program")
    implementation: str = Field(..., description="Reference implementation in Python")

    def generate_examples(self, num_examples: int, seed: Optional[int] = None) -> List[Example]:
        """Generate additional examples using augmentation specifications"""
        if seed is not None:
            random.seed(seed)
            # Import numpy only when needed
            import numpy as np
            np.random.seed(seed)

        examples = [Example(input=ex.input, output=ex.output) for ex in self.base_examples]

        # Generate additional examples using augmentation
        while len(examples) < num_examples:
            try:
                # Generate input values using augmentation specs
                input_values = []
                for param_spec in self.inputs:
                    if param_spec.augmentation:
                        value = param_spec.augmentation.generate_value()
                    else:
                        # Fallback to a reasonable default
                        if param_spec.type == ParameterType.INT:
                            value = random.randint(-10, 10)
                        elif param_spec.type == ParameterType.FLOAT:
                            value = random.uniform(-10.0, 10.0)
                        elif param_spec.type == ParameterType.STR:
                            value = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(5))
                        elif param_spec.type == ParameterType.BOOL:
                            value = random.choice([True, False])
                        elif param_spec.type == ParameterType.LIST_INT:
                            length = random.randint(1, 5)
                            value = [random.randint(-5, 5) for _ in range(length)]
                        else:
                            value = 0  # Default fallback
                    input_values.append(value)

                # Calculate expected output using the reference implementation
                output_value = self._execute_implementation(input_values)

                # Create example
                if len(input_values) == 1:
                    example = Example(input=input_values[0], output=output_value)
                else:
                    example = Example(input=input_values, output=output_value)

                examples.append(example)

            except Exception as e:
                # If execution fails, skip this example
                continue

        return examples[:num_examples]

    def _execute_implementation(self, input_values: List[Any]) -> Any:
        """Execute the reference implementation with given inputs"""
        # Create a safe execution environment
        local_vars = {}

        # Extract function name from implementation
        func_match = re.search(r'def (\w+)\(', self.implementation)
        if not func_match:
            raise ValueError("Could not find function definition in implementation")

        func_name = func_match.group(1)

        # Execute the implementation
        exec(self.implementation, {}, local_vars)

        # Call the function with input values
        func = local_vars[func_name]
        if len(self.inputs) == 1:
            result = func(input_values[0])
        else:
            result = func(*input_values)

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with existing code"""
        return {
            "description": self.description,
            "inputs": [{"type": inp.type.value, "description": inp.description} for inp in self.inputs],
            "outputs": [{"type": out.type.value, "description": out.description} for out in self.outputs],
            "base_examples": [{"input": ex.input, "output": ex.output} for ex in self.base_examples],
            "implementation": self.implementation
        }


class ProgramRegistry:
    """Registry for managing program specifications"""

    def __init__(self):
        self.programs: Dict[str, ProgramSpecification] = {}

    def register(self, spec: ProgramSpecification):
        """Register a program specification"""
        self.programs[spec.name] = spec

    def get(self, name: str) -> Optional[ProgramSpecification]:
        """Get a program specification by name"""
        return self.programs.get(name)

    def list_names(self) -> List[str]:
        """List all registered program names"""
        return list(self.programs.keys())

    def generate_examples(self, name: str, num_examples: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate examples for a specific program"""
        spec = self.get(name)
        if not spec:
            raise ValueError(f"Program '{name}' not found")

        examples = spec.generate_examples(num_examples, seed)
        return [{"input": ex.input, "output": ex.output} for ex in examples]


# Predefined augmentation patterns
def create_randint_augmentation(min_val: int, max_val: int, description: str = "") -> AugmentationSpec:
    """Create a RandInt augmentation specification"""
    return AugmentationSpec(
        type="RandInt",
        parameters=[min_val, max_val],
        description=description or f"Random integer between {min_val} and {max_val}"
    )


def create_choice_augmentation(choices: List[Any], description: str = "") -> AugmentationSpec:
    """Create a Choice augmentation specification"""
    return AugmentationSpec(
        type="Choice",
        parameters=choices,
        description=description or f"Random choice from {choices}"
    )


def create_randarray_augmentation(min_len: int, max_len: int, min_val: int, max_val: int, description: str = "") -> AugmentationSpec:
    """Create a RandArray augmentation specification"""
    return AugmentationSpec(
        type="RandArray",
        parameters=[min_len, max_len, min_val, max_val],
        description=description or f"Random array of length {min_len}-{max_len} with values {min_val}-{max_val}"
    )


def create_randstring_augmentation(min_len: int, max_len: int, description: str = "") -> AugmentationSpec:
    """Create a RandString augmentation specification"""
    return AugmentationSpec(
        type="RandString",
        parameters=[min_len, max_len],
        description=description or f"Random string of length {min_len}-{max_len}"
    )


def create_randbool_augmentation(description: str = "") -> AugmentationSpec:
    """Create a RandBool augmentation specification"""
    return AugmentationSpec(
        type="RandBool",
        parameters=[],
        description=description or "Random boolean value"
    )


# Create the default registry with all the original programs
def create_default_registry() -> ProgramRegistry:
    """Create the default registry with all original program specifications"""
    registry = ProgramRegistry()

    # Basic Math Operations
    registry.register(ProgramSpecification(
        name="sum_up_to_n",
        description="Sum up all numbers up to the input number N",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number N",
            augmentation=create_randint_augmentation(1, 20, "Random positive integer for sum calculation")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="The sum of all numbers up to N", augmentation=None)],
        base_examples=[
            Example(input=5, output=15),
            Example(input=10, output=55),
            Example(input=3, output=6),
            Example(input=1, output=1),
            Example(input=0, output=0)
        ],
        implementation="""def program(n):
    return sum(range(1, n + 1))"""
    ))

    registry.register(ProgramSpecification(
        name="max_of_two",
        description="Return the larger of two numbers",
        inputs=[
            ParameterSpec(
                type=ParameterType.INT,
                description="First number",
                augmentation=create_randint_augmentation(-10, 20, "Random integer for first number")
            ),
            ParameterSpec(
                type=ParameterType.INT,
                description="Second number",
                augmentation=create_randint_augmentation(-10, 20, "Random integer for second number")
            )
        ],
        outputs=[ParameterSpec(type=ParameterType.INT, description="The larger number", augmentation=None)],
        base_examples=[
            Example(input=[5, 3], output=5),
            Example(input=[10, 15], output=15),
            Example(input=[-2, -5], output=-2),
            Example(input=[0, 0], output=0),
            Example(input=[7, 7], output=7)
        ],
        implementation="""def program(a, b):
    return a if a > b else b"""
    ))

    registry.register(ProgramSpecification(
        name="absolute_value",
        description="Return the absolute value of a number",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(-20, 20, "Random integer for absolute value calculation")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="The absolute value", augmentation=None)],
        base_examples=[
            Example(input=-5, output=5),
            Example(input=3, output=3),
            Example(input=0, output=0),
            Example(input=-100, output=100),
            Example(input=42, output=42)
        ],
        implementation="""def program(n):
    return n if n >= 0 else -n"""
    ))

    registry.register(ProgramSpecification(
        name="is_even",
        description="Check if a number is even",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(-10, 20, "Random integer for evenness check")
        )],
        outputs=[ParameterSpec(type=ParameterType.BOOL, description="True if even, False if odd", augmentation=None)],
        base_examples=[
            Example(input=4, output=True),
            Example(input=7, output=False),
            Example(input=0, output=True),
            Example(input=-2, output=True),
            Example(input=-3, output=False)
        ],
        implementation="""def program(n):
    return n % 2 == 0"""
    ))

    registry.register(ProgramSpecification(
        name="factorial",
        description="Calculate the factorial of a number",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number (non-negative)",
            augmentation=create_randint_augmentation(0, 8, "Random small integer for factorial calculation")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="The factorial", augmentation=None)],
        base_examples=[
            Example(input=0, output=1),
            Example(input=1, output=1),
            Example(input=3, output=6),
            Example(input=4, output=24),
            Example(input=5, output=120)
        ],
        implementation="""def program(n):
    if n <= 1:
        return 1
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result"""
    ))

    registry.register(ProgramSpecification(
        name="power_of_two",
        description="Calculate 2 to the power of n",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The exponent",
            augmentation=create_randint_augmentation(0, 10, "Random small exponent for power calculation")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="2^n", augmentation=None)],
        base_examples=[
            Example(input=0, output=1),
            Example(input=1, output=2),
            Example(input=3, output=8),
            Example(input=4, output=16),
            Example(input=5, output=32)
        ],
        implementation="""def program(n):
    return 2 ** n"""
    ))

    registry.register(ProgramSpecification(
        name="square",
        description="Calculate the square of a number",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(-10, 11, "Random integer for square calculation")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="The square of the number", augmentation=None)],
        base_examples=[
            Example(input=0, output=0),
            Example(input=1, output=1),
            Example(input=3, output=9),
            Example(input=-4, output=16),
            Example(input=5, output=25)
        ],
        implementation="""def program(n):
    return n * n"""
    ))

    registry.register(ProgramSpecification(
        name="cube",
        description="Calculate the cube of a number",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(-5, 6, "Random small integer for cube calculation")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="The cube of the number", augmentation=None)],
        base_examples=[
            Example(input=0, output=0),
            Example(input=1, output=1),
            Example(input=2, output=8),
            Example(input=-3, output=-27),
            Example(input=4, output=64)
        ],
        implementation="""def program(n):
    return n * n * n"""
    ))

    registry.register(ProgramSpecification(
        name="min_of_two",
        description="Return the smaller of two numbers",
        inputs=[
            ParameterSpec(
                type=ParameterType.INT,
                description="First number",
                augmentation=create_randint_augmentation(-10, 20, "Random integer for first number")
            ),
            ParameterSpec(
                type=ParameterType.INT,
                description="Second number",
                augmentation=create_randint_augmentation(-10, 20, "Random integer for second number")
            )
        ],
        outputs=[ParameterSpec(type=ParameterType.INT, description="The smaller number", augmentation=None)],
        base_examples=[
            Example(input=[5, 3], output=3),
            Example(input=[10, 15], output=10),
            Example(input=[-2, -5], output=-5),
            Example(input=[0, 0], output=0),
            Example(input=[7, 7], output=7)
        ],
        implementation="""def program(a, b):
    return a if a < b else b"""
    ))

    registry.register(ProgramSpecification(
        name="is_positive",
        description="Check if a number is positive",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(-10, 20, "Random integer for positivity check")
        )],
        outputs=[ParameterSpec(type=ParameterType.BOOL, description="True if positive, False otherwise", augmentation=None)],
        base_examples=[
            Example(input=5, output=True),
            Example(input=-3, output=False),
            Example(input=0, output=False),
            Example(input=10, output=True),
            Example(input=-1, output=False)
        ],
        implementation="""def program(n):
    return n > 0"""
    ))

    registry.register(ProgramSpecification(
        name="is_negative",
        description="Check if a number is negative",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(-10, 20, "Random integer for negativity check")
        )],
        outputs=[ParameterSpec(type=ParameterType.BOOL, description="True if negative, False otherwise", augmentation=None)],
        base_examples=[
            Example(input=-5, output=True),
            Example(input=3, output=False),
            Example(input=0, output=False),
            Example(input=-10, output=True),
            Example(input=1, output=False)
        ],
        implementation="""def program(n):
    return n < 0"""
    ))

    registry.register(ProgramSpecification(
        name="double",
        description="Double a number (multiply by 2)",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(-20, 21, "Random integer for doubling")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="The number multiplied by 2", augmentation=None)],
        base_examples=[
            Example(input=0, output=0),
            Example(input=3, output=6),
            Example(input=-4, output=-8),
            Example(input=10, output=20),
            Example(input=-1, output=-2)
        ],
        implementation="""def program(n):
    return n * 2"""
    ))

    registry.register(ProgramSpecification(
        name="half",
        description="Divide a number by 2 (integer division)",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(-20, 21, "Random integer for halving")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="The number divided by 2", augmentation=None)],
        base_examples=[
            Example(input=0, output=0),
            Example(input=6, output=3),
            Example(input=-8, output=-4),
            Example(input=10, output=5),
            Example(input=5, output=2)
        ],
        implementation="""def program(n):
    return n // 2"""
    ))

    registry.register(ProgramSpecification(
        name="add_ten",
        description="Add 10 to a number",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(-20, 21, "Random integer for adding ten")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="The number plus 10", augmentation=None)],
        base_examples=[
            Example(input=0, output=10),
            Example(input=5, output=15),
            Example(input=-3, output=7),
            Example(input=10, output=20),
            Example(input=-15, output=-5)
        ],
        implementation="""def program(n):
    return n + 10"""
    ))

    registry.register(ProgramSpecification(
        name="subtract_five",
        description="Subtract 5 from a number",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(-15, 26, "Random integer for subtracting five")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="The number minus 5", augmentation=None)],
        base_examples=[
            Example(input=10, output=5),
            Example(input=5, output=0),
            Example(input=0, output=-5),
            Example(input=-3, output=-8),
            Example(input=15, output=10)
        ],
        implementation="""def program(n):
    return n - 5"""
    ))

    registry.register(ProgramSpecification(
        name="is_zero",
        description="Check if a number is zero",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_choice_augmentation([0, -1, 1, 5, -5, 10], "Choice between zero and non-zero numbers")
        )],
        outputs=[ParameterSpec(type=ParameterType.BOOL, description="True if zero, False otherwise", augmentation=None)],
        base_examples=[
            Example(input=0, output=True),
            Example(input=1, output=False),
            Example(input=-1, output=False),
            Example(input=10, output=False),
            Example(input=-5, output=False)
        ],
        implementation="""def program(n):
    return n == 0"""
    ))

    registry.register(ProgramSpecification(
        name="sum_of_two",
        description="Add two numbers together",
        inputs=[
            ParameterSpec(
                type=ParameterType.INT,
                description="First number",
                augmentation=create_randint_augmentation(-10, 21, "Random integer for first number")
            ),
            ParameterSpec(
                type=ParameterType.INT,
                description="Second number",
                augmentation=create_randint_augmentation(-10, 21, "Random integer for second number")
            )
        ],
        outputs=[ParameterSpec(type=ParameterType.INT, description="The sum of the two numbers", augmentation=None)],
        base_examples=[
            Example(input=[3, 5], output=8),
            Example(input=[0, 0], output=0),
            Example(input=[-2, 7], output=5),
            Example(input=[10, -3], output=7),
            Example(input=[-5, -3], output=-8)
        ],
        implementation="""def program(a, b):
    return a + b"""
    ))

    registry.register(ProgramSpecification(
        name="difference",
        description="Calculate the difference between two numbers (a - b)",
        inputs=[
            ParameterSpec(
                type=ParameterType.INT,
                description="First number",
                augmentation=create_randint_augmentation(-10, 21, "Random integer for first number")
            ),
            ParameterSpec(
                type=ParameterType.INT,
                description="Second number",
                augmentation=create_randint_augmentation(-10, 21, "Random integer for second number")
            )
        ],
        outputs=[ParameterSpec(type=ParameterType.INT, description="The difference (a - b, augmentation=None)", augmentation=None)],
        base_examples=[
            Example(input=[8, 3], output=5),
            Example(input=[5, 5], output=0),
            Example(input=[2, 7], output=-5),
            Example(input=[10, -3], output=13),
            Example(input=[-5, -2], output=-3)
        ],
        implementation="""def program(a, b):
    return a - b"""
    ))

    registry.register(ProgramSpecification(
        name="product",
        description="Multiply two numbers together",
        inputs=[
            ParameterSpec(
                type=ParameterType.INT,
                description="First number",
                augmentation=create_randint_augmentation(-8, 9, "Random small integer for first number")
            ),
            ParameterSpec(
                type=ParameterType.INT,
                description="Second number",
                augmentation=create_randint_augmentation(-8, 9, "Random small integer for second number")
            )
        ],
        outputs=[ParameterSpec(type=ParameterType.INT, description="The product of the two numbers", augmentation=None)],
        base_examples=[
            Example(input=[3, 4], output=12),
            Example(input=[0, 5], output=0),
            Example(input=[-2, 3], output=-6),
            Example(input=[7, -2], output=-14),
            Example(input=[-3, -4], output=12)
        ],
        implementation="""def program(a, b):
    return a * b"""
    ))

    # Array Operations
    registry.register(ProgramSpecification(
        name="array_sum",
        description="Sum all elements in an array",
        inputs=[ParameterSpec(
            type=ParameterType.LIST_INT,
            description="Input array",
            augmentation=create_randarray_augmentation(1, 8, -10, 20, "Random array for sum calculation")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Sum of all elements", augmentation=None)],
        base_examples=[
            Example(input=[[1, 2, 3, 4]], output=10),
            Example(input=[[5, -2, 8]], output=11),
            Example(input=[[-1, -2, -3]], output=-6),
            Example(input=[[0, 0, 0]], output=0),
            Example(input=[[42]], output=42)
        ],
        implementation="""def program(arr):
    return sum(arr)"""
    ))

    registry.register(ProgramSpecification(
        name="array_max",
        description="Find maximum element in array",
        inputs=[ParameterSpec(
            type=ParameterType.LIST_INT,
            description="Input array",
            augmentation=create_randarray_augmentation(1, 8, -10, 20, "Random array for max calculation")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Maximum element", augmentation=None)],
        base_examples=[
            Example(input=[[1, 5, 3, 9, 2]], output=9),
            Example(input=[[-1, -5, -2]], output=-1),
            Example(input=[[100]], output=100),
            Example(input=[[0, 0, 1]], output=1),
            Example(input=[[7, 3, 7, 1]], output=7)
        ],
        implementation="""def program(arr):
    return max(arr)"""
    ))

    registry.register(ProgramSpecification(
        name="array_min",
        description="Find minimum element in array",
        inputs=[ParameterSpec(
            type=ParameterType.LIST_INT,
            description="Input array",
            augmentation=create_randarray_augmentation(1, 8, -10, 20, "Random array for min calculation")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Minimum element", augmentation=None)],
        base_examples=[
            Example(input=[[1, 5, 3, 9, 2]], output=1),
            Example(input=[[-1, -5, -2]], output=-5),
            Example(input=[[100]], output=100),
            Example(input=[[0, 0, -1]], output=-1),
            Example(input=[[7, 3, 7, 1]], output=1)
        ],
        implementation="""def program(arr):
    return min(arr)"""
    ))

    registry.register(ProgramSpecification(
        name="array_length",
        description="Get length of array",
        inputs=[ParameterSpec(
            type=ParameterType.LIST_INT,
            description="Input array",
            augmentation=create_randarray_augmentation(0, 10, -5, 5, "Random array for length calculation")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Length of array", augmentation=None)],
        base_examples=[
            Example(input=[[1, 2, 3]], output=3),
            Example(input=[[]], output=0),
            Example(input=[[42]], output=1),
            Example(input=[[1, 2, 3, 4, 5]], output=5),
            Example(input=[[-1, 0, 1, 2]], output=4)
        ],
        implementation="""def program(arr):
    return len(arr)"""
    ))

    registry.register(ProgramSpecification(
        name="array_reverse",
        description="Reverse array elements",
        inputs=[ParameterSpec(
            type=ParameterType.LIST_INT,
            description="Input array",
            augmentation=create_randarray_augmentation(1, 8, -10, 20, "Random array for reverse operation")
        )],
        outputs=[ParameterSpec(type=ParameterType.LIST_INT, description="Reversed array", augmentation=None)],
        base_examples=[
            Example(input=[[1, 2, 3]], output=[3, 2, 1]),
            Example(input=[[5, 4]], output=[4, 5]),
            Example(input=[[42]], output=[42]),
            Example(input=[[]], output=[]),
            Example(input=[[1, 2, 3, 4]], output=[4, 3, 2, 1])
        ],
        implementation="""def program(arr):
    return arr[::-1]"""
    ))

    registry.register(ProgramSpecification(
        name="array_contains",
        description="Check if array contains a specific value",
        inputs=[
            ParameterSpec(
                type=ParameterType.LIST_INT,
                description="Input array",
                augmentation=create_randarray_augmentation(1, 8, -10, 20, "Random array for contains check")
            ),
            ParameterSpec(
                type=ParameterType.INT,
                description="Value to search for",
                augmentation=create_randint_augmentation(-10, 20, "Random value to search for")
            )
        ],
        outputs=[ParameterSpec(type=ParameterType.BOOL, description="True if value found", augmentation=None)],
        base_examples=[
            Example(input=[[1, 2, 3], 2], output=True),
            Example(input=[[1, 2, 3], 5], output=False),
            Example(input=[[], 1], output=False),
            Example(input=[[0, -1, 2], -1], output=True),
            Example(input=[[5, 5, 5], 5], output=True)
        ],
        implementation="""def program(arr, value):
    return value in arr"""
    ))

    registry.register(ProgramSpecification(
        name="array_first",
        description="Get first element of array",
        inputs=[ParameterSpec(
            type=ParameterType.LIST_INT,
            description="Input array",
            augmentation=create_randarray_augmentation(1, 8, -10, 20, "Random array for first element")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="First element", augmentation=None)],
        base_examples=[
            Example(input=[[1, 2, 3]], output=1),
            Example(input=[[42]], output=42),
            Example(input=[[-5, 10, 0]], output=-5),
            Example(input=[[0, 1, 2]], output=0),
            Example(input=[[100, 200]], output=100)
        ],
        implementation="""def program(arr):
    return arr[0]"""
    ))

    registry.register(ProgramSpecification(
        name="array_last",
        description="Get last element of array",
        inputs=[ParameterSpec(
            type=ParameterType.LIST_INT,
            description="Input array",
            augmentation=create_randarray_augmentation(1, 8, -10, 20, "Random array for last element")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Last element", augmentation=None)],
        base_examples=[
            Example(input=[[1, 2, 3]], output=3),
            Example(input=[[42]], output=42),
            Example(input=[[-5, 10, 0]], output=0),
            Example(input=[[0, 1, 2]], output=2),
            Example(input=[[100, 200]], output=200)
        ],
        implementation="""def program(arr):
    return arr[-1]"""
    ))

    registry.register(ProgramSpecification(
        name="array_product",
        description="Multiply all elements in an array",
        inputs=[ParameterSpec(
            type=ParameterType.LIST_INT,
            description="Input array",
            augmentation=create_randarray_augmentation(1, 6, -5, 5, "Random small array for product calculation")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Product of all elements", augmentation=None)],
        base_examples=[
            Example(input=[[1, 2, 3, 4]], output=24),
            Example(input=[[2, 3, 5]], output=30),
            Example(input=[[-2, 3, -1]], output=6),
            Example(input=[[0, 1, 2]], output=0),
            Example(input=[[1, 1, 1, 1]], output=1)
        ],
        implementation="""def program(arr):
    result = 1
    for num in arr:
        result *= num
    return result"""
    ))

    registry.register(ProgramSpecification(
        name="array_count_positive",
        description="Count positive elements in array",
        inputs=[ParameterSpec(
            type=ParameterType.LIST_INT,
            description="Input array",
            augmentation=create_randarray_augmentation(1, 8, -10, 20, "Random array for positive count")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Number of positive elements", augmentation=None)],
        base_examples=[
            Example(input=[[1, -2, 3, -4, 5]], output=3),
            Example(input=[[-1, -2, -3]], output=0),
            Example(input=[[0, 1, 2]], output=2),
            Example(input=[[42]], output=1),
            Example(input=[[0, 0, 0]], output=0)
        ],
        implementation="""def program(arr):
    return sum(1 for x in arr if x > 0)"""
    ))

    registry.register(ProgramSpecification(
        name="array_count_even",
        description="Count even elements in array",
        inputs=[ParameterSpec(
            type=ParameterType.LIST_INT,
            description="Input array",
            augmentation=create_randarray_augmentation(1, 8, -10, 20, "Random array for even count")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Number of even elements", augmentation=None)],
        base_examples=[
            Example(input=[[1, 2, 3, 4, 5]], output=2),
            Example(input=[[1, 3, 5]], output=0),
            Example(input=[[2, 4, 6]], output=3),
            Example(input=[[0, 1, 2]], output=2),
            Example(input=[[-2, -1, 0]], output=2)
        ],
        implementation="""def program(arr):
    return sum(1 for x in arr if x % 2 == 0)"""
    ))

    registry.register(ProgramSpecification(
        name="array_second_largest",
        description="Find second largest element in array",
        inputs=[ParameterSpec(
            type=ParameterType.LIST_INT,
            description="Input array",
            augmentation=create_randarray_augmentation(2, 8, -10, 20, "Random array for second largest")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Second largest element", augmentation=None)],
        base_examples=[
            Example(input=[[1, 5, 3, 9, 2]], output=5),
            Example(input=[[10, 5, 8]], output=8),
            Example(input=[[1, 1, 2, 2]], output=2),
            Example(input=[[-5, -2, -10]], output=-5),
            Example(input=[[100, 50]], output=50)
        ],
        implementation="""def program(arr):
    sorted_arr = sorted(arr, reverse=True)
    return sorted_arr[1]"""
    ))

    registry.register(ProgramSpecification(
        name="array_range",
        description="Find range (max - min) of array",
        inputs=[ParameterSpec(
            type=ParameterType.LIST_INT,
            description="Input array",
            augmentation=create_randarray_augmentation(1, 8, -10, 20, "Random array for range calculation")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Range of array", augmentation=None)],
        base_examples=[
            Example(input=[[1, 5, 3, 9, 2]], output=8),
            Example(input=[[10, 5, 8]], output=5),
            Example(input=[[1, 1, 1]], output=0),
            Example(input=[[-5, -2, -10]], output=8),
            Example(input=[[42]], output=0)
        ],
        implementation="""def program(arr):
    return max(arr) - min(arr)"""
    ))

    registry.register(ProgramSpecification(
        name="array_median",
        description="Find median of array",
        inputs=[ParameterSpec(
            type=ParameterType.LIST_INT,
            description="Input array",
            augmentation=create_randarray_augmentation(1, 8, -10, 20, "Random array for median calculation")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Median of array", augmentation=None)],
        base_examples=[
            Example(input=[[1, 3, 5]], output=3),
            Example(input=[[1, 2, 3, 4]], output=2),
            Example(input=[[10, 5, 8]], output=8),
            Example(input=[[1, 1, 1]], output=1),
            Example(input=[[42]], output=42)
        ],
        implementation="""def program(arr):
    sorted_arr = sorted(arr)
    n = len(sorted_arr)
    if n % 2 == 0:
        return (sorted_arr[n//2 - 1] + sorted_arr[n//2]) // 2
    else:
        return sorted_arr[n//2]"""
    ))

    registry.register(ProgramSpecification(
        name="array_unique_count",
        description="Count unique elements in array",
        inputs=[ParameterSpec(
            type=ParameterType.LIST_INT,
            description="Input array",
            augmentation=create_randarray_augmentation(1, 8, -5, 5, "Random array for unique count")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Number of unique elements", augmentation=None)],
        base_examples=[
            Example(input=[[1, 2, 3, 4]], output=4),
            Example(input=[[1, 1, 2, 2]], output=2),
            Example(input=[[1, 1, 1]], output=1),
            Example(input=[[1, 2, 1, 2]], output=2),
            Example(input=[[42]], output=1)
        ],
        implementation="""def program(arr):
    return len(set(arr))"""
    ))

    # String Operations
    registry.register(ProgramSpecification(
        name="string_length",
        description="Get length of string",
        inputs=[ParameterSpec(
            type=ParameterType.STR,
            description="Input string",
            augmentation=create_randstring_augmentation(0, 10, "Random string for length calculation")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Length of string", augmentation=None)],
        base_examples=[
            Example(input="hello", output=5),
            Example(input="", output=0),
            Example(input="a", output=1),
            Example(input="python", output=6),
            Example(input="12345", output=5)
        ],
        implementation="""def program(s):
    return len(s)"""
    ))

    registry.register(ProgramSpecification(
        name="string_reverse",
        description="Reverse a string",
        inputs=[ParameterSpec(
            type=ParameterType.STR,
            description="Input string",
            augmentation=create_randstring_augmentation(1, 10, "Random string for reverse operation")
        )],
        outputs=[ParameterSpec(type=ParameterType.STR, description="Reversed string", augmentation=None)],
        base_examples=[
            Example(input="hello", output="olleh"),
            Example(input="a", output="a"),
            Example(input="python", output="nohtyp"),
            Example(input="123", output="321"),
            Example(input="", output="")
        ],
        implementation="""def program(s):
    return s[::-1]"""
    ))

    registry.register(ProgramSpecification(
        name="string_uppercase",
        description="Convert string to uppercase",
        inputs=[ParameterSpec(
            type=ParameterType.STR,
            description="Input string",
            augmentation=create_randstring_augmentation(1, 10, "Random string for uppercase conversion")
        )],
        outputs=[ParameterSpec(type=ParameterType.STR, description="Uppercase string", augmentation=None)],
        base_examples=[
            Example(input="hello", output="HELLO"),
            Example(input="Python", output="PYTHON"),
            Example(input="a", output="A"),
            Example(input="123", output="123"),
            Example(input="", output="")
        ],
        implementation="""def program(s):
    return s.upper()"""
    ))

    registry.register(ProgramSpecification(
        name="string_lowercase",
        description="Convert string to lowercase",
        inputs=[ParameterSpec(
            type=ParameterType.STR,
            description="Input string",
            augmentation=create_randstring_augmentation(1, 10, "Random string for lowercase conversion")
        )],
        outputs=[ParameterSpec(type=ParameterType.STR, description="Lowercase string", augmentation=None)],
        base_examples=[
            Example(input="HELLO", output="hello"),
            Example(input="Python", output="python"),
            Example(input="A", output="a"),
            Example(input="123", output="123"),
            Example(input="", output="")
        ],
        implementation="""def program(s):
    return s.lower()"""
    ))

    registry.register(ProgramSpecification(
        name="string_count_vowels",
        description="Count vowels in string",
        inputs=[ParameterSpec(
            type=ParameterType.STR,
            description="Input string",
            augmentation=create_randstring_augmentation(1, 10, "Random string for vowel count")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Number of vowels", augmentation=None)],
        base_examples=[
            Example(input="hello", output=2),
            Example(input="python", output=1),
            Example(input="aeiou", output=5),
            Example(input="xyz", output=0),
            Example(input="a", output=1)
        ],
        implementation="""def program(s):
    vowels = 'aeiouAEIOU'
    return sum(1 for c in s if c in vowels)"""
    ))

    registry.register(ProgramSpecification(
        name="string_palindrome",
        description="Check if string is palindrome",
        inputs=[ParameterSpec(
            type=ParameterType.STR,
            description="Input string",
            augmentation=create_randstring_augmentation(1, 10, "Random string for palindrome check")
        )],
        outputs=[ParameterSpec(type=ParameterType.BOOL, description="True if palindrome", augmentation=None)],
        base_examples=[
            Example(input="racecar", output=True),
            Example(input="hello", output=False),
            Example(input="a", output=True),
            Example(input="", output=True),
            Example(input="anna", output=True)
        ],
        implementation="""def program(s):
    return s == s[::-1]"""
    ))

    registry.register(ProgramSpecification(
        name="string_concat",
        description="Concatenate two strings",
        inputs=[
            ParameterSpec(
                type=ParameterType.STR,
                description="First string",
                augmentation=create_randstring_augmentation(1, 5, "Random first string")
            ),
            ParameterSpec(
                type=ParameterType.STR,
                description="Second string",
                augmentation=create_randstring_augmentation(1, 5, "Random second string")
            )
        ],
        outputs=[ParameterSpec(type=ParameterType.STR, description="Concatenated string", augmentation=None)],
        base_examples=[
            Example(input=["hello", "world"], output="helloworld"),
            Example(input=["a", "b"], output="ab"),
            Example(input=["", "test"], output="test"),
            Example(input=["python", ""], output="python"),
            Example(input=["123", "456"], output="123456")
        ],
        implementation="""def program(s1, s2):
    return s1 + s2"""
    ))

    registry.register(ProgramSpecification(
        name="string_first_char",
        description="Get first character of string",
        inputs=[ParameterSpec(
            type=ParameterType.STR,
            description="Input string",
            augmentation=create_randstring_augmentation(1, 10, "Random string for first character")
        )],
        outputs=[ParameterSpec(type=ParameterType.STR, description="First character", augmentation=None)],
        base_examples=[
            Example(input="hello", output="h"),
            Example(input="a", output="a"),
            Example(input="python", output="p"),
            Example(input="123", output="1"),
            Example(input="", output="")
        ],
        implementation="""def program(s):
    return s[0] if s else ''"""
    ))

    registry.register(ProgramSpecification(
        name="string_last_char",
        description="Get last character of string",
        inputs=[ParameterSpec(
            type=ParameterType.STR,
            description="Input string",
            augmentation=create_randstring_augmentation(1, 10, "Random string for last character")
        )],
        outputs=[ParameterSpec(type=ParameterType.STR, description="Last character", augmentation=None)],
        base_examples=[
            Example(input="hello", output="o"),
            Example(input="a", output="a"),
            Example(input="python", output="n"),
            Example(input="123", output="3"),
            Example(input="", output="")
        ],
        implementation="""def program(s):
    return s[-1] if s else ''"""
    ))

    registry.register(ProgramSpecification(
        name="string_contains",
        description="Check if string contains substring",
        inputs=[
            ParameterSpec(
                type=ParameterType.STR,
                description="Main string",
                augmentation=create_randstring_augmentation(1, 10, "Random main string")
            ),
            ParameterSpec(
                type=ParameterType.STR,
                description="Substring to search for",
                augmentation=create_randstring_augmentation(1, 5, "Random substring")
            )
        ],
        outputs=[ParameterSpec(type=ParameterType.BOOL, description="True if substring found", augmentation=None)],
        base_examples=[
            Example(input=["hello world", "world"], output=True),
            Example(input=["hello world", "python"], output=False),
            Example(input=["test", "t"], output=True),
            Example(input=["", "test"], output=False),
            Example(input=["abc", ""], output=True)
        ],
        implementation="""def program(s, sub):
    return sub in s"""
    ))

    # Mathematical Operations (continued)
    registry.register(ProgramSpecification(
        name="count_up_to_n",
        description="Count from 1 to n",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number N",
            augmentation=create_randint_augmentation(1, 10, "Random small integer for counting")
        )],
        outputs=[ParameterSpec(type=ParameterType.LIST_INT, description="List of numbers from 1 to N", augmentation=None)],
        base_examples=[
            Example(input=5, output=[1, 2, 3, 4, 5]),
            Example(input=3, output=[1, 2, 3]),
            Example(input=1, output=[1]),
            Example(input=0, output=[]),
            Example(input=7, output=[1, 2, 3, 4, 5, 6, 7])
        ],
        implementation="""def program(n):
    return list(range(1, n + 1))"""
    ))

    registry.register(ProgramSpecification(
        name="is_greater_than_ten",
        description="Check if number is greater than 10",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(0, 20, "Random integer for comparison with 10")
        )],
        outputs=[ParameterSpec(type=ParameterType.BOOL, description="True if greater than 10", augmentation=None)],
        base_examples=[
            Example(input=15, output=True),
            Example(input=5, output=False),
            Example(input=10, output=False),
            Example(input=20, output=True),
            Example(input=0, output=False)
        ],
        implementation="""def program(n):
    return n > 10"""
    ))

    registry.register(ProgramSpecification(
        name="remainder_by_three",
        description="Calculate remainder when divided by 3",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(-20, 20, "Random integer for remainder calculation")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Remainder when divided by 3", augmentation=None)],
        base_examples=[
            Example(input=7, output=1),
            Example(input=6, output=0),
            Example(input=8, output=2),
            Example(input=-5, output=1),
            Example(input=0, output=0)
        ],
        implementation="""def program(n):
    return n % 3"""
    ))

    registry.register(ProgramSpecification(
        name="is_divisible_by_five",
        description="Check if number is divisible by 5",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(-20, 20, "Random integer for divisibility check")
        )],
        outputs=[ParameterSpec(type=ParameterType.BOOL, description="True if divisible by 5", augmentation=None)],
        base_examples=[
            Example(input=15, output=True),
            Example(input=7, output=False),
            Example(input=0, output=True),
            Example(input=-10, output=True),
            Example(input=12, output=False)
        ],
        implementation="""def program(n):
    return n % 5 == 0"""
    ))

    registry.register(ProgramSpecification(
        name="triple",
        description="Multiply a number by 3",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(-10, 10, "Random integer for tripling")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="The number multiplied by 3", augmentation=None)],
        base_examples=[
            Example(input=5, output=15),
            Example(input=0, output=0),
            Example(input=-3, output=-9),
            Example(input=10, output=30),
            Example(input=-1, output=-3)
        ],
        implementation="""def program(n):
    return n * 3"""
    ))

    registry.register(ProgramSpecification(
        name="gcd",
        description="Calculate greatest common divisor of two numbers",
        inputs=[
            ParameterSpec(
                type=ParameterType.INT,
                description="First number",
                augmentation=create_randint_augmentation(1, 20, "Random positive integer for GCD")
            ),
            ParameterSpec(
                type=ParameterType.INT,
                description="Second number",
                augmentation=create_randint_augmentation(1, 20, "Random positive integer for GCD")
            )
        ],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Greatest common divisor", augmentation=None)],
        base_examples=[
            Example(input=[12, 18], output=6),
            Example(input=[7, 13], output=1),
            Example(input=[8, 12], output=4),
            Example(input=[15, 25], output=5),
            Example(input=[6, 6], output=6)
        ],
        implementation="""def program(a, b):
    while b:
        a, b = b, a % b
    return a"""
    ))

    registry.register(ProgramSpecification(
        name="lcm",
        description="Calculate least common multiple of two numbers",
        inputs=[
            ParameterSpec(
                type=ParameterType.INT,
                description="First number",
                augmentation=create_randint_augmentation(1, 15, "Random positive integer for LCM")
            ),
            ParameterSpec(
                type=ParameterType.INT,
                description="Second number",
                augmentation=create_randint_augmentation(1, 15, "Random positive integer for LCM")
            )
        ],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Least common multiple", augmentation=None)],
        base_examples=[
            Example(input=[12, 18], output=36),
            Example(input=[4, 6], output=12),
            Example(input=[7, 13], output=91),
            Example(input=[8, 12], output=24),
            Example(input=[5, 5], output=5)
        ],
        implementation="""def program(a, b):
    def gcd(x, y):
        while y:
            x, y = y, x % y
        return x
    return (a * b) // gcd(a, b)"""
    ))

    registry.register(ProgramSpecification(
        name="fibonacci",
        description="Calculate nth Fibonacci number",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number n",
            augmentation=create_randint_augmentation(0, 15, "Random small integer for Fibonacci")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="nth Fibonacci number", augmentation=None)],
        base_examples=[
            Example(input=0, output=0),
            Example(input=1, output=1),
            Example(input=5, output=5),
            Example(input=7, output=13),
            Example(input=10, output=55)
        ],
        implementation="""def program(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b"""
    ))

    registry.register(ProgramSpecification(
        name="prime_check",
        description="Check if number is prime",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(0, 20, "Random integer for prime check")
        )],
        outputs=[ParameterSpec(type=ParameterType.BOOL, description="True if prime", augmentation=None)],
        base_examples=[
            Example(input=7, output=True),
            Example(input=4, output=False),
            Example(input=1, output=False),
            Example(input=13, output=True),
            Example(input=0, output=False)
        ],
        implementation="""def program(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True"""
    ))

    registry.register(ProgramSpecification(
        name="digit_sum",
        description="Sum of digits in a number",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(0, 1000, "Random integer for digit sum")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Sum of digits", augmentation=None)],
        base_examples=[
            Example(input=123, output=6),
            Example(input=456, output=15),
            Example(input=0, output=0),
            Example(input=999, output=27),
            Example(input=10, output=1)
        ],
        implementation="""def program(n):
    return sum(int(digit) for digit in str(abs(n)))"""
    ))

    registry.register(ProgramSpecification(
        name="reverse_number",
        description="Reverse the digits of a number",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(0, 1000, "Random integer for digit reversal")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Reversed number", augmentation=None)],
        base_examples=[
            Example(input=123, output=321),
            Example(input=456, output=654),
            Example(input=0, output=0),
            Example(input=100, output=1),
            Example(input=7, output=7)
        ],
        implementation="""def program(n):
    return int(str(n)[::-1])"""
    ))

    registry.register(ProgramSpecification(
        name="perfect_square",
        description="Check if number is a perfect square",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(0, 100, "Random integer for perfect square check")
        )],
        outputs=[ParameterSpec(type=ParameterType.BOOL, description="True if perfect square", augmentation=None)],
        base_examples=[
            Example(input=16, output=True),
            Example(input=7, output=False),
            Example(input=0, output=True),
            Example(input=25, output=True),
            Example(input=15, output=False)
        ],
        implementation="""def program(n):
    if n < 0:
        return False
    root = int(n**0.5)
    return root * root == n"""
    ))

    registry.register(ProgramSpecification(
        name="factorial_mod",
        description="Calculate factorial modulo a number",
        inputs=[
            ParameterSpec(
                type=ParameterType.INT,
                description="The input number n",
                augmentation=create_randint_augmentation(0, 10, "Random small integer for factorial mod")
            ),
            ParameterSpec(
                type=ParameterType.INT,
                description="The modulo number",
                augmentation=create_randint_augmentation(2, 20, "Random modulo for factorial")
            )
        ],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Factorial modulo", augmentation=None)],
        base_examples=[
            Example(input=[5, 7], output=1),
            Example(input=[4, 5], output=4),
            Example(input=[0, 10], output=1),
            Example(input=[3, 6], output=0),
            Example(input=[6, 13], output=5)
        ],
        implementation="""def program(n, mod):
    if n <= 1:
        return 1
    result = 1
    for i in range(1, n + 1):
        result = (result * i) % mod
    return result"""
    ))

    registry.register(ProgramSpecification(
        name="power_mod",
        description="Calculate power modulo a number",
        inputs=[
            ParameterSpec(
                type=ParameterType.INT,
                description="Base number",
                augmentation=create_randint_augmentation(1, 10, "Random base for power mod")
            ),
            ParameterSpec(
                type=ParameterType.INT,
                description="Exponent",
                augmentation=create_randint_augmentation(1, 10, "Random exponent for power mod")
            ),
            ParameterSpec(
                type=ParameterType.INT,
                description="Modulo number",
                augmentation=create_randint_augmentation(2, 20, "Random modulo for power")
            )
        ],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Power modulo", augmentation=None)],
        base_examples=[
            Example(input=[2, 3, 5], output=3),
            Example(input=[3, 4, 7], output=4),
            Example(input=[5, 2, 10], output=5),
            Example(input=[2, 0, 7], output=1),
            Example(input=[4, 3, 13], output=12)
        ],
        implementation="""def program(base, exp, mod):
    if exp == 0:
        return 1
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp // 2
        base = (base * base) % mod
    return result"""
    ))

    registry.register(ProgramSpecification(
        name="divisor_count",
        description="Count number of divisors of a number",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(1, 50, "Random positive integer for divisor count")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Number of divisors", augmentation=None)],
        base_examples=[
            Example(input=12, output=6),
            Example(input=7, output=2),
            Example(input=16, output=5),
            Example(input=1, output=1),
            Example(input=25, output=3)
        ],
        implementation="""def program(n):
    count = 0
    for i in range(1, n + 1):
        if n % i == 0:
            count += 1
    return count"""
    ))

    registry.register(ProgramSpecification(
        name="collatz_steps",
        description="Count steps in Collatz sequence",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(1, 20, "Random positive integer for Collatz sequence")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Number of steps", augmentation=None)],
        base_examples=[
            Example(input=6, output=8),
            Example(input=1, output=0),
            Example(input=3, output=7),
            Example(input=10, output=6),
            Example(input=16, output=4)
        ],
        implementation="""def program(n):
    steps = 0
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        steps += 1
    return steps"""
    ))

    registry.register(ProgramSpecification(
        name="sum_of_squares",
        description="Sum of squares of numbers from 1 to n",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number n",
            augmentation=create_randint_augmentation(1, 10, "Random small integer for sum of squares")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Sum of squares", augmentation=None)],
        base_examples=[
            Example(input=3, output=14),
            Example(input=1, output=1),
            Example(input=4, output=30),
            Example(input=5, output=55),
            Example(input=2, output=5)
        ],
        implementation="""def program(n):
    return sum(i*i for i in range(1, n + 1))"""
    ))

    registry.register(ProgramSpecification(
        name="binomial_coeff",
        description="Calculate binomial coefficient C(n,k)",
        inputs=[
            ParameterSpec(
                type=ParameterType.INT,
                description="n value",
                augmentation=create_randint_augmentation(0, 10, "Random n for binomial coefficient")
            ),
            ParameterSpec(
                type=ParameterType.INT,
                description="k value",
                augmentation=create_randint_augmentation(0, 10, "Random k for binomial coefficient")
            )
        ],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Binomial coefficient", augmentation=None)],
        base_examples=[
            Example(input=[5, 2], output=10),
            Example(input=[4, 1], output=4),
            Example(input=[6, 3], output=20),
            Example(input=[0, 0], output=1),
            Example(input=[3, 3], output=1)
        ],
        implementation="""def program(n, k):
    if k > n:
        return 0
    if k == 0 or k == n:
        return 1
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result"""
    ))

    registry.register(ProgramSpecification(
        name="next_prime",
        description="Find next prime number after n",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(0, 20, "Random integer for next prime")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Next prime number", augmentation=None)],
        base_examples=[
            Example(input=10, output=11),
            Example(input=1, output=2),
            Example(input=7, output=11),
            Example(input=0, output=2),
            Example(input=15, output=17)
        ],
        implementation="""def program(n):
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0:
                return False
        return True

    candidate = n + 1
    while not is_prime(candidate):
        candidate += 1
    return candidate"""
    ))

    registry.register(ProgramSpecification(
        name="sum_divisors",
        description="Sum of all divisors of a number",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(1, 30, "Random positive integer for sum of divisors")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Sum of divisors", augmentation=None)],
        base_examples=[
            Example(input=12, output=28),
            Example(input=6, output=12),
            Example(input=1, output=1),
            Example(input=8, output=15),
            Example(input=16, output=31)
        ],
        implementation="""def program(n):
    total = 0
    for i in range(1, n + 1):
        if n % i == 0:
            total += i
    return total"""
    ))

    registry.register(ProgramSpecification(
        name="digit_product",
        description="Product of digits in a number",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(0, 1000, "Random integer for digit product")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Product of digits", augmentation=None)],
        base_examples=[
            Example(input=123, output=6),
            Example(input=456, output=120),
            Example(input=0, output=0),
            Example(input=999, output=729),
            Example(input=10, output=0)
        ],
        implementation="""def program(n):
    if n == 0:
        return 0
    product = 1
    for digit in str(abs(n)):
        product *= int(digit)
    return product"""
    ))

    registry.register(ProgramSpecification(
        name="is_perfect",
        description="Check if number is perfect (sum of proper divisors equals number)",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(1, 30, "Random positive integer for perfect number check")
        )],
        outputs=[ParameterSpec(type=ParameterType.BOOL, description="True if perfect number", augmentation=None)],
        base_examples=[
            Example(input=6, output=True),
            Example(input=28, output=True),
            Example(input=12, output=False),
            Example(input=1, output=False),
            Example(input=8, output=False)
        ],
        implementation="""def program(n):
    if n <= 1:
        return False
    total = 0
    for i in range(1, n):
        if n % i == 0:
            total += i
    return total == n"""
    ))

    registry.register(ProgramSpecification(
        name="greatest_digit",
        description="Find greatest digit in a number",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(0, 1000, "Random integer for greatest digit")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Greatest digit", augmentation=None)],
        base_examples=[
            Example(input=123, output=3),
            Example(input=456, output=6),
            Example(input=0, output=0),
            Example(input=999, output=9),
            Example(input=10, output=1)
        ],
        implementation="""def program(n):
    return max(int(digit) for digit in str(abs(n)))"""
    ))

    # Comparison Operations
    registry.register(ProgramSpecification(
        name="max_of_three",
        description="Return the largest of three numbers",
        inputs=[
            ParameterSpec(
                type=ParameterType.INT,
                description="First number",
                augmentation=create_randint_augmentation(-10, 20, "Random integer for first number")
            ),
            ParameterSpec(
                type=ParameterType.INT,
                description="Second number",
                augmentation=create_randint_augmentation(-10, 20, "Random integer for second number")
            ),
            ParameterSpec(
                type=ParameterType.INT,
                description="Third number",
                augmentation=create_randint_augmentation(-10, 20, "Random integer for third number")
            )
        ],
        outputs=[ParameterSpec(type=ParameterType.INT, description="The largest number", augmentation=None)],
        base_examples=[
            Example(input=[5, 3, 9], output=9),
            Example(input=[10, 15, 8], output=15),
            Example(input=[-2, -5, -1], output=-1),
            Example(input=[0, 0, 0], output=0),
            Example(input=[7, 7, 7], output=7)
        ],
        implementation="""def program(a, b, c):
    return max(a, b, c)"""
    ))

    registry.register(ProgramSpecification(
        name="min_of_three",
        description="Return the smallest of three numbers",
        inputs=[
            ParameterSpec(
                type=ParameterType.INT,
                description="First number",
                augmentation=create_randint_augmentation(-10, 20, "Random integer for first number")
            ),
            ParameterSpec(
                type=ParameterType.INT,
                description="Second number",
                augmentation=create_randint_augmentation(-10, 20, "Random integer for second number")
            ),
            ParameterSpec(
                type=ParameterType.INT,
                description="Third number",
                augmentation=create_randint_augmentation(-10, 20, "Random integer for third number")
            )
        ],
        outputs=[ParameterSpec(type=ParameterType.INT, description="The smallest number", augmentation=None)],
        base_examples=[
            Example(input=[5, 3, 9], output=3),
            Example(input=[10, 15, 8], output=8),
            Example(input=[-2, -5, -1], output=-5),
            Example(input=[0, 0, 0], output=0),
            Example(input=[7, 7, 7], output=7)
        ],
        implementation="""def program(a, b, c):
    return min(a, b, c)"""
    ))

    # Classification Operations
    registry.register(ProgramSpecification(
        name="grade_classifier",
        description="Classify grade based on score",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The score (0-100)",
            augmentation=create_randint_augmentation(0, 100, "Random score for grade classification")
        )],
        outputs=[ParameterSpec(type=ParameterType.STR, description="Grade classification", augmentation=None)],
        base_examples=[
            Example(input=95, output="A"),
            Example(input=85, output="B"),
            Example(input=75, output="C"),
            Example(input=65, output="D"),
            Example(input=55, output="F")
        ],
        implementation="""def program(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'"""
    ))

    registry.register(ProgramSpecification(
        name="sign_function",
        description="Return sign of a number (-1, 0, 1)",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(-20, 20, "Random integer for sign function")
        )],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Sign of number", augmentation=None)],
        base_examples=[
            Example(input=5, output=1),
            Example(input=-3, output=-1),
            Example(input=0, output=0),
            Example(input=10, output=1),
            Example(input=-15, output=-1)
        ],
        implementation="""def program(n):
    if n > 0:
        return 1
    elif n < 0:
        return -1
    else:
        return 0"""
    ))

    registry.register(ProgramSpecification(
        name="triangle_type",
        description="Classify triangle type based on side lengths",
        inputs=[
            ParameterSpec(
                type=ParameterType.INT,
                description="First side length",
                augmentation=create_randint_augmentation(1, 10, "Random first side length")
            ),
            ParameterSpec(
                type=ParameterType.INT,
                description="Second side length",
                augmentation=create_randint_augmentation(1, 10, "Random second side length")
            ),
            ParameterSpec(
                type=ParameterType.INT,
                description="Third side length",
                augmentation=create_randint_augmentation(1, 10, "Random third side length")
            )
        ],
        outputs=[ParameterSpec(type=ParameterType.STR, description="Triangle type", augmentation=None)],
        base_examples=[
            Example(input=[3, 4, 5], output="right"),
            Example(input=[3, 3, 3], output="equilateral"),
            Example(input=[3, 3, 4], output="isosceles"),
            Example(input=[3, 4, 6], output="scalene"),
            Example(input=[1, 2, 3], output="invalid")
        ],
        implementation="""def program(a, b, c):
    sides = sorted([a, b, c])
    if sides[0] + sides[1] <= sides[2]:
        return 'invalid'
    elif a == b == c:
        return 'equilateral'
    elif a == b or b == c or a == c:
        return 'isosceles'
    elif a*a + b*b == c*c or a*a + c*c == b*b or b*b + c*c == a*a:
        return 'right'
    else:
        return 'scalene'"""
    ))

    registry.register(ProgramSpecification(
        name="leap_year",
        description="Check if year is a leap year",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The year",
            augmentation=create_randint_augmentation(1900, 2100, "Random year for leap year check")
        )],
        outputs=[ParameterSpec(type=ParameterType.BOOL, description="True if leap year", augmentation=None)],
        base_examples=[
            Example(input=2000, output=True),
            Example(input=2020, output=True),
            Example(input=2021, output=False),
            Example(input=1900, output=False),
            Example(input=1600, output=True)
        ],
        implementation="""def program(year):
    if year % 4 != 0:
        return False
    elif year % 100 != 0:
        return True
    elif year % 400 != 0:
        return False
    else:
        return True"""
    ))

    registry.register(ProgramSpecification(
        name="quadrant",
        description="Determine quadrant of point (x,y)",
        inputs=[
            ParameterSpec(
                type=ParameterType.INT,
                description="x coordinate",
                augmentation=create_randint_augmentation(-10, 10, "Random x coordinate")
            ),
            ParameterSpec(
                type=ParameterType.INT,
                description="y coordinate",
                augmentation=create_randint_augmentation(-10, 10, "Random y coordinate")
            )
        ],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Quadrant number", augmentation=None)],
        base_examples=[
            Example(input=[3, 4], output=1),
            Example(input=[-2, 5], output=2),
            Example(input=[-3, -4], output=3),
            Example(input=[2, -3], output=4),
            Example(input=[0, 0], output=0)
        ],
        implementation="""def program(x, y):
    if x == 0 and y == 0:
        return 0
    elif x > 0 and y > 0:
        return 1
    elif x < 0 and y > 0:
        return 2
    elif x < 0 and y < 0:
        return 3
    else:
        return 4"""
    ))

    registry.register(ProgramSpecification(
        name="number_category",
        description="Categorize number as positive, negative, or zero",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="The input number",
            augmentation=create_randint_augmentation(-10, 10, "Random integer for number categorization")
        )],
        outputs=[ParameterSpec(type=ParameterType.STR, description="Number category", augmentation=None)],
        base_examples=[
            Example(input=5, output="positive"),
            Example(input=-3, output="negative"),
            Example(input=0, output="zero"),
            Example(input=10, output="positive"),
            Example(input=-15, output="negative")
        ],
        implementation="""def program(n):
    if n > 0:
        return 'positive'
    elif n < 0:
        return 'negative'
    else:
        return 'zero'"""
    ))

    registry.register(ProgramSpecification(
        name="compare_strings",
        description="Compare two strings lexicographically",
        inputs=[
            ParameterSpec(
                type=ParameterType.STR,
                description="First string",
                augmentation=create_randstring_augmentation(1, 5, "Random first string")
            ),
            ParameterSpec(
                type=ParameterType.STR,
                description="Second string",
                augmentation=create_randstring_augmentation(1, 5, "Random second string")
            )
        ],
        outputs=[ParameterSpec(type=ParameterType.INT, description="Comparison result", augmentation=None)],
        base_examples=[
            Example(input=["apple", "banana"], output=-1),
            Example(input=["banana", "apple"], output=1),
            Example(input=["hello", "hello"], output=0),
            Example(input=["a", "b"], output=-1),
            Example(input=["z", "a"], output=1)
        ],
        implementation="""def program(s1, s2):
    if s1 < s2:
        return -1
    elif s1 > s2:
        return 1
    else:
        return 0"""
    ))

    registry.register(ProgramSpecification(
        name="season_from_month",
        description="Determine season from month number",
        inputs=[ParameterSpec(
            type=ParameterType.INT,
            description="Month number (1-12)",
            augmentation=create_randint_augmentation(1, 12, "Random month for season determination")
        )],
        outputs=[ParameterSpec(type=ParameterType.STR, description="Season name", augmentation=None)],
        base_examples=[
            Example(input=3, output="spring"),
            Example(input=6, output="summer"),
            Example(input=9, output="fall"),
            Example(input=12, output="winter"),
            Example(input=1, output="winter")
        ],
        implementation="""def program(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'fall'"""
    ))

    return registry


# Global registry instance
DEFAULT_REGISTRY = create_default_registry()


def get_program_registry() -> ProgramRegistry:
    """Get the default program registry"""
    return DEFAULT_REGISTRY


def load_programs_from_yaml(yaml_file: str) -> ProgramRegistry:
    """Load program specifications from a YAML file"""
    import yaml

    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    registry = ProgramRegistry()

    for program_data in data.get('programs', []):
        # Convert YAML data to ProgramSpecification
        spec = ProgramSpecification(**program_data)
        registry.register(spec)

    return registry


def save_programs_to_yaml(registry: ProgramRegistry, yaml_file: str):
    """Save program specifications to a YAML file"""
    import yaml

    data = {
        'programs': [spec.model_dump() for spec in registry.programs.values()]
    }

    with open(yaml_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, indent=2)


def test_all_programs(registry: Optional[ProgramRegistry] = None) -> Dict[str, bool]:
    """Test all programs in the registry to ensure they produce correct outputs"""
    if registry is None:
        registry = get_program_registry()

    results = {}

    for name, spec in registry.programs.items():
        print(f"Testing program: {name}")
        success = True

        try:
            # Test base examples
            for i, example in enumerate(spec.base_examples):
                # Prepare input values
                if isinstance(example.input, list):
                    input_values = example.input
                else:
                    input_values = [example.input]

                # Execute the program
                actual_output = spec._execute_implementation(input_values)
                expected_output = example.output

                # Compare outputs
                if actual_output != expected_output:
                    print(f"  ❌ Example {i+1}: Expected {expected_output}, got {actual_output}")
                    print(f"     Input: {example.input}")
                    success = False
                else:
                    print(f"  ✅ Example {i+1}: {example.input} -> {actual_output}")

            # Test a few generated examples
            generated_examples = spec.generate_examples(3, seed=42)
            for i, example in enumerate(generated_examples):
                # Prepare input values
                if isinstance(example.input, list):
                    input_values = example.input
                else:
                    input_values = [example.input]

                # Execute the program
                actual_output = spec._execute_implementation(input_values)
                expected_output = example.output

                # Compare outputs
                if actual_output != expected_output:
                    print(f"  ❌ Generated {i+1}: Expected {expected_output}, got {actual_output}")
                    print(f"     Input: {example.input}")
                    success = False
                else:
                    print(f"  ✅ Generated {i+1}: {example.input} -> {actual_output}")

        except Exception as e:
            print(f"  ❌ Error testing {name}: {e}")
            success = False

        results[name] = success
        print(f"  {'✅ PASSED' if success else '❌ FAILED'}: {name}")
        print()

    # Summary
    passed = sum(results.values())
    total = len(results)
    print(f"Summary: {passed}/{total} programs passed")

    return results
