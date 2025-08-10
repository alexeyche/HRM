import ast
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
from collections import defaultdict


class ASTNodeType(Enum):
    """Enumeration of AST node types for minimal language graph neural network modeling"""

    # Structural
    MODULE = "module"               # Python module root
    FUNCTION_DEF = "function_def"    # Function definition
    RETURN = "return"               # Return statement

    # Control Flow
    FOR = "for"                     # For loop
    WHILE = "while"                 # While loop
    IF = "if"                       # If statement

    # Variables and Names
    VARIABLE = "variable"           # Variable reference (Name node)
    VARIABLE_SYMBOL = "variable_symbol"  # Unique symbol node per distinct variable
    ASSIGNMENT = "assignment"       # Assignment statement
    AUGMENTED_ASSIGNMENT = "augmented_assignment"  # +=, -=, etc.

    # Expressions
    BINARY_OPERATION = "binary_operation"  # +, -, *, /, %, **, etc.
    UNARY_OPERATION = "unary_operation"    # +, -, not
    COMPARISON = "comparison"       # ==, !=, <, <=, >, >=
    BOOLEAN_OPERATION = "boolean_operation"  # and, or
    FUNCTION_CALL = "function_call" # Function call
    ATTRIBUTE = "attribute"         # Object attribute access
    SUBSCRIPT = "subscript"         # Indexing/slicing
    SLICE = "slice"                 # Slice operation

    # Data Structures
    LIST = "list"                   # List literal

    # Literals
    CONSTANT = "constant"           # Literal values (numbers, strings, etc.)
    EXPRESSION = "expression"       # Expression wrapper


class EdgeType(Enum):
    """Edge types for AST graph representation"""

    AST = "ast"                    # Parent -> child structural edge
    SYMBOL = "symbol"              # Variable occurrence -> variable symbol node
    NEXT_SIBLING = "next_sibling"  # Sibling order edge


class ASTGraph:
    """Container for nodes and edges in the simplified AST graph"""

    def __init__(self) -> None:
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Tuple[int, int, EdgeType]] = []

    def add_node(self, node_type: ASTNodeType, **attrs: Any) -> int:
        idx = len(self.nodes)
        self.nodes.append({"type": node_type, **attrs})
        return idx

    def add_edge(self, src: int, dst: int, edge_type: EdgeType) -> None:
        self.edges.append((src, dst, edge_type))


class ASTSimplifier:
    """Converts Python AST to simplified graph representation"""

    @staticmethod
    def python_to_ast(code: str) -> ast.AST:
        """Parse Python code to AST"""
        return ast.parse(code)

    @staticmethod
    def ast_to_simplified_json(code: str) -> Dict[str, Any]:
        """Convert Python code to human-readable JSON AST"""
        tree = ASTSimplifier.python_to_ast(code)

        def traverse_to_readable(node) -> Any:
            """Convert AST node to human-readable format"""
            if isinstance(node, ast.FunctionDef):
                return {
                    "function": {
                        "name": node.name,
                        "params": [arg.arg for arg in node.args.args],
                        "body": [traverse_to_readable(stmt) for stmt in node.body]
                    }
                }
            elif isinstance(node, ast.Return):
                return {"return": traverse_to_readable(node.value)}
            elif isinstance(node, ast.For):
                return {
                    "for": {
                        "target": traverse_to_readable(node.target),
                        "iter": traverse_to_readable(node.iter),
                        "body": [traverse_to_readable(stmt) for stmt in node.body]
                    }
                }
            elif isinstance(node, ast.While):
                return {
                    "while": {
                        "test": traverse_to_readable(node.test),
                        "body": [traverse_to_readable(stmt) for stmt in node.body]
                    }
                }
            elif isinstance(node, ast.If):
                result = {
                    "if": {
                        "test": traverse_to_readable(node.test),
                        "body": [traverse_to_readable(stmt) for stmt in node.body]
                    }
                }
                if node.orelse:
                    result["if"]["else"] = [traverse_to_readable(stmt) for stmt in node.orelse]
                return result
            elif isinstance(node, ast.Name):
                return {"var": node.id}
            elif isinstance(node, ast.BinOp):
                op_map: Dict[Any, str] = {
                    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
                    ast.Mod: "%", ast.Pow: "**"
                }
                op_name = op_map.get(type(node.op), str(type(node.op).__name__))
                return {op_name: [traverse_to_readable(node.left), traverse_to_readable(node.right)]}
            elif isinstance(node, ast.UnaryOp):
                op_map = {ast.USub: "-", ast.UAdd: "+", ast.Not: "not"}
                op_name = op_map.get(type(node.op), str(type(node.op).__name__))
                return {op_name: traverse_to_readable(node.operand)}
            elif isinstance(node, ast.Compare):
                op_map = {
                    ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
                    ast.Gt: ">", ast.GtE: ">="
                }
                if len(node.ops) == 1 and len(node.comparators) == 1:
                    op_name = op_map.get(type(node.ops[0]), str(type(node.ops[0]).__name__))
                    return {op_name: [traverse_to_readable(node.left), traverse_to_readable(node.comparators[0])]}
                else:
                    # Multiple comparisons
                    return {
                        "compare": {
                            "left": traverse_to_readable(node.left),
                            "ops": [op_map.get(type(op), str(type(op).__name__)) for op in node.ops],
                            "comparators": [traverse_to_readable(comp) for comp in node.comparators]
                        }
                    }
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    return {
                        "call": {
                            "function": node.func.id,
                            "args": [traverse_to_readable(arg) for arg in node.args]
                        }
                    }
                else:
                    return {
                        "call": {
                            "function": traverse_to_readable(node.func),
                            "args": [traverse_to_readable(arg) for arg in node.args]
                        }
                    }
            elif isinstance(node, ast.Constant):
                return {"const": node.value}
            elif isinstance(node, ast.List):
                return {"list": [traverse_to_readable(elt) for elt in node.elts]}
            elif isinstance(node, ast.Tuple):
                return {"list": [traverse_to_readable(elt) for elt in node.elts]}
            elif isinstance(node, ast.AugAssign):
                op_map = {
                    ast.Add: "+=", ast.Sub: "-=", ast.Mult: "*=", ast.Div: "/=",
                    ast.Mod: "%=", ast.Pow: "**="
                }
                op_name = op_map.get(type(node.op), str(type(node.op).__name__))
                return {
                    "aug_assign": {
                        "target": traverse_to_readable(node.target),
                        "op": op_name,
                        "value": traverse_to_readable(node.value)
                    }
                }
            elif isinstance(node, ast.Assign):
                return {"assign": [traverse_to_readable(elt) for elt in node.targets]}
            elif isinstance(node, ast.Expr):
                return {"expr": traverse_to_readable(node.value)}
            elif isinstance(node, ast.Attribute):
                return {
                    "attribute": {
                        "value": traverse_to_readable(node.value),
                        "attr": node.attr
                    }
                }
            elif isinstance(node, ast.Subscript):
                return {
                    "subscript": {
                        "value": traverse_to_readable(node.value),
                        "slice": traverse_to_readable(node.slice)
                    }
                }
            elif isinstance(node, ast.Slice):
                return {
                    "slice": {
                        "lower": traverse_to_readable(node.lower) if node.lower else None,
                        "upper": traverse_to_readable(node.upper) if node.upper else None,
                        "step": traverse_to_readable(node.step) if node.step else None
                    }
                }
            elif isinstance(node, ast.BoolOp):
                op_map: Dict[Any, str] = {ast.And: "and", ast.Or: "or"}
                op_name = op_map.get(type(node.op), str(type(node.op).__name__))
                return {
                    "bool_op": {
                        "op": op_name,
                        "values": [traverse_to_readable(val) for val in node.values]
                    }
                }
            else:
                raise ValueError(f"Unsupported node type: {type(node)}")

        # Start traversal from the function definition
        result = traverse_to_readable(tree.body[0])  # type: ignore

        return result

    @staticmethod
    def ast_to_graph(code: str) -> Dict[str, Any]:
        """Convert Python code to a GNN-friendly AST graph using enum node types.

        Returns a dictionary with:
          - nodes: List[Dict] each containing 'type': ASTNodeType and optional attributes
          - edges: List[Tuple[int, int, EdgeType]]
          - root: index of the root node
        """
        py_tree = ASTSimplifier.python_to_ast(code)

        graph = ASTGraph()
        children_by_parent: Dict[int, List[int]] = defaultdict(list)

        # Canonicalize variables to per-program IDs
        variable_to_id: Dict[str, int] = {}

        class VariableCollector(ast.NodeVisitor):
            def visit_Name(self, node: ast.Name) -> None:  # type: ignore[override]
                if node.id not in variable_to_id:
                    variable_to_id[node.id] = len(variable_to_id) + 1  # 0 reserved for UNK

        VariableCollector().visit(py_tree)

        # Create variable symbol nodes (unique per distinct variable)
        symbol_node_index: Dict[str, int] = {
            name: graph.add_node(ASTNodeType.VARIABLE_SYMBOL, name=name, var_id=vid)
            for name, vid in variable_to_id.items()
        }

        # Operator mappings
        binop_map: Dict[Any, str] = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Mod: "%", ast.Pow: "**"}
        unaryop_map: Dict[Any, str] = {ast.USub: "-", ast.UAdd: "+", ast.Not: "not"}
        cmpop_map: Dict[Any, str] = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
        }
        boolop_map: Dict[Any, str] = {ast.And: "and", ast.Or: "or"}

        def link(parent_idx: int, child_idx: int) -> None:
            graph.add_edge(parent_idx, child_idx, EdgeType.AST)
            children_by_parent[parent_idx].append(child_idx)

        def add(node: ast.AST, parent: Optional[int] = None) -> int:
            # Structural roots
            if isinstance(node, ast.Module):
                idx = graph.add_node(ASTNodeType.MODULE)
                for stmt in node.body:
                    cidx = add(stmt, idx)
                    link(idx, cidx)
                return idx

            if isinstance(node, ast.FunctionDef):
                idx = graph.add_node(
                    ASTNodeType.FUNCTION_DEF,
                    name=node.name,
                    params=[arg.arg for arg in node.args.args],
                )
                for stmt in node.body:
                    cidx = add(stmt, idx)
                    link(idx, cidx)
                return idx

            if isinstance(node, ast.Return):
                idx = graph.add_node(ASTNodeType.RETURN)
                if node.value is not None:
                    v = add(node.value, idx)
                    link(idx, v)
                return idx

            # Control flow
            if isinstance(node, ast.For):
                idx = graph.add_node(ASTNodeType.FOR)
                t = add(node.target, idx)
                i = add(node.iter, idx)
                link(idx, t)
                link(idx, i)
                for stmt in node.body:
                    cidx = add(stmt, idx)
                    link(idx, cidx)
                for stmt in node.orelse or []:
                    cidx = add(stmt, idx)
                    link(idx, cidx)
                return idx

            if isinstance(node, ast.While):
                idx = graph.add_node(ASTNodeType.WHILE)
                test_idx = add(node.test, idx)
                link(idx, test_idx)
                for stmt in node.body:
                    cidx = add(stmt, idx)
                    link(idx, cidx)
                for stmt in node.orelse or []:
                    cidx = add(stmt, idx)
                    link(idx, cidx)
                return idx

            if isinstance(node, ast.If):
                idx = graph.add_node(ASTNodeType.IF)
                test_idx = add(node.test, idx)
                link(idx, test_idx)
                for stmt in node.body:
                    cidx = add(stmt, idx)
                    link(idx, cidx)
                for stmt in node.orelse or []:
                    cidx = add(stmt, idx)
                    link(idx, cidx)
                return idx

            # Variables and assignments
            if isinstance(node, ast.Name):
                var_id = variable_to_id.get(node.id, 0)
                idx = graph.add_node(
                    ASTNodeType.VARIABLE,
                    name=node.id,
                    var_id=var_id,
                    ctx=type(node.ctx).__name__,
                )
                # Link occurrence to its symbol node
                if node.id in symbol_node_index:
                    graph.add_edge(idx, symbol_node_index[node.id], EdgeType.SYMBOL)
                return idx

            if isinstance(node, ast.Assign):
                idx = graph.add_node(ASTNodeType.ASSIGNMENT)
                for tgt in node.targets:
                    t = add(tgt, idx)
                    link(idx, t)
                val = add(node.value, idx)
                link(idx, val)
                return idx

            if isinstance(node, ast.AugAssign):
                idx = graph.add_node(ASTNodeType.AUGMENTED_ASSIGNMENT, op=binop_map.get(type(node.op), type(node.op).__name__))
                t = add(node.target, idx)
                v = add(node.value, idx)
                link(idx, t)
                link(idx, v)
                return idx

            # Expressions
            if isinstance(node, ast.BinOp):
                idx = graph.add_node(ASTNodeType.BINARY_OPERATION, op=binop_map.get(type(node.op), type(node.op).__name__))
                l = add(node.left, idx)
                r = add(node.right, idx)
                link(idx, l)
                link(idx, r)
                return idx

            if isinstance(node, ast.UnaryOp):
                idx = graph.add_node(ASTNodeType.UNARY_OPERATION, op=unaryop_map.get(type(node.op), type(node.op).__name__))
                o = add(node.operand, idx)
                link(idx, o)
                return idx

            if isinstance(node, ast.Compare):
                if len(node.ops) == 1 and len(node.comparators) == 1:
                    op_name = cmpop_map.get(type(node.ops[0]), type(node.ops[0]).__name__)
                    idx = graph.add_node(ASTNodeType.COMPARISON, op=op_name)
                    l = add(node.left, idx)
                    r = add(node.comparators[0], idx)
                    link(idx, l)
                    link(idx, r)
                    return idx
                else:
                    ops = [cmpop_map.get(type(op), type(op).__name__) for op in node.ops]
                    idx = graph.add_node(ASTNodeType.COMPARISON, ops=ops)
                    l = add(node.left, idx)
                    link(idx, l)
                    for comp in node.comparators:
                        c = add(comp, idx)
                        link(idx, c)
                    return idx

            if isinstance(node, ast.BoolOp):
                op_name = boolop_map.get(type(node.op), type(node.op).__name__)
                idx = graph.add_node(ASTNodeType.BOOLEAN_OPERATION, op=op_name)
                for val in node.values:
                    v = add(val, idx)
                    link(idx, v)
                return idx

            if isinstance(node, ast.Call):
                func_name: Optional[str] = node.func.id if isinstance(node.func, ast.Name) else None  # type: ignore[attr-defined]
                idx = graph.add_node(ASTNodeType.FUNCTION_CALL, function=func_name)
                f = add(node.func, idx)
                link(idx, f)
                for arg in node.args:
                    a = add(arg, idx)
                    link(idx, a)
                return idx

            if isinstance(node, ast.Attribute):
                idx = graph.add_node(ASTNodeType.ATTRIBUTE, attr=node.attr)
                v = add(node.value, idx)
                link(idx, v)
                return idx

            if isinstance(node, ast.Subscript):
                idx = graph.add_node(ASTNodeType.SUBSCRIPT)
                v = add(node.value, idx)
                s = add(node.slice, idx)
                link(idx, v)
                link(idx, s)
                return idx

            if isinstance(node, ast.Slice):
                idx = graph.add_node(ASTNodeType.SLICE)
                if node.lower is not None:
                    l = add(node.lower, idx)
                    link(idx, l)
                if node.upper is not None:
                    u = add(node.upper, idx)
                    link(idx, u)
                if node.step is not None:
                    st = add(node.step, idx)
                    link(idx, st)
                return idx

            if isinstance(node, ast.Constant):
                val = node.value
                if isinstance(val, bool):
                    idx = graph.add_node(ASTNodeType.CONSTANT, dtype="bool", value=bool(val))
                elif isinstance(val, int):
                    idx = graph.add_node(ASTNodeType.CONSTANT, dtype="int", value=int(val))
                elif isinstance(val, float):
                    idx = graph.add_node(ASTNodeType.CONSTANT, dtype="float", value=float(val))
                elif isinstance(val, str):
                    idx = graph.add_node(ASTNodeType.CONSTANT, dtype="str", value=str(val))
                else:
                    idx = graph.add_node(ASTNodeType.CONSTANT, dtype=type(val).__name__, value=str(val))
                return idx

            if isinstance(node, ast.List):
                idx = graph.add_node(ASTNodeType.LIST)
                for elt in node.elts:
                    e = add(elt, idx)
                    link(idx, e)
                return idx

            if isinstance(node, ast.Tuple):
                idx = graph.add_node(ASTNodeType.LIST)
                for elt in node.elts:
                    e = add(elt, idx)
                    link(idx, e)
                return idx

            if isinstance(node, ast.Expr):
                idx = graph.add_node(ASTNodeType.EXPRESSION)
                v = add(node.value, idx)
                link(idx, v)
                return idx

            # Nodes we intentionally skip or treat as no-ops (e.g., Pass, Break, Continue)
            if isinstance(node, (ast.Pass, ast.Break, ast.Continue)):
                # Represent as an empty expression wrapper for simplicity
                idx = graph.add_node(ASTNodeType.EXPRESSION)
                return idx

            # Fallback: treat unhandled nodes as expression wrappers around their children
            idx = graph.add_node(ASTNodeType.EXPRESSION)
            for child in ast.iter_child_nodes(node):
                c = add(child, idx)
                link(idx, c)
            return idx

        root_index = add(py_tree)

        # Add next_sibling edges per parent to encode order
        for parent, children in children_by_parent.items():
            for a, b in zip(children, children[1:]):
                graph.add_edge(a, b, EdgeType.NEXT_SIBLING)

        return {"nodes": graph.nodes, "edges": graph.edges, "root": root_index}

    @staticmethod
    def _build_children_lists(num_nodes: int, edges: List[Dict[str, Any]]) -> Dict[int, List[int]]:
        """Build a map from parent node index to list of child node indices."""
        children: Dict[int, List[int]] = {}
        next_siblings: Dict[int, List[int]] = {}

        for i in range(num_nodes):
            children[i] = []
            next_siblings[i] = []

        for edge in edges:
            # Support both dict-form and tuple-form edges
            if isinstance(edge, tuple) and len(edge) == 3:
                src, dst, et = edge
                edge_type = et.value if isinstance(et, Enum) else et
            else:
                src = edge["src"]
                dst = edge["dst"]
                et = edge["edge_type"]
                edge_type = et.value if isinstance(et, Enum) else et

            if edge_type == EdgeType.AST.value:
                children[src].append(dst)
            elif edge_type == EdgeType.NEXT_SIBLING.value:
                next_siblings[src].append(dst)

        # Sort children by NEXT_SIBLING order
        for parent_idx, child_list in children.items():
            if len(child_list) > 1:
                # Reconstruct the order using NEXT_SIBLING edges
                ordered_children = []
                current = child_list[0]
                visited = set()

                while current is not None and current not in visited:
                    visited.add(current)
                    ordered_children.append(current)
                    # Find next sibling
                    next_sibs = next_siblings.get(current, [])
                    current = next_sibs[0] if next_sibs else None

                # Add any remaining children that weren't in the sibling chain
                for child in child_list:
                    if child not in visited:
                        ordered_children.append(child)

                children[parent_idx] = ordered_children

        return children

    @staticmethod
    def ast_to_program(graph: Dict[str, Any]) -> str:
        """Convert AST graph back to Python code string.

        This is a simplified reconstruction that handles the core language constructs
        supported by our AST representation.
        """
        nodes = graph["nodes"]
        edges = graph["edges"]

        # Build children map for efficient traversal
        children = ASTSimplifier._build_children_lists(len(nodes), edges)

        # Start from the module node (index 0)
        module_idx = 0

        def indent_block(text: str, levels: int = 1) -> str:
            pad = "    " * levels
            lines = text.splitlines()
            if not lines:
                return pad
            return "\n".join(pad + line if line else pad for line in lines)

        def reconstruct_node(idx: int) -> str:
            """Recursively reconstruct a node and its children."""
            if idx >= len(nodes):
                raise ValueError(f"Invalid node index at node #{idx}")

            node = nodes[idx]
            node_type = node["type"]
            if isinstance(node_type, ASTNodeType):
                node_type = node_type.value

            # Get children in order
            node_children = children.get(idx, [])

            def _is_invalid(code_str: str) -> bool:
                if code_str is None:
                    return True
                stripped = code_str.strip()
                if stripped == "":
                    return True
                # Treat our placeholders as invalid expressions in code positions
                return stripped.startswith("#") or "<invalid" in stripped

            def _raise_invalid(reason: str, node_idx: int, node_obj: Dict[str, Any]) -> None:
                ntype = node_obj.get("type")
                # Normalize enum types for display
                if isinstance(ntype, ASTNodeType):
                    ntype = ntype.value
                child_count = len(children.get(node_idx, []))
                raise ValueError(f"{reason} at node #{node_idx} (type={ntype}, children={child_count})")

            if node_type == "module":
                # Module contains function definitions
                if node_children:
                    return "\n".join(reconstruct_node(child) for child in node_children)
                return ""

            elif node_type == "function_def":
                name = node.get("name", "program")
                params = node.get("params", ["n"])
                param_str = ", ".join(params)

                # Get function body statements
                body_stmts: List[str] = [reconstruct_node(child) for child in node_children]

                body_text = "\n".join(body_stmts)
                body = indent_block(body_text, levels=1)
                if body:
                    body = "\n" + body + "\n"

                return f"def {name}({param_str}):{body}"

            elif node_type == "return":
                if node_children:
                    value = reconstruct_node(node_children[0])
                    return f"return {value}"
                return "return"

            elif node_type == "assignment":
                if len(node_children) >= 2:
                    target = reconstruct_node(node_children[0])
                    value = reconstruct_node(node_children[1])
                    # If either side is invalid/comment placeholder, mark as invalid assignment
                    if (not target) or target.lstrip().startswith("#") or (not value) or value.lstrip().startswith("#"):
                        _raise_invalid("Invalid assignment", idx, node)
                    return f"{target} = {value}"
                _raise_invalid("Invalid assignment (arity)", idx, node)
                return ""

            elif node_type == "augmented_assignment":
                op = node.get("op", "+=")
                if len(node_children) >= 2:
                    target = reconstruct_node(node_children[0])
                    value = reconstruct_node(node_children[1])
                    return f"{target} {op} {value}"
                return f"# <invalid_augmented_assignment_{op}>"

            elif node_type == "if":
                if len(node_children) >= 2:
                    condition = reconstruct_node(node_children[0])
                    if _is_invalid(condition):
                        _raise_invalid("Invalid if condition", idx, node)
                    body_code = reconstruct_node(node_children[1])
                    # Ensure non-empty/non-comment body
                    if _is_invalid(body_code) or all((ln.strip() == "" or ln.lstrip().startswith('#')) for ln in body_code.splitlines()):
                        _raise_invalid("Invalid if body", idx, node)
                    body = indent_block(body_code, levels=1)

                    # Check if there's an else clause
                    else_body = ""
                    if len(node_children) >= 3:
                        else_code = reconstruct_node(node_children[2])
                        if _is_invalid(else_code) or all((ln.strip() == "" or ln.lstrip().startswith('#')) for ln in else_code.splitlines()):
                            _raise_invalid("Invalid else body", idx, node)
                        else_body = "\nelse:\n" + indent_block(else_code, levels=1)

                    return f"if {condition}:\n{body}{else_body}"
                return "# <invalid_if>"

            elif node_type == "for":
                if len(node_children) >= 3:
                    target = reconstruct_node(node_children[0])
                    iterator = reconstruct_node(node_children[1])
                    if _is_invalid(target):
                        _raise_invalid("Invalid for target", idx, node)
                    if _is_invalid(iterator):
                        _raise_invalid("Invalid for iterator", idx, node)
                    # Concatenate all body statements
                    body_parts = [reconstruct_node(child) for child in node_children[2:]]
                    body_code = "\n".join(body_parts)
                    if _is_invalid(body_code) or all((ln.strip() == "" or ln.lstrip().startswith('#')) for ln in body_code.splitlines()):
                        _raise_invalid("Invalid for body", idx, node)
                    body = indent_block(body_code, levels=1)
                    return f"for {target} in {iterator}:\n{body}"
                elif len(node_children) >= 2:
                    iterator = reconstruct_node(node_children[0])
                    if _is_invalid(iterator):
                        _raise_invalid("Invalid for iterator", idx, node)
                    body_code = reconstruct_node(node_children[1])
                    if _is_invalid(body_code) or all((ln.strip() == "" or ln.lstrip().startswith('#')) for ln in body_code.splitlines()):
                        _raise_invalid("Invalid for body", idx, node)
                    body = indent_block(body_code, levels=1)
                    return f"for i in {iterator}:\n{body}"
                _raise_invalid("Invalid for (arity)", idx, node)
                return ""

            elif node_type == "while":
                if len(node_children) >= 2:
                    condition = reconstruct_node(node_children[0])
                    if _is_invalid(condition):
                        _raise_invalid("Invalid while condition", idx, node)
                    # Concatenate all body statements
                    body_parts = [reconstruct_node(child) for child in node_children[1:]]
                    body_code = "\n".join(body_parts)
                    if _is_invalid(body_code) or all((ln.strip() == "" or ln.lstrip().startswith('#')) for ln in body_code.splitlines()):
                        _raise_invalid("Invalid while body", idx, node)
                    body = indent_block(body_code, levels=1)
                    return f"while {condition}:\n{body}"
                _raise_invalid("Invalid while (arity)", idx, node)
                return ""

            elif node_type == "variable":
                # Use the actual variable name if available, otherwise fall back to var_id
                var_name = node.get("name")
                if var_name:
                    return var_name
                else:
                    var_id = node.get("var_id", 0)
                    return f"var_{var_id}"

            elif node_type == "constant":
                value = node.get("value")
                dtype = node.get("dtype", "int")

                if dtype == "str":
                    return f'"{value}"'
                elif dtype == "bool":
                    return str(value)
                else:  # int or default
                    return str(value)

            elif node_type == "binary_operation":
                op = node.get("op", "+")
                if len(node_children) >= 2:
                    left = reconstruct_node(node_children[0])
                    right = reconstruct_node(node_children[1])
                    return f"({left} {op} {right})"
                # Fallback placeholder for missing operand to keep code parseable
                left = reconstruct_node(node_children[0]) if node_children else "0"
                return f"({left} {op} 0)"

            elif node_type == "unary_operation":
                op = node.get("op", "-")
                if node_children:
                    operand = reconstruct_node(node_children[0])
                    if op == "not":
                        return f"not {operand}"
                    return f"{op}{operand}"
                return f"# <invalid_unary_op_{op}>"

            elif node_type == "comparison":
                op = node.get("op", "==")
                if len(node_children) >= 2:
                    left = reconstruct_node(node_children[0])
                    right = reconstruct_node(node_children[1])
                    return f"({left} {op} {right})"
                return f"# <invalid_comparison_{op}>"

            elif node_type == "boolean_operation":
                op = node.get("op", "and")
                if len(node_children) >= 2:
                    left = reconstruct_node(node_children[0])
                    right = reconstruct_node(node_children[1])
                    return f"({left} {op} {right})"
                return f"# <invalid_boolean_op_{op}>"

            elif node_type == "function_call":
                func_name = node.get("function", "unknown")
                # Human-friendly rewrite: getitem(a,b) -> a[b]
                if func_name == "getitem" and len(node_children) >= 2:
                    value = reconstruct_node(node_children[0])
                    index = reconstruct_node(node_children[1])
                    return f"{value}[{index}]"

                # Method-call pretty print for common patterns where we encoded
                # an attribute call as FUNCTION_CALL with the object as arg[0]
                method_like = {"append", "upper", "lower", "add", "extend", "insert", "pop", "clear"}
                if func_name in method_like and len(node_children) >= 1:
                    obj = reconstruct_node(node_children[0])
                    call_args = [reconstruct_node(child) for child in node_children[1:]]
                    args_str = ", ".join(call_args)
                    return f"{obj}.{func_name}({args_str})"

                # Recognize sorted(x)[::-1] pattern we normalized from reverse=True
                if func_name == "sorted" and len(node_children) == 1:
                    # If the single argument is a subscript slice with step -1, print as sorted(arg, reverse=True)
                    arg_code = reconstruct_node(node_children[0])
                    if arg_code.endswith("[::-1]"):
                        base = arg_code[:-6]
                        return f"sorted({base}, reverse=True)"

                if node_children:
                    args = [reconstruct_node(child) for child in node_children]
                    args_str = ", ".join(args)
                    return f"{func_name}({args_str})"
                return f"{func_name}()"

            elif node_type == "list":
                if node_children:
                    elements = [reconstruct_node(child) for child in node_children]
                    return f"[{', '.join(elements)}]"
                return "[]"

            elif node_type == "attribute":
                attr_name = node.get("attr", "unknown")
                if node_children:
                    obj = reconstruct_node(node_children[0])
                    return f"{obj}.{attr_name}"
                return f"# <invalid_attribute_{attr_name}>"

            elif node_type == "subscript":
                if len(node_children) >= 2:
                    value = reconstruct_node(node_children[0])
                    slice_expr = reconstruct_node(node_children[1])
                    if _is_invalid(value) or _is_invalid(slice_expr):
                        _raise_invalid("Invalid subscript", idx, node)
                    return f"{value}[{slice_expr}]"
                _raise_invalid("Invalid subscript (arity)", idx, node)
                return ""

            elif node_type == "expression":
                if node_children:
                    return reconstruct_node(node_children[0])
                _raise_invalid("Empty expression", idx, node)
                return ""

            else:
                return f"# <unknown_node_type_{node_type}>"

        # Start reconstruction from the module
        result = reconstruct_node(module_idx)

        # Clean up the result
        if result.startswith("#"):
            raise ValueError("Failed reconstruction: top-level placeholder")

        return result

