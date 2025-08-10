from dataset.grammar_actions import ParserState, Action, ActionKind

state = ParserState()
print('Initial state:', state.stack)
print('Initial valid actions:', state.get_valid_actions())

# Apply PROD_FUNCTION_DEF
new_roles = state.apply_action(Action(ActionKind.PROD_FUNCTION_DEF))
print('After PROD_FUNCTION_DEF:', state.stack, 'new_roles:', new_roles)
for role in new_roles:
    state.push(role)
print('After pushing STMT:', state.stack)

# Apply PROD_RETURN
new_roles = state.apply_action(Action(ActionKind.PROD_RETURN))
print('After PROD_RETURN:', state.stack, 'new_roles:', new_roles)
for role in new_roles:
    state.push(role)
print('After pushing EXPR:', state.stack)

# Apply PROD_VARIABLE
new_roles = state.apply_action(Action(ActionKind.PROD_VARIABLE))
print('After PROD_VARIABLE:', state.stack, 'new_roles:', new_roles)
print('Final stack:', state.stack)
