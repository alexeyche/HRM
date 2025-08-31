## Current state of the codebase

We are mostly done with our grammar and autoencoder implementation.
Core files:
- `dataset/programs.py` represent current dataset and program space we are interesting in
- `dataset/grammar.py` and `tests/test_grammar.py` cover a lot of important functionality
  - main nuances
- `models/ast_autoencder.py` and more specifically `models/generation_head.py` doing most of heavily lifting learning the context free grammar structure and learning to generate programs

## Main problem

Code generation with our custom `models/generation_head.py` works but it's pretty slow.
It uses Earley parser that works only per one sample, i.e we cannot scale it up to batch learning.

## Suggestion

Let's introduce new head with same API as old generation head in `models/struct_generation_head.py`

Main features:
- Instead of custom implementation it uses [Torch Struct library](https://github.com/harvardnlp/pytorch-struct) (already installed)
- It batch by default
- In the same way as old generation head it represents rule based structured generation
- It benefits and built upon the NLTK grammar in `dataset/grammar.py`
- In the same way as old generation head it can learn
  - production rules
  - literals
  - identifiers
- Let's have old generation head as good reference yet let's not be bounded by the code we used to have, let's benefit completely from Torch Struct library
- Let's have same comprehensive test coverage as we have
  - for `parse_tokens_to_productions` in `tests/test_grammar.py`
  - generation head in `tests/test_generation_head.py`
  - generation loss in `tests/test_generation_loss.py`
- Implement flag in `train_ast_autoencoder.py` to switch between two heads
- Feel free to mirror our old implementation just prefix it with `struct_`, e.g. `test_generation_head.py` -> `test_struct_generation_head.py`