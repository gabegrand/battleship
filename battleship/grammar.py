import random
import sys

from eig.battleship import Parser
from nltk import CFG
from nltk.grammar import Nonterminal
from nltk.parse.generate import generate
from tqdm import tqdm


class BattleshipGrammar:
    def __init__(self, include_lambdas: bool = False):
        grammar_str = GRAMMAR_BASE
        if include_lambdas:
            grammar_str += GRAMMAR_LAMBDA
        self.grammar = CFG.fromstring(grammar_str)

        # Difference between the program AST depth and the grammar depth
        # The grammar always follows A -> B -> <program> so it has +2 depth relative to the AST
        self.GRAMMAR_DEPTH_OFFSET = 2

    def generate(
        self,
        n: int = None,
        depth: int = None,
        start: Nonterminal = None,
        enforce_type: bool = False,
    ):
        """Enumerates all possible programs in the grammar."""
        valid, invalid = [], []

        for tokens in tqdm(
            generate(self.grammar, n=n, depth=depth, start=start), total=n
        ):
            program = " ".join(tokens)
            try:
                Parser.parse(program, enforce_type=enforce_type)
                valid.append(program)
            except:
                invalid.append(program)

        return valid, invalid

    def sample(
        self, min_depth: int = 1, max_depth: int = 16, allow_single_token: bool = False
    ):
        """Returns a random sample from the grammar using uniform probabilities over the rules.

        Return: (program, depth)

        NOTE: min_depth and max_depth are specified in terms of the program AST depth, not the grammar depth.
        Similarly, the return depth is the program AST depth, not the grammar depth.

        """

        def _sample(grammar, fragments, depth):
            if depth <= 0:
                raise RecursionError(f"Maximum recursion depth exceeded.")
            for frag in fragments:
                if isinstance(frag, str):
                    yield frag
                else:
                    productions = grammar.productions(lhs=frag)
                    if not productions:
                        raise ValueError(f"Nonterminal {frag} not found in grammar.")
                    production = random.choice(productions)
                    for sym in _sample(grammar, production.rhs(), depth - 1):
                        yield sym

        while True:
            try:
                program = " ".join(
                    _sample(
                        grammar=self.grammar,
                        fragments=[self.grammar.start()],
                        depth=max_depth + self.GRAMMAR_DEPTH_OFFSET,
                    )
                )
            except RecursionError as error:
                return None

            generated_depth = Parser.depth(program)
            if generated_depth >= min_depth:
                # Single-token programs have no parentheses; e.g., "TRUE", "B4", etc.
                if "(" in program or allow_single_token:
                    return (program, generated_depth)


# Grammar for Battleship DSL
# Adapted from https://arxiv.org/abs/1711.06351
# See Table SI-1 and SI-2 in the supplementary material
GRAMMAR_BASE = """
# Answer types
A -> B
A -> N
A -> C
A -> O
A -> L

# Yes/no
B -> 'TRUE'
B -> 'FALSE'
B -> '(' 'not' B ')'
B -> '(' 'and' B B ')'
B -> '(' 'or' B B ')'
B -> '(' '==' B B ')'
B -> '(' '==' N N ')'
B -> '(' '==' O O ')'
B -> '(' '==' C C ')'
# B -> '(' '==' setN ')' # not implemented in parser
B -> '(' '>' N N ')'
B -> '(' '<' N N ')'
B -> '(' 'touch' S S ')'
B -> '(' 'isSubset' setL setL ')'

# Numbers
N -> '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
N -> '(' '+' N N ')'
N -> '(' '+' B B ')'
N -> '(' '-' N N ')'
N -> '(' 'size' S ')'
N -> '(' 'rowL' L ')'
N -> '(' 'colL' L ')'
N -> '(' 'setSize' setL ')'

# Colors
C -> S | 'Water'
S -> 'Blue' | 'Red' | 'Purple'
C -> '(' 'color' L ')'

# Orientation
O -> 'H' | 'V'
O -> '(' 'orient' S ')'

# Locations
# NOTE: The parser requires enumerating each possible location
L -> '1A' | '1B' | '1C' | '1D' | '1E' | '1F'
L -> '2A' | '2B' | '2C' | '2D' | '2E' | '2F'
L -> '3A' | '3B' | '3C' | '3D' | '3E' | '3F'
L -> '4A' | '4B' | '4C' | '4D' | '4E' | '4F'
L -> '5A' | '5B' | '5C' | '5D' | '5E' | '5F'
L -> '6A' | '6B' | '6C' | '6D' | '6E' | '6F'

L -> '(' 'topleft' setL ')'
L -> '(' 'bottomright' setL ')'

# Sets
setS -> '(' 'set' 'AllColors' ')'
setL -> '(' 'set' 'AllTiles' ')'
setL -> '(' 'coloredTiles' C ')'
setL -> '(' 'setDifference' setL setL ')'
setL -> '(' 'union' setL setL ')'
setL -> '(' 'intersection' setL setL ')'
setL -> '(' 'unique' setL ')'
"""

# Grammar for lambda expressions
GRAMMAR_LAMBDA = """

# Sets
B -> '(' 'any' setB ')'
B -> '(' 'all' setB ')'
N -> '(' '++' setB ')'
N -> '(' '++' setN ')'

# Mapping
setB -> '(' 'map' fyB setL ')'
setB -> '(' 'map' fxB setS ')'
setN -> '(' 'map' fxN setS ')'
setL -> '(' 'map' fxL setS ')'

# Lambda expressions
fyB -> '(' 'lambda' 'y0' B ')'
fxB -> '(' 'lambda' 'x0' B ')'
fxN -> '(' 'lambda' 'x0' N ')'
fxL -> '(' 'lambda' 'x0' L ')'

# Lambda variables
# NOTE: These will create invalid programs when used outside of a lambda expression
S -> 'x0'
L -> 'y0'
"""
