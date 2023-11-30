from nltk import CFG

# Grammar for Battleship DSL
# Adapted from https://arxiv.org/abs/1711.06351
# See Table SI-1 and SI-2 in the supplementary material
BattleshipCFG = CFG.fromstring(
    """
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
B -> '(' 'any' setB ')'
B -> '(' 'all' setB ')'
B -> '(' '>' N N ')'
B -> '(' '<' N N ')'
B -> '(' 'touch' S S ')'
B -> '(' 'isSubset' setL setL ')'

# Numbers
N -> '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
N -> '(' '+' N N ')'
N -> '(' '+' B B ')'
N -> '(' '++' setN ')'
N -> '(' '++' setB ')'
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

# Sets
setS -> '(' 'set' 'AllColors' ')'
setL -> '(' 'set' 'AllTiles' ')'
setL -> '(' 'coloredTiles' C ')'
setL -> '(' 'setDifference' setL setL ')'
setL -> '(' 'union' setL setL ')'
setL -> '(' 'intersection' setL setL ')'
setL -> '(' 'unique' setL ')'

"""
)
