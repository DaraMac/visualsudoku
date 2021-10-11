import constraint as con
from solver_utils import *


def convert_dict(d):
    """Takes a single solution dictionary and puts it into a 2D list format."""
    d_list = list(d.values())
    d_list = [d_list[i*9:i*9+9] for i in range(9)]

    return d_list


problem = con.Problem()

problem.addVariables(range(81), range(1, 10))

for i in range(9):
    problem.addConstraint(con.AllDifferentConstraint(), range(i*9, 9 + i*9))
    problem.addConstraint(con.AllDifferentConstraint(), range(i, 9*9+i, 9))

for x in range(9):
    for y in range(9):
        if grid[y][x] != 0:

# for i in range(3):
#     for j  in range(3):
#         problem.addConstraint(con.AllDifferentConstraint(), )
# 
# (3*i + [0, 1, 2]) + (9*(j+1) * [0, 1, 2])
# 
# 00 01 02  03 04 05  06 07 08
# 09 10 11  12 13 14  15 16 17
# 18 19 20  21 22 23  24 25 26
# 
# 27 28 29
# 36 37 38
# 45 46 47
# 
# 54 55 56
# 63 64 65
# 72 73 74

s = problem.getSolution()
# solution = convert_dict(problem.getSolution()) # only 1 sol
