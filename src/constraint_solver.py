from itertools import chain

import constraint as con
from solver_utils import *


def convert_dict(d):
    """Takes a single solution dictionary and puts it into a 2D list format."""
    d_list = [d[i] for i in range(81)]
    d_list = [d_list[i*9:i*9+9] for i in range(9)]

    return d_list


def solve(grid):
    """Takes the grid in 2D list format and returns it in solved form.

    Doesn't currently check if multiple solutions exist but could be easily modified to do so."""

    listed_grid = list(chain(*grid))

    problem = con.Problem()
    #problem.addVariables(range(81), range(1, 10))
    for i in range(81):
        if listed_grid[i] == 0:
            problem.addVariable(i, range(1,10))
        else:
            problem.addVariable(i, [listed_grid[i]])


    for i in range(9):
        problem.addConstraint(con.AllDifferentConstraint(), range(i*9, 9 + i*9))
        problem.addConstraint(con.AllDifferentConstraint(), range(i, 9*9+i, 9))


    problem.addConstraint(con.AllDifferentConstraint(), [0,  1,  2,  9,  10, 11, 18, 19, 20])
    problem.addConstraint(con.AllDifferentConstraint(), [3,  4,  5,  12, 13, 14, 21, 22, 23])
    problem.addConstraint(con.AllDifferentConstraint(), [6,  7,  8,  15, 16, 17, 24, 25, 26])

    problem.addConstraint(con.AllDifferentConstraint(), [27, 28, 29, 36, 37, 38, 45, 46, 47])
    problem.addConstraint(con.AllDifferentConstraint(), [30, 31, 32, 39, 40, 41, 48, 49, 50])
    problem.addConstraint(con.AllDifferentConstraint(), [33, 34, 35, 42, 43, 44, 51, 52, 53])

    problem.addConstraint(con.AllDifferentConstraint(), [54, 55, 56, 63, 64, 65, 72, 73, 74])
    problem.addConstraint(con.AllDifferentConstraint(), [57, 58, 59, 66, 67, 68, 75, 76, 77])
    problem.addConstraint(con.AllDifferentConstraint(), [60, 61, 62, 69, 70, 71, 78, 79, 80])


    #for i in range(81):
    #    if listed_grid[i] != 0:
    #        def fix(n, grid_val=listed_grid[i]):
    #            if n == grid_val:
    #                return True
    #        problem.addConstraint(fix, [i])
    #        # problem.addConstraint(lambda k: k == listed_grid[i], [i])


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
# 27 28 29  30 31 32  33 34 35
# 36 37 38  39 40 41  42 43 44
# 45 46 47  48 49 50  51 52 53
# 
# 54 55 56  57 58 59  60 61 62
# 63 64 65  66 67 68  69 70 71
# 72 73 74  75 76 77  78 79 80

    s = problem.getSolution()

    return convert_dict(s)
