import constraint as con
from solver_utils import *


def convert_dict(d):
    """Takes a single solution dictionary and puts it into a 2D list format."""
    d_list = list(d.values())
    d_list = [d_list[i:i+9] for i in range(9)]

    return d_list


problem = con.Problem()

problem.addVariables(range(81), range(1, 10))

for i in range(9):
    problem.addConstraint(con.AllDifferentConstraint(), range(i*9, 9 + i*9))
    problem.addConstraint(con.AllDifferentConstraint(), range(i, 9*9+i, 9))

s = problem.getSolution()
# solution = convert_dict(problem.getSolution()) # only 1 sol
