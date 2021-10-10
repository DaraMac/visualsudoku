from constraint import *

problem = Problem()

problem.addVariables(range(81), range(1, 10))

for i in range(9):
    problem.addConstraint(AllDifferentConstraint(), range(i*9, 9 + i*9))
    problem.addConstraint(AllDifferentConstraint(), range(i, 9*8+i, 9))

solution = problem.getSolution() # only 1 sol

def convert_dict(d):
    """Takes a single solution dictionary and puts it into a 2D list format."""
    d_list = list(d.values())
    d_list = [d_list[i:i+9] for i in range(9)]

    return d_list
