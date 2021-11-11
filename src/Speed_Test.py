import timeit
import csv
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np

import brute_solver
import solver_utils as util
import constraint_solver
import SA

def convert_sudoku(sudoku):
    """Takes a sudoku as a single 81 digit string with 0's for blanks and returns 2D grid."""
    sudoku_list = [list(map(int, sudoku[i*9:i*9+9])) for i in range(9)]
    return sudoku_list

with open("small_test_set.csv", newline='') as f:
    reader = csv.reader(f)
    tests = []
    for line in reader:
        tests.append(convert_sudoku(line[0]))

# t = timeit.Timer("brute_solver.solve(tests[0])", globals=globals())

times = {"brute":[], "constraint":[], "anneal":[]}
# choosing 100 as that is the length of the small_test_set file
for i in range(100):
    times["brute"].append(timeit.timeit(f"brute_solver.solve(tests[{i}])", globals=globals(), number=1))
    times["constraint"].append(timeit.timeit(f"constraint_solver.solve(tests[{i}])", globals=globals(), number=1))
    times["anneal"].append(timeit.timeit(f"SA.solve(p)", setup=f"p = np.array(tests[{i}])", globals=globals(), number=1))

print(f"Brute:\nMean {mean(times['brute'])}\nMax {max(times['brute'])}\nMin {min(times['brute'])}\n")
print(f"Constraint:\nMean {mean(times['constraint'])}\nMax {max(times['constraint'])}\nMin {min(times['constraint'])}\n")
print(f"Simulated Annealing:\nMean {mean(times['anneal'])}\nMax {max(times['anneal'])}\nMin {min(times['anneal'])}\n")
