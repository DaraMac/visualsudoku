import csv
import timeit
import brute_solver
import solver_utils as util
# import constraint_solver
# import SA

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
