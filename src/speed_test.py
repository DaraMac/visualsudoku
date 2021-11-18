import timeit
import csv
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import interactive # to make plots show while testing
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


# with open("test_set.csv", newline='') as f:
#     reader = csv.reader(f)
#     tests = []
#     for line in reader:
#         tests.append(convert_sudoku(line[0]))


times = {"brute":[], "constraint":[], "anneal":[]}
# choosing 100 as that is the length of the small_test_set file
for i in range(100):
# for i in range(1000): # for the larger test set file
    times["brute"].append(timeit.timeit(f"brute_solver.solve(tests[{i}])", globals=globals(), number=1))
    times["constraint"].append(timeit.timeit(f"constraint_solver.solve(tests[{i}])", globals=globals(), number=1))
    times["anneal"].append(timeit.timeit(f"SA.solve(p)", setup=f"p = np.array(tests[{i}])", globals=globals(), number=1))

print(f"Brute:\nMean {mean(times['brute'])}\nMax {max(times['brute'])}\nMin {min(times['brute'])}\n")
print(f"Constraint:\nMean {mean(times['constraint'])}\nMax {max(times['constraint'])}\nMin {min(times['constraint'])}\n")
print(f"Simulated Annealing:\nMean {mean(times['anneal'])}\nMax {max(times['anneal'])}\nMin {min(times['anneal'])}\n")

brute = times["brute"]
constraint = times["constraint"]
anneal = times["anneal"]


######################################################################
# Graphs

# interactive(True) # to make plots show while testing
# Note that even in the OO-style, we use `.pyplot.figure` to create the figure.
# fig, ax = plt.subplots()  # Create a figure and an axes.
# ax.plot(x, x, label='linear')  # Plot some data on the axes.
# ax.plot(x, x**2, label='quadratic')  # Plot more data on the axes...
# ax.plot(x, x**3, label='cubic')  # ... and some more.
# ax.set_xlabel('x label')  # Add an x-label to the axes.
# ax.set_ylabel('y label')  # Add a y-label to the axes.
# ax.set_title("Simple Plot")  # Add a title to the axes.
# ax.legend()  # Add a legend.

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

fig.suptitle("Speed comparison of Sudoku solving algorithms")

ax1.hist(brute)
ax1.set_title("Brute Force")

ax2.hist(constraint)
ax2.set_title("Constraint Solver")

ax3.hist(anneal)
ax3.set_title("Simulated Annealing")

fig2, ax = plt.subplots()
ax.hist([brute, constraint, anneal],
        bins=30,
        label=["brute", "constraint", "simulated annealing"])
ax.legend()

plt.show()
