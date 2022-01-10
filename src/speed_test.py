import timeit
import csv
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import interactive # to make plots show while testing
import numpy as np
import pandas as pd
import seaborn as sns

import brute_solver
import solver_utils as util
import constraint_solver
import SA

def convert_sudoku(sudoku):
    """Takes a sudoku as a single 81 digit string with 0's for blanks and returns 2D grid."""
    sudoku_list = [list(map(int, sudoku[i*9:i*9+9])) for i in range(9)]
    return sudoku_list

# format of both test set csvs is: quizzes, solutions
with open("small_test_set.csv", newline='') as f:
# with open("test_set.csv", newline='') as f:
    reader = csv.reader(f)
    tests = []
    for line in reader:
        tests.append(convert_sudoku(line[0]))


times = {"brute":[], "constraint":[], "anneal":[]}
for i in range(100): # the length of small_test_set.csv
# for i in range(1000): # for the larger test_set.csv file
    times["brute"].append(timeit.timeit(f"brute_solver.solve(tests[{i}])", globals=globals(), number=1))
    times["constraint"].append(timeit.timeit(f"constraint_solver.solve(tests[{i}])", globals=globals(), number=1))
    times["anneal"].append(timeit.timeit(f"SA.solve(p)", setup=f"p = np.array(tests[{i}])", globals=globals(), number=1))

print(f"Brute:\nMean {mean(times['brute'])}\nMax {max(times['brute'])}\nMin {min(times['brute'])}\n")
print(f"Constraint:\nMean {mean(times['constraint'])}\nMax {max(times['constraint'])}\nMin {min(times['constraint'])}\n")
print(f"Simulated Annealing:\nMean {mean(times['anneal'])}\nMax {max(times['anneal'])}\nMin {min(times['anneal'])}\n")

brute = times["brute"]
constraint = times["constraint"]
anneal = times["anneal"]
# genetic = times["genetic"]

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

# fig, ((ax11, ax12), (ax13, ax14)) = plt.subplots(2, 2)
# 
# fig.suptitle("Speed comparison of Sudoku solving algorithms")
# 
# ax11.hist(brute)
# ax11.set_title("Brute Force")
# 
# ax12.hist(constraint)
# ax12.set_title("Constraint Solver")
# 
# ax13.hist(anneal)
# ax13.set_title("Simulated Annealing")
# 
# fig2, ax2 = plt.subplots()
# ax2.hist([brute, constraint, anneal],
#         bins=30,
#         label=["brute", "constraint", "simulated annealing"])
# ax2.set_xlabel("Time to solve (seconds)")
# ax2.set_ylabel("No. of Sudoku")
# ax2.legend()

# fig3, ax3 = plt.subplots()
# ax3.scatter(anneal, constraint)

sns.set_theme(style="ticks")

df = pd.DataFrame.from_dict(times)
p = sns.pairplot(df)
# p.fig.suptitle("Runtimes over 100 Sudoku") # y= some height>1

plt.show()
