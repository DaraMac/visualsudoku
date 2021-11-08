import csv
from random import sample

nums = sample(range(1, 1000001), 1000)
nums.append(0)
num_set = set(nums)


with open("sudoku.csv", newline='') as f:
    with open("test_set.csv", 'w', newline='') as out:
        lines = csv.reader(f)
        out_writer = csv.writer(out)
        for i, s in enumerate(lines):
            if i in num_set:
                out_writer.writerow(s)
