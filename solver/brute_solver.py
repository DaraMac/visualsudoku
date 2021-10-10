# developed in Python 3.9
# define sudoku as a 9x9 2D list with 0s for unfilled in values, where each sublist represents a row, starting indexing from top left corner

from solver_utils import * 

def solve(grid):
    """Takes a valid sudoku grid and returns it solved."""
    original_layout = [[True if grid[y][x] != 0 else False for x in range(9)] for y in range(9)]
    
    #with open("logs.txt", 'w') as f:
    count = 0
    i = 0
    while i < 81:
        count += 1

        y = i // 9
        x = i % 9
        
        if original_layout[y][x]:
            i += 1
            continue

        while grid[y][x] < 9:
            grid[y][x] += 1
            if check_location(grid, x, y):
                i += 1
                break
        else:
            grid[y][x] = 0
            i -= 1
            while original_layout[i // 9][i % 9]:
                i -= 1

    return grid
