# developed in Python 3.9.7
# define sudoku as a 9x9 2D list with 0s for unfilled in values

# nums = frozenset([1, 2, 3, 4, 5, 6, 7, 8, 9])

def check_location(grid, x, y) -> bool:
    """Given the grid in its current state and the 0-indexed co-ordinates of the number just added, returns True if it's allowed to be there.

    This function assumes the rest of the grid is valid and that each sublist represents a row."""
    if check_row(grid, y) and check_col(grid, x) and check_box(grid, x, y):
        return True
    else:
        return False

def check_row(grid, y) -> bool:
    """Given the current grid and a row, returns True if that row is valid."""
    checked = [False for _ in range(9)]
    for n in grid[y]:
        if n == 0:
            continue
        if not checked[n-1]:
            checked[n-1] = True
        else:
            return False
    return True # only hit if it passes through the whole row with no duplicates
