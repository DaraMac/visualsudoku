def read_grid(filename: str):
    """Reads in a single sudoku puzzle in txt format and returns it as a 2D list.

    Expects spaces between numbers in a row, no spaces between rows, and 0s represent unfilled values.
    Doesn't make any assumptions or checks on dimensions."""
    grid = []
    with open(filename) as f:
        for line in f:
            grid.append(list(map(int, line.split())))
    return grid


def print_grid(grid):
    """Takes a grid in 2D array format and prints it nicely for comparison purposes."""
    print('\n'.join(map(lambda r: ' '.join(map(str, r)), grid)))
