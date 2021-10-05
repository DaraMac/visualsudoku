# developed in Python 3.9
# define sudoku as a 9x9 2D list with 0s for unfilled in values, where each sublist represents a row, starting indexing from top left corner

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
    checked = [False]*9
    for n in grid[y]:
        if n == 0:
            continue
        if not checked[n-1]:
            checked[n-1] = True
        else:
            return False
    return True # only hit if it passes through the whole row with no duplicates


def check_col(grid, x) -> bool:
    """Given the current grid and a column, returns True if that column is valid."""
    checked = [False]*9
    for i in range(9):
        if grid[i][x] == 0:
            continue
        if not checked[grid[i][x]-1]:
            checked[grid[i][x]-1] = True
        else:
            return False
    return True


def check_box(grid, x, y) -> bool:
    """Determines which of the 9 boxes a location lies in and checks if that box is valid."""
    if x < 3:
        x_box = 0
    elif x < 6:
        x_box = 1
    else:
        x_box = 2

    if y < 3:
        y_box = 0
    elif y < 6:
        y_box = 1
    else:
        y_box = 2

    box = [grid[3*y_box + j][3*x_box + i] for i in range(3) for j in range(3)]
    checked = [False]*9
    for n in box:
        if n == 0:
            continue
        if not checked[n-1]:
            checked[n-1] = True
        else:
            return False
    return True


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
        
#        if count > 9950:
#            f.write("i=" + str(i) + " x=" + str(x) + " y=" + str(y) + " val=" + str(grid[y][x]) + " Original=" + str(original_layout[y][x]) +"\n")
#            for row in grid:
#                f.write(str(row) + "\n")
#            f.write("\n")
#
#        if count > 10000:
#            break
#

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

s1 = read_grid("s1.txt")
s2 = read_grid("s2.txt")
s3 = read_grid("s3.txt")
s4 = read_grid("s4.txt")
very_easy = read_grid("very_easy.txt")
hard = read_grid("hard.txt")
