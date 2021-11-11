import random
import numpy as np
import math 
from random import choice
import statistics 

startingSudoku = """
                    936000200
                    000093740
                    040821009
                    472000006
                    000759000
                    100000378
                    500416020
                    021370000
                    004000157
                """

sudoku = np.array([[int(i) for i in line] for line in startingSudoku.split()])

def fix_sudoku_values(fixed_sudoku):
    for i in range (0,9):
        for j in range (0,9):
            if fixed_sudoku[i,j] != 0:
                fixed_sudoku[i,j] = 1
    
    return(fixed_sudoku)

# Cost Function    
def number_0f_errors(sudoku):
    errors = 0 
    for i in range (0,9):
        errors += sum_error_row_column(i ,i ,sudoku)
    return(errors)

def sum_error_row_column(row, column, sudoku):
    errors = (9 - len(np.unique(sudoku[:,column]))) + (9 - len(np.unique(sudoku[row,:])))
    return(errors)


def create_list_3x3_blocks ():
    blocks = []
    for r in range (0,9):
        tmp = []
        block1 = [i + 3*((r)%3) for i in range(0,3)]
        block2 = [i + 3*math.trunc((r)/3) for i in range(0,3)]
        for x in block1:
            for y in block2:
                tmp.append([x,y])
        blocks.append(tmp)
    return(blocks)

def randomly_fill_3x3_blocks(sudoku, blocks):
    for block in blocks:
        for box in block:
            if sudoku[box[0],box[1]] == 0:
                current_block = sudoku[block[0][0]:(block[-1][0]+1),block[0][1]:(block[-1][1]+1)]
                sudoku[box[0],box[1]] = choice([i for i in range(1,10) if i not in current_block])
    return sudoku

def sum_of_one_block (sudoku, block):
    finalSum = 0
    for box in block:
        finalSum += sudoku[box[0], box[1]]
    return(finalSum)

def two_random_cells_within_block(fixed_sudoku, block):
    while (1):
        first_cell = random.choice(block)
        second_cell = choice([cell for cell in block if cell is not first_cell ])

        if fixed_sudoku[first_cell[0], first_cell[1]] != 1 and fixed_sudoku[second_cell[0], second_cell[1]] != 1:
            return([first_cell, second_cell])

def switch_cells(sudoku, cells_to_switch):
    proposed_sudoku = np.copy(sudoku)
    place_holder = proposed_sudoku[cells_to_switch[0][0], cells_to_switch[0][1]]
    proposed_sudoku[cells_to_switch[0][0], cells_to_switch[0][1]] = proposed_sudoku[cells_to_switch[1][0], cells_to_switch[1][1]]
    proposed_sudoku[cells_to_switch[1][0], cells_to_switch[1][1]] = place_holder
    return (proposed_sudoku)

def proposed_state (sudoku, fixed_sudoku, blocks):
    random_block = random.choice(blocks)

    if sum_of_one_block(fixed_sudoku, random_block) > 6:  
        return(sudoku, 1, 1)
    cells_to_switch = two_random_cells_within_block(fixed_sudoku, random_block)
    proposed_sudoku = switch_cells(sudoku,  cells_to_switch)
    return([proposed_sudoku, cells_to_switch])

def choose_new_state (current_sudoku, fixed_sudoku, block, sigma):
    proposal = proposed_state(current_sudoku, fixed_sudoku, block)
    new_sudoku = proposal[0]
    cells_to_check = proposal[1]
    current_cost = sum_error_row_column(cells_to_check[0][0], cells_to_check[0][1], current_sudoku) + sum_error_row_column(cells_to_check[1][0], cells_to_check[1][1], current_sudoku)
    new_cost = sum_error_row_column(cells_to_check[0][0], cells_to_check[0][1], new_sudoku) + sum_error_row_column(cells_to_check[1][0], cells_to_check[1][1], new_sudoku)
    # current_cost = number_0f_errors(current_sudoku)
    # new_cost = number_0f_errors(new_sudoku)
    cost_difference = new_cost - current_cost
    rho = math.exp(-cost_difference/sigma)
    if(np.random.uniform(1,0,1) < rho):
        return([new_sudoku, cost_difference])
    return([current_sudoku, 0])


def choose_number_of_itterations(fixed_sudoku):
    itterations = 0
    for i in range (0,9):
        for j in range (0,9):
            if fixed_sudoku[i,j] != 0:
                itterations += 1
    return itterations

def compute_initial_sigma (sudoku, fixed_sudoku, blocks):
    list_of_differences = []
    tmp = sudoku
    for i in range(1,10):
        tmp = proposed_state(tmp, fixed_sudoku, blocks)[0]
        list_of_differences.append(number_0f_errors(tmp))
    return (statistics.pstdev(list_of_differences))


def solve(sudoku):
    # f = open("demofile2.txt", "a")
    solutionFound = 0
    while (solutionFound == 0):
        decreaseFactor = 0.98
        stuckCount = 0
        fixedSudoku = np.copy(sudoku)
        fix_sudoku_values(fixedSudoku)
        listOfBlocks = create_list_3x3_blocks()
        tmpSudoku = randomly_fill_3x3_blocks(sudoku, listOfBlocks)
        sigma = compute_initial_sigma(sudoku, fixedSudoku, listOfBlocks)
        score = number_0f_errors(tmpSudoku)
        itterations = choose_number_of_itterations(fixedSudoku)
        if score <= 0:
            solutionFound = 1

        while solutionFound == 0:
            previousScore = score
            for i in range (0, itterations):
                newState = choose_new_state(tmpSudoku, fixedSudoku, listOfBlocks, sigma)
                tmpSudoku = newState[0]
                scoreDiff = newState[1]
                score += scoreDiff
                print(score)
                # f.write(str(score) + '\n')
                if score <= 0:
                    solutionFound = 1
                    break

            sigma *= decreaseFactor
            if score <= 0:
                solutionFound = 1
                break
            if score >= previousScore:
                stuckCount += 1
            else:
                stuckCount = 0
            if (stuckCount > 80):
                sigma += 2
            if(number_0f_errors(tmpSudoku)==0):
                break
    # f.close()
    return(tmpSudoku)


# solution = solve_sudoku(sudoku)
# print(number_0f_errors(solution))
# print(solution)
