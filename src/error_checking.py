import solver_utils as util 

# Input to this as a function should be a list of dicts
# where the keys in each dict are numbers 0-9 inclusive (0 meaning a blank cell)
# and the value for each key is the log of the probability the digit recogniser gives for that
# key being the digit in the cell
#
# The list will probably be length 81 corresponding directly to the whole Sudoku grid
# and each dict could have the full 10 entries unless we decide to cut off ones with too low
# or zero probability
#
# The configuration produced will then have to be checked to see if it's valid in its current state
# and then checked to see if it has a (unique?) solution
# This might happen here or in another function outside this

import heapq

test_input = [[1, 2, 3], [0, 7, 8], [3], [2, 4]]
# dict_input = [{1:1, 2:2, 3:3}, {0:0, 7:7, 8:8}, {3:3}, {2:2, 4:4}]

def get_errors(grid):
    """Returns list of tuples of indices (x, y) that are invalid in a grid."""
    errors = []
    for x in range(9):
        for y in range(9):
            n = grid[y][x]

            if n == 0:
                continue

            if grid[y].count(n) > 1:
                errors.append((x, y))
            elif [grid[j][x] for j in range(9)].count(n) > 1:
                errors.append((x, y))
            elif util.get_box(grid, x, y).count(n) > 1:
                errors.append((x, y))

    return errors


def remove_errors(grid, errors):
    """Takes a grid and a (non-empty) list of (x, y) co-ords of duplication errors in that grid and simply zeros them."""
    for x, y in errors:
        grid[y][x] = 0
    return grid


# TODO check this will work as all the logs will be negative values because probabilities are less than 1!
def enum_grids(probabilities):
    # Make sure lists are sorted
    for l in probabilities:
            l.sort(reverse=True)

    # Figure out biggest sum
    current_sum = sum([l[0] for l in probabilities])

    # Make starting item
    item = (-current_sum , probabilities)

    heap = []
    seen = set()
    heapq.heappush(heap, item)
    seen.add(str(item))

    while len(heap) > 0:
            item = heapq.heappop(heap)
            seen.discard(str(item))
            assert len(heap) == len(seen)
            current_sum = -item[0]
            lol = item[1]

            print("{} = sum({})".format(-item[0], [l[0] for l in lol]))
            yield [l[0] for l in lol] # new

            for i in range(len(lol)):
                    if len(lol[i]) > 1:
                            newsum = current_sum - lol[i][0] + lol[i][1]
                            newlist = lol.copy() # This doesn't work like you'd think
                            newlist[i] = lol[i][1:]
                            item = (-newsum, newlist)
                            if not (str(item) in seen):
                                    heapq.heappush(heap, item)
                                    seen.add(str(item))
                                    # print("heap =", heap)

# 7, 0.05751910805
# TODO check this will work as all the logs will be negative values because probabilities are less than 1!
# really it's useless unless we include spaces
def enum_errors(error_log_probs):
    """Like enum_grids but just for errors and in a slightly different format.

    error_log_probs is same format as for get_probable_grid."""

    # error_log_probs = [[(1, -0.69, 2, -.022, ..)], [..], ...]

    # Make sure lists are sorted
    for l in error_log_probs:
        l.sort(key=lambda t: t[1], reverse=True)

    # Figure out biggest sum
    current_sum = sum([l[0][1] for l in error_log_probs])

    # Make starting item
    item = (-current_sum , error_log_probs)

    heap = []
    seen = set()
    heapq.heappush(heap, item)
    seen.add(str(item))

    while len(heap) > 0:
            item = heapq.heappop(heap)
            seen.discard(str(item))
            assert len(heap) == len(seen)
            current_sum = -item[0]
            ls = item[1]

            # print("{} = sum({})".format(-item[0], [l[0] for l in ls]))
            yield [l[0] for l in ls]

            for i in range(len(ls)):
                    if len(ls[i]) > 1:
                            newsum = current_sum - ls[i][0][1] + ls[i][1][1]
                            newlist = ls.copy() # TODO this doesn't work like you'd think
                            newlist[i] = ls[i][1:]
                            item = (-newsum, newlist)
                            if not (str(item) in seen):
                                    heapq.heappush(heap, item)
                                    seen.add(str(item))
                                    # print("heap =", heap)


# TODO maybe dont use log probs?
# Make work with blanks!!!!
def get_probable_grid(grid, errors, probs):
    """Given the list of errors in a grid, returns the first valid grid that has the highest probability based on the digit recogniser probabilities.

    Where errors is the tuple of indices from get_errors, and probs is a list of lists of probabilities for the digits 1-9 according to the character recognition.
    The first element in the tuple will be the number and the second element the probability.
    The lists will be ordered so that index [i] in errors is the sudoku square with probabilities [i] in error_log_probs."""
    # error_log_probs = [[(1, -0.69, 2, -.022, ..)], [..], ...]

    # ndarry conversion to be consistent down the line
    # [1:] as we don't want the probability it is 0 as that's impossible
    probs = [list(enumerate(p.tolist()))[1:] for p in probs]
    eps = [probs[y*9 + x] for x, y in errors]

    gen = enum_errors(eps)
    # the most likely one will just be the configuration it's already in
    # so we skip that by calling next()
    gen.__next__()

    # config = gen.__next__()
    for ps in gen:
        for e, p in zip(errors, ps):
            x, y = e[0], e[1]
            grid[9*y + x] = p[0]

        # TODO add prob threshold cutoff point so it doesnt get ridiculous
        if util.check_valid(grid):
            yield grid
        # TODO maybe make a copy of the grid so as to not lose the original here?
