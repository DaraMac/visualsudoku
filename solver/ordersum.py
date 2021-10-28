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
dict_input = [{1:1, 2:2, 3:3}, {0:0, 7:7, 8:8}, {3:3}, {2:2, 4:4}

# Make sure lists are sorted
for l in test_input:
	l.sort(reverse=True)

# Figure out biggest sum
current_sum = sum([l[0] for l in test_input])

# Make starting item
item = (-current_sum , test_input)

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

	for i in range(len(lol)):
		if len(lol[i]) > 1:
			newsum = current_sum - lol[i][0] + lol[i][1]
			newlist = lol.copy()
			newlist[i] = lol[i][1:]
			item = (-newsum, newlist)
			if not (str(item) in seen):
				heapq.heappush(heap, item)
				seen.add(str(item))
