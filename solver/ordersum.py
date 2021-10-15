#!
import heapq

input = [ [1, 2, 3], [0, 7, 8], [3], [2, 4] ]

# Make sure lists are sorted
for l in input:
	l.sort(reverse=True)

# Figure out biggest sum
sum = sum([ l[0] for l in input])

# Make starting object
object = (-sum , input)

heap = [ ] 
seen = set()
heapq.heappush(heap, object)
seen.add(str(object))

while len(heap) > 0:
	object = heapq.heappop(heap)
	seen.discard(str(object))
	assert len(heap) == len(seen)
	sum = -object[0]
	lol = object[1]

	print("{} = sum({})".format(-object[0], [l[0] for l in lol]))

	for i in range(len(lol)):
		if len(lol[i]) > 1:
			newsum = sum - lol[i][0] + lol[i][1]
			newlist = lol.copy()
			newlist[i] = lol[i][1:]
			object = (-newsum, newlist)
			if not (str(object) in seen):
				heapq.heappush(heap, object)
				seen.add(str(object))

