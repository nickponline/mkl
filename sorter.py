import itertools

A = [(1,2), (2,1)]
A.sort(key=lambda k: k[1])
print A