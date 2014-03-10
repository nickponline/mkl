from __future__ import division
import random
iterations = 10
segmentSize = 1 / iterations
variableMin = 0
variableMax = 1024
for i in range(iterations):
    segmentMin = i * segmentSize
    #segmentMax = (i+1) * segmentSize
    point = segmentMin + (random.random() * segmentSize)
    pointValue = (point * (variableMax - variableMin)) + variableMin
    print point, pointValue