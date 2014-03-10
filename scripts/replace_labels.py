#!/usr/bin/env python

import os

PREFIX="../new-data"


def K(x):
    lookup = { "LOW" : "-1", "HIGH" : "1" }
    return lookup[x]

for filename in os.listdir(PREFIX):
    if filename.endswith("labels.txt"):
        labels = file(PREFIX + "/" + filename).read().strip().split("\n")

        labels = map(K, labels)

        newfile = file(PREFIX + "/" + filename, "w")
        newfile.write("\n".join(labels))

        #print "\n".join(labels)
        #print labels
    
