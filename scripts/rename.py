#!/usr/bin/env python

import os

for filename in os.listdir("../500-sets"):
    newname = filename
    newname = newname.replace("dmel08-hc_gisIn_", "P_")
    newname = newname.replace("dmel08-lcT2-9593_gisIn_", "N1_")
    newname = newname.replace("dmel08-lcT3-9593_gisIn_", "N2_")
    newname = newname.replace("labels.txt", ".labs")
    newname = newname.replace(".txt", ".feats")
    newname = newname.replace("_protId_sampl_SVM", "")
    newname = newname.replace("_.", ".")
    newname = newname.lower()
    print newname
    os.rename("../500-sets/" + filename, "../500-sets/" + newname)
