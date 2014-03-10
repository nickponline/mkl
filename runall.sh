#!/bin/bash

for J in 2 3
do
for K in go_bp go_cc goip go_mf ip go_bp 
do
echo $J $K
python irisvmms-500.py new-data/gisIn33_dmel08HcLcT${J}-500_${K}_protId_SVM.txt new-data/gisIn33_dmel08HcLcT${J}-500_${K}_protId_SVM_labels.txt
done
done

