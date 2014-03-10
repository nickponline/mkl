#!/bin/bash

#for K in go_bp go_cc goccgomfgobp goipdaiaoa go_mf ia ip oa 
for K in go_mf ia ip oa 
do
	python irisvmms-500.py 500-sets/n1_${K}.feats 500-sets/n1_${K}.labs 500-sets/n2_${K}.feats 500-sets/n2_${K}.labs 500-sets/p_${K}.feats 500-sets/p_${K}.labs 
done
