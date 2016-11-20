#!/bin/bash
#this script trains and runs the networks using the 'best' methods tried so far

set -e

##remove any cached training/validation/testing data
#rm -f data_*.npz

#train the coarse network
echo "Training coarse net"
./main.sh train coarse -s 3000 -n

#train the detailed network
echo "Training detailed net"
sed -i 's/\(POS_PROBS =\) \[[0-9., ]\+\]/\1 [0.3, 0.3, 0.4]/' pkg/network_input.py
./main.sh train detailed -s 2000 -n
sed -i 's/\(POS_PROBS =\) \[[0-9., ]\+\]/\1 [0.03, 0.03, 0.94]/' pkg/network_input.py
./main.sh train detailed -s 3000

#run the coarse network on the images
echo "Running coarse net"
python3 use_network.py run images data_filter.txt -c -o coarse_output

#run the detailed network on the images
echo "Running detailed net"
python3 use_network.py run images data_filter.txt -o detailed_output

