#!/bin/bash
#this script trains and runs the networks using the 'best' methods tried so far

set -e

#remove any cached training/validation/testing data
rm -f data_*.npz

#train the coarse network
echo "Training coarse net"
./main.sh train coarse -s 3000 -n

#train the detailed network
echo "Training detailed net"
sed -i 's/\(if not containsBuoy and random.random() >\) [0-9.]\+/\1 0.1/' pkg/network_input.py
./main.sh train detailed -s 3000 -n
sed -i 's/\(if not containsBuoy and random.random() >\) [0-9.]\+/\1 0.3/' pkg/network_input.py
rm -f data_detailed_*.npz
./main.sh train detailed -s 3000
sed -i 's/if True \(and containsBuoy: #add rotated images\)/if False \1/' pkg/network_input.py
rm -f data_detailed_*.npz
./main.sh train detailed -s 3000
sed -i 's/if False \(and containsBuoy: #add rotated images\)/if True \1/' pkg/network_input.py

#run the coarse network on the images
echo "Running coarse net"
python3 use_network.py run images data_filter.txt -c -o coarse_output

#run the detailed network on the images
echo "Running detailed net"
python3 use_network.py run images data_filter.txt -o detailed_output

