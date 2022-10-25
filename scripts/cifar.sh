#!/bin/bash

# Move to the base directory
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
echo "parent_path: $parent_path"
cd "$parent_path"

# Download and unzip the data 
cd ../data
mkdir cifar
cd cifar
wget -P ../data/ https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

tar -xf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz
cd cifar-10-batches-py
mv * ../
cd ..
rm cifar-10-batches-py

