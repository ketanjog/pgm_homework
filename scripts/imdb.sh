#!/bin/bash

# Move to the base directory
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
echo "parent_path: $parent_path"
cd "$parent_path"

# Download and unzip the data 
wget -P ../data http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
cd ../data
tar -xf ../data/aclImdb_v1.tar.gz
rm aclImdb_v1.tar.gz
cd ..

# Create the csv file

python pgm/data/create_imdb_data.py







