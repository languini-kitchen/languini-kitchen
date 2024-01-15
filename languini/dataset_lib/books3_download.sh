#!/bin/bash

# Run this script from the languini-kitchen root folder.
# Example:
# ~/languini-kitchen$ ./languini/dataset_lib/books3_download.sh

destination="data"

# Calculate size in kilobytes
# 1 Gigabyte is 1,073,741,824 bytes or 1,048,576 Kilobytes
required_space=$(( (37 + 102) * 1048576 )) # space needed for tar file and extracted content

# Make sure destination exists
mkdir -p "$destination"

# Check if there is enough disk space
available_space=$(df "$destination" | awk 'NR==2 {print $4}') # disk free space in KB

if [ "$available_space" -lt "$required_space" ]; then
    echo "Insufficient disk space. You need at least $(($required_space / 1048576)) GB."
    exit 1
fi

# Download and extract the dataset
echo "Downloading dataset..."
# previous hosts
# https://the-eye.eu/public/AI/pile_preliminary_components/books3.tar.gz
# https://thenose.cc/public/AI/EleutherAI_ThePile_v1/pile_preliminary_components/books3.tar.gz
wget --continue http://62.212.86.148/datasets/EleutherAI_ThePile_v1/pile_preliminary_components/books3.tar.gz -P "$destination"

tar -xvzf "$destination/books3.tar.gz" -C "$destination"

echo "Done."

