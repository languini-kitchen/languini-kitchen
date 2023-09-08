#!/bin/bash

# This is an easy to use script which will download the books3 data and the file names 
# for the languini splits and then tokenise them using the default 16k spm vocab. 
# Downloading may take a few hours depending on the download speed.
# Tokenising may also take several hours depending on the CPUs of the machine. 
# 
# Example:
# ./languini/dataset_lib/download_and_tokenise.sh
#

#!/bin/bash

# Check if current folder is named "languini-kitchen"
if [[ $(basename "$PWD") != "languini-kitchen" ]]; then
    echo "Error: Run this script from the root of 'languini-kitchen'."
    exit 1
fi

# Check if a symlink called "data" exists in the current directory
if [[ ! -L "data" ]]; then
    echo "Error: 'data' symlink does not exist. See README."
    exit 1
fi

# Download the dataset
chmod +x languini/dataset_lib/books3_download.sh
bash ./languini/dataset_lib/books3_download.sh

# Download the list of file names for the languini books splits
mkdir -p data/books
wget https://zenodo.org/record/8326484/files/file_list_test_iid.npy -P data/books
wget https://zenodo.org/record/8326484/files/file_list_test_java.npy -P data/books
wget https://zenodo.org/record/8326484/files/file_list_test_langlearn.npy -P data/books
wget https://zenodo.org/record/8326484/files/file_list_test_statistics.npy -P data/books
wget https://zenodo.org/record/8326484/files/file_list_test_woodworking.npy -P data/books
wget https://zenodo.org/record/8326484/files/file_list_train.npy -P data/books

# Tokenise the languini books data with the default 16k sentencepiece vocabulary.
python3 languini/dataset_lib/tokenise_languini_books.py --split_npy_file data/books/file_list_test_langlearn.npy --spm_model languini/vocabs/spm_models/books_16384.model
python3 languini/dataset_lib/tokenise_languini_books.py --split_npy_file data/books/file_list_test_statistics.npy --spm_model languini/vocabs/spm_models/books_16384.model
python3 languini/dataset_lib/tokenise_languini_books.py --split_npy_file data/books/file_list_test_woodworking.npy --spm_model languini/vocabs/spm_models/books_16384.model
python3 languini/dataset_lib/tokenise_languini_books.py --split_npy_file data/books/file_list_test_java.npy --spm_model languini/vocabs/spm_models/books_16384.model
python3 languini/dataset_lib/tokenise_languini_books.py --split_npy_file data/books/file_list_test_iid.npy --spm_model languini/vocabs/spm_models/books_16384.model
python3 languini/dataset_lib/tokenise_languini_books.py --split_npy_file data/books/file_list_train.npy --spm_model languini/vocabs/spm_models/books_16384.model


