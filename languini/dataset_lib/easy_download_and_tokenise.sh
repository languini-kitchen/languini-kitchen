#!/bin/bash

# This is an easy to use script which will download the books3 data and the file names 
# for the languini splits and then tokenise them using the default 16k spm vocab. 
# Downloading may take a few hours depending on the download speed.
# Tokenising may also take several hours depending on the CPUs of the machine. 
# 
# Example:
# ./languini/dataset_lib/easy_download_and_tokenise.sh
#

# Check if the script is run relatively to the root of languini-kitchen
if [ ! -f $PWD/languini/dataset_lib/easy_download_and_tokenise.sh ]; then
    echo "Error: Run this script from the root of the repository."
    exit 1
fi

mkdir -p data

# Download the dataset
./languini/dataset_lib/books3_download.sh

# Download the list of file names for the languini books splits
mkdir -p data/books
wget https://zenodo.org/record/8375423/files/file_list_test_iid.npy -P data/books
wget https://zenodo.org/record/8375423/files/file_list_test_java.npy -P data/books
wget https://zenodo.org/record/8375423/files/file_list_test_langlearn.npy -P data/books
wget https://zenodo.org/record/8375423/files/file_list_test_statistics.npy -P data/books
wget https://zenodo.org/record/8375423/files/file_list_test_woodworking.npy -P data/books
wget https://zenodo.org/record/8375423/files/file_list_test_discworld.npy -P data/books
wget https://zenodo.org/record/8375423/files/file_list_train.npy -P data/books

# Tokenise the languini books data with the default 16k sentencepiece vocabulary.
python3 languini/dataset_lib/tokenise_languini_books.py --split_npy_file data/books/file_list_test_langlearn.npy --spm_model languini/vocabs/spm_models/books_16384.model
python3 languini/dataset_lib/tokenise_languini_books.py --split_npy_file data/books/file_list_test_statistics.npy --spm_model languini/vocabs/spm_models/books_16384.model
python3 languini/dataset_lib/tokenise_languini_books.py --split_npy_file data/books/file_list_test_woodworking.npy --spm_model languini/vocabs/spm_models/books_16384.model
python3 languini/dataset_lib/tokenise_languini_books.py --split_npy_file data/books/file_list_test_java.npy --spm_model languini/vocabs/spm_models/books_16384.model
python3 languini/dataset_lib/tokenise_languini_books.py --split_npy_file data/books/file_list_test_discworld.npy --spm_model languini/vocabs/spm_models/books_16384.model
python3 languini/dataset_lib/tokenise_languini_books.py --split_npy_file data/books/file_list_test_iid.npy --spm_model languini/vocabs/spm_models/books_16384.model
python3 languini/dataset_lib/tokenise_languini_books.py --split_npy_file data/books/file_list_train.npy --spm_model languini/vocabs/spm_models/books_16384.model


