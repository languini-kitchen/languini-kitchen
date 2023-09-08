# Copyright 2023 The Languini Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import csv
import argparse
import numpy as np
import sentencepiece as spm

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def tokenize_file(args):
    """
    Tokenizes a given file using a SentencePiece model and saves the output as a numpy array.

    Args:
        args (tuple): A tuple containing a SentencePiece model, the path of the input file, and the output directory path.
    """
    sp, path, output_folder = args
    outpath = os.path.join(output_folder, "files", os.path.splitext(os.path.basename(path))[0] + ".npy")  # save as .npy
    
    with open(path, 'r') as f:
        text = f.read()
        
    data = np.asarray(sp.EncodeAsIds(text), dtype=get_dtype(sp.GetPieceSize()))
    np.save(outpath, data)


def get_dtype(vocab_size):
    """
    Returns the smallest numpy integer data type that can hold the vocabulary size.

    Args:
        vocab_size (int): Size of the vocabulary.

    Returns:
        numpy.dtype: The smallest integer data type that can hold the vocab size.
    """
    if vocab_size < np.iinfo(np.uint16).max:
        return np.uint16
    elif vocab_size < np.iinfo(np.uint32).max:
        return np.uint32


def validate_files(output_folder, file_names):
    """
    Validates that all files were correctly processed and have non-zero size.

    Args:
        output_folder (str): The directory where the tokenized files are stored.
        file_names (list): List of the file names.

    Raises:
        ValueError: If a file does not exist or has size 0.
    """
    for fn in tqdm(file_names):
        npy_path = os.path.join(output_folder, "files", f"{fn}.npy")
        if not os.path.exists(npy_path):
            raise ValueError(f"File {npy_path} does not exist.")
        data = np.load(npy_path)
        if data.size == 0:
            raise ValueError(f"File {npy_path} has size 0.")


def get_books3_file_paths(books3_dir, file_names):
    """
    Walks through the raw books3 folder and checks if the filenames exist.

    Args:
        books3_dir (str): Path to the books3 root folder.
        file_names (list): List of the file names.

    Raises>
        ValueError: If a file name is not found.
    """
    all_files = {}
    for root, dirs, files in tqdm(list(os.walk(books3_dir))):
        for file_name in files:
            all_files[os.path.splitext(file_name)[0]] = os.path.join(root, file_name)
    
    books3_paths = []
    for fn in file_names:
        if not fn in all_files.keys():
            raise ValueError(f"Filename {fn} not found when walking through {books3_dir}")
        else:
            books3_paths.append(all_files[fn])
    
    return list(books3_paths)
            

def main(books3_dir, spm_model_path, split_npy_file, output_dir):
    """
    Main function to tokenise all files in parallel using multiple processes.

    Args:
        books3_dir (str): Path to the books3 folder.
        spm_model_path (str): Path to the SentencePiece model.
        split_npy_file (str): Path to the npy file containing file paths.
        output_dir (str): Path to the output directory where tokenized files will be stored.
    """
    # Load SentencePiece model
    print("Load SentencePiece model ...")
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_model_path)

    # Load file_list
    books3_dir = os.path.join(books3_dir, "books3")
    print(f"Checking if all files exist in {books3_dir} ...")
    file_names = [os.path.splitext(os.path.basename(fn))[0] for fn in np.load(split_npy_file)[:,0]]
    books3_file_paths = get_books3_file_paths(books3_dir, file_names)
    
    # Create output folder
    vocab_name = os.path.splitext(os.path.basename(spm_model_path))[0]
    output_folder = os.path.join(output_dir, vocab_name)
    print(f"Creating output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "files"), exist_ok=True)  # Sub-folder for files

    # Tokenise 
    print("Tokenise all files ...")
    args = [(sp, path, output_folder) for path in books3_file_paths]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(tokenize_file, args), total=len(books3_file_paths)))
    
    # Validate and get book names and lengths
    print("Checking tokenised files ...")
    validate_files(output_folder, file_names)
    print("done!")

if __name__ == "__main__":
    # Use argparse to get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_npy_file", required=True, help="Path to the Languini npy file containing selected books")
    parser.add_argument("--spm_model", required=True, help="Path to SentencePiece model")
    parser.add_argument("--books3_dir", default='data', help="Path to books3 directory with the raw dataset")
    parser.add_argument("--output_dir", default='data/books', help="Path to the output directory")
    args = parser.parse_args()

    main(args.books3_dir, 
         args.spm_model, 
         args.split_npy_file, 
         args.output_dir)
