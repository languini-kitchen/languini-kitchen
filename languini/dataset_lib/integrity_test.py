import os
import hashlib
import sys

def main(directory):
    """
    This script traverses the specified directory, retrieves file names (excluding paths) 
    and their sizes, and sorts them first by name (lexicographically) and then by size 
    (numerically). A SHA-256 hash of the sorted list is computed and printed.
    
    :param directory: The directory path to traverse.
    """
    file_list = []
    for foldername, _, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(foldername, filename)
            size = os.path.getsize(filepath)
            file_list.append((filename, size))
            
    file_list.sort(key=lambda x: (x[0], x[1]))
    
    hasher = hashlib.sha256()
    
    for filename, size in file_list:
        line = f"{size}\t{filename}\n"
        hasher.update(line.encode())
    
    print(hasher.hexdigest())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <directory_path>")
        print("directory_path: The path of the directory to traverse.")
        sys.exit(1)
    main(sys.argv[1])