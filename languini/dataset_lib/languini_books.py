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
import torch
import numpy as np


def distribute_docs(book_lengths, batch_size):
    """
    Given a book_length object (loaded from book_lengths.npy), this function 
    distributes all documents into batch_size lists and returns them. 
    batch_size ought to be a multiple of the number of workers so that each 
    worker knows which books it needs to load. 
    
    Parameters:
        book_lengths: list
            A list of tuples containg the path to a file and its length.

        batch_size: int
            Number of lists to split the data into.

    Returns:
        list of list of str:
            A list containing batch_size lists of file paths
        list of list of int:
            A list containing batch_size lists of the number of token in the respective file
        list of int:
            The total number of tokens for each queue.        
    """
    
    # Create a list to hold subset of the file paths
    file_queues = [[] for _ in range(batch_size)]

    # Create a list to hold the token count of each file
    token_counts = [[] for _ in range(batch_size)]
    
    # Create a list to hold the amount of tokens in each doc queue
    queue_total_tokens = [0 for _ in range(batch_size)]
    
    # Now, distribute the documents by adding the next doc to the 
    # queue with the smallest load    
    for path, length in book_lengths:        
        # select shortest queue
        min_queue = np.argmin(queue_total_tokens)
        
        # add the current document and its length to the lists
        file_queues[min_queue].append(path)
        token_counts[min_queue].append(int(length))
        
        # Increase the total token count of the queue by the length of the document
        queue_total_tokens[min_queue] += int(length)
    
    return file_queues, token_counts, queue_total_tokens


class LanguiniDatasetIterator:
    def __init__(self, data_path, split, global_batch_size, batch_idxs, micro_batches, sequence_length, device, end_of_doc_token, shift_n=-1, repeat=False, buffer_size=4096*10):
        """
        Initialise the dataset iterator. 

        Args:
            data_path (str): path to the dataset root folder which contains book_lengths.npy and the files folder.
            split (str): data split: train or test
            global_batch_size (int): batch size across all devices.
            batch_idxs (list of int): list of elements from the global batch which are processed by this device.
            micro_batches (int): split the batch into micro batches for each gradient accumulation step.
            sequence_length (int): sequence length of each batch.
            device (int): cuda device to which we will copy the data.
            end_of_doc_token (int): id of the end of document token in the vocab to be inserted between docs.
            shift_n (int): The number of tokens to remove from the queue. If stream_n == -1, it will remove the entire batch from the queue. Otherwise, just stream_n tokens. 
            repeat (bool): if a document queue should just repeat once it reached the end.
            buffer_size (int): size of the token buffer.
        """
        self.data_path = data_path
        self.split = split
        self.global_batch_size = global_batch_size
        self.batch_idxs = batch_idxs
        self.bsz = len(self.batch_idxs)  # local batch size (device)
        self.micro_batches = micro_batches
        self.seq_len = sequence_length
        self.device = device
        self.end_of_doc_token = end_of_doc_token
        self.repeat = repeat
        self.shift_n = shift_n
        self.buffer_size = buffer_size

        # The number of tokens we shift after a batch is the sequence length if shift_n == -1. Otherwise it is the sequence length. It cannot be 0 or larger than the sequence length.
        assert self.shift_n <= self.seq_len and self.shift_n != 0, "Cannot shift 0 tokens or more tokens than there are in a sequence!"

        # Ensure that batch_idxs is within [0, global_batch_size)
        assert all(0 <= idx < global_batch_size for idx in batch_idxs), "All batch_idxs must be in range [0, global_batch_size)"
        
        # The trainer doesn't support evaluation with microbatches yet.
        if self.split != 'train':
            assert self.micro_batches == 1, "Currently the trainer only support 1 gradient accumulation step. Microbatches > 1 is only supported for the train split."

        # check if local batch size is divisible by microbatches
        assert self.bsz % self.micro_batches == 0, f"Device batch size {self.bsz} is not divisible by {self.micro_batches} micro batches for gradient accumulation."

        # Load the book lengths and keep the queues which matter for this process.
        data_root = os.path.split(data_path)[0]
        if split == "train":
            book_lengths = np.load(os.path.join(data_root, "file_list_train.npy"))
        elif split == "test":
            book_lengths = np.load(os.path.join(data_root, "file_list_test_iid.npy"))
        elif split == "langlearn":
            book_lengths = np.load(os.path.join(data_root, "file_list_test_langlearn.npy"))
        elif split == "discworld":
            book_lengths = np.load(os.path.join(data_root, "file_list_test_discworld.npy"))
        elif split == "java":
            book_lengths = np.load(os.path.join(data_root, "file_list_test_java.npy"))
        elif split == "stats":
            book_lengths = np.load(os.path.join(data_root, "file_list_test_statistics.npy"))
        elif split == "wood":
            book_lengths = np.load(os.path.join(data_root, "file_list_test_woodworking.npy"))
        else:
            raise ValueError(f"{split} is an invalid split.")
        all_file_paths, token_counts, total_tokens = distribute_docs(book_lengths, batch_size=self.global_batch_size)
        self.file_paths = [all_file_paths[idx] for idx in self.batch_idxs]
        self.reset()

    def reset(self):
        # store the current document and position
        self.curr_file_idx = [0] * self.bsz
        self.curr_file_pos = [0] * self.bsz
        
        # current files and buffers to read less often
        self.curr_files = [self._get_memmap(os.path.join(self.data_path,self.file_paths[i][0])) for i in range(self.bsz)]
        self.buffer_pos = [self.buffer_size] * self.bsz  # this will load the proper amount upon the first fill.
        self.buffers = np.zeros(shape=(self.bsz, self.buffer_size))
        self.is_depleted = [False] * self.bsz  # will become true if there are no more documents left
    
    def __iter__(self):
        return self
    
    def _get_memmap(self, npy_file):
        # Load the numpy file normally
        np_array = np.load(npy_file, mmap_mode='r')
        # Get a memmap object of the loaded numpy array's data buffer
        return np.lib.format.open_memmap(npy_file, mode='r', dtype=np_array.dtype, shape=np_array.shape)

    def _fill_buffer(self, index):
        """
        Method to refill the buffer with new data from the current file. If the current
        file doesn't have enough data left, it loads new data until the end, increases the current file index,
        updates the memory-mapped file with the next file from the file paths, adds the end of document token,
        and reads the remaining data from the next file.
        """
        buffer_full = False
        
        # Keep the buffer data after buffer_pos
        new_data = self.buffers[index][self.buffer_pos[index]:]

        while not buffer_full:
            # Compute the amount of data to load
            load_size = self.buffer_size - len(new_data)

            # Compute the remaining length of the current file
            remaining_length = len(self.curr_files[index]) - self.curr_file_pos[index]

            # Check if current file has enough data left (+1 for the end of document token that will be added to the end of a file)
            if load_size <= remaining_length:
                # Load data from the current file into the buffer
                new_data = np.concatenate((new_data, self.curr_files[index][self.curr_file_pos[index] : self.curr_file_pos[index] + load_size]))
                
                # Increase current file position by the amount of data that was loaded
                self.curr_file_pos[index] += load_size

                # buffer is full again
                buffer_full = True

            else:  # the current file is not enough
                # Load the remaining data from the current file
                new_data = np.concatenate((new_data, self.curr_files[index][self.curr_file_pos[index]:]))

                # Increase the current file index
                self.curr_file_idx[index] += 1

                # If the current file index is out of bounds, reset it to 0 (if repeat is enabled)
                if self.curr_file_idx[index] == len(self.file_paths[index]):
                    if self.repeat:
                        self.curr_file_idx[index] = 0
                    else: # there are no more files! So we write the data we have to the buffer
                        # mark this source of data as depleted
                        self.is_depleted[index] = True

                        # Add the end of document token to the remaining data 
                        new_data = np.append(new_data, self.end_of_doc_token)

                        # Write the remaining data to the end of the buffer
                        starting_pos = self.buffer_size - len(new_data)
                        self.buffers[index][starting_pos:] = new_data
                        self.buffer_pos[index] = starting_pos

                        return

                # Map the next file into memory and set the pointer to 0
                filepath = os.path.join(self.data_path, self.file_paths[index][self.curr_file_idx[index]])
                self.curr_files[index] = self._get_memmap(filepath)
                self.curr_file_pos[index] = 0

                # Add the end of document token before loading from the next document
                new_data = np.append(new_data, self.end_of_doc_token) 

                # restart while loop
                buffer_full = False
            
        # Replace the old buffer with the new data
        self.buffers[index] = new_data

        # Reset buffer position
        self.buffer_pos[index] = 0
    
    def __next__(self):
        """
        Method to obtain the next batch of sequences. It will check if the buffer needs to be refilled. 
        If so, it calls the _fill_buffer method (sequentially). After that, it builds a new batch from the buffer.
        """
        for i in range(self.bsz):
            if not self.is_depleted[i]:
                # Check if the buffer for the current batch index needs to be refilled
                if self.buffer_pos[i] + self.seq_len + 1 > self.buffer_size:
                    self._fill_buffer(i)
        
        # Create an empty tensor in RAM to store the data
        seq = torch.zeros(size=(self.bsz, self.seq_len + 1), dtype=torch.int64)
        is_empty = [False] * self.bsz
        is_padded = False

        # iterate through all queues
        for i in range(self.bsz):

            # if the queue is not depleted, just copy over the data
            if not self.is_depleted[i]:
                # Extract sequence + 1 from the buffer and copy to seq
                seq[i, :] = torch.from_numpy(self.buffers[i][self.buffer_pos[i] : self.buffer_pos[i] + self.seq_len + 1])

                # Update the buffer position
                if self.shift_n == -1:
                    self.buffer_pos[i] += self.seq_len
                else:
                    self.buffer_pos[i] += self.shift_n
            
            # if it is, check if buffer pos is not at the end
            elif self.buffer_pos[i] != self.buffer_size - 1:
                
                # check if the buffer has enough data, copy from buffer to seq
                if self.buffer_pos[i] + self.seq_len + 1 <= self.buffer_size:
                    seq[i, :] = torch.from_numpy(self.buffers[i][self.buffer_pos[i] : self.buffer_pos[i] + self.seq_len + 1])

                    # Update the buffer position
                    if self.shift_n == -1:
                        self.buffer_pos[i] += self.seq_len
                    else:
                        self.buffer_pos[i] += self.shift_n
                
                # otherwise, copy the buffer leftover to the beginning of seq
                else:
                    temp = torch.from_numpy(self.buffers[i][self.buffer_pos[i]:])
                    pos = temp.shape[0]
                    seq[i, :pos] = temp
                    self.buffer_pos[i] = self.buffer_size - 1
                    is_empty[i] = False
                    is_padded = True

            else:
                # queue is depleted and buffer is already at the end -> nothing to do
                is_empty[i] = True
                is_padded = True
                pass
        
        # if all queues are empty raise an exception
        if all(is_empty):
            raise StopIteration
        
        # Move data to the device
        seq = torch.reshape(seq, (self.micro_batches, self.bsz // self.micro_batches, self.seq_len + 1))
        seq = seq.to(self.device, non_blocking=True)

        # Create batches
        batch_x = seq[:, :, :-1]
        batch_y = seq[:, :, 1:]

        return batch_x, batch_y, is_padded
