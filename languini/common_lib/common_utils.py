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

import torch
import random 


def traverse(tree, func):
    """Applies func to all elements of a nested list or dict structure. """

    if isinstance(tree, list) or isinstance(tree, tuple):
        return [traverse(lst, func) for lst in tree]
    
    if isinstance(tree, dict):
        return {k: traverse(tree[k], func) for k in tree.keys()}
    
    return func(tree)


def flatten(tree):
    """Applies func to all elements of a nested list or dict structure. """        
    if isinstance(tree, list) or isinstance(tree, tuple):
        lst = []
        for l in tree:
            lst += flatten(l)
        return lst
    
    if isinstance(tree, dict):
        lst = []
        for key in tree.keys():
            lst += flatten(tree[key])
        return lst
    
    return [tree]


def get_total_tensor_size(tensor):
    """Returns the total sum of elements in a datastructure made up of tensors."""
    def only_count_tensors(x):
        if x is not None and isinstance(x, torch.Tensor):
            return int(x.numel())
        else:
            return 0
    
    return sum(flatten(traverse(tensor, func=only_count_tensors)))


if __name__ == "__main__":
    count = 0

    def build(first=True, p=1):
        global count
        if random.random() > 0.6**p and not first:
            count += 1
            return torch.randn(10)
        if random.random() > 0.5:
            return [build(first=False, p=p+1) for _ in range(5)]
        else:
            return {f"{i}":build(first=False, p=p+1) for i in range(6)}
    
    random.seed(1)
    x = build()
    print(count)

    y = traverse(x, func=lambda x: x.numel() == 10)
    y = flatten(y)

    print(len(y))
    print(all(y))