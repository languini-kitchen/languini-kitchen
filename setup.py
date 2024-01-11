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

from setuptools import setup
from setuptools import find_packages

"""setup.py for the Languini Kitchen.

Install for development:
  pip install -e .
"""

install_requires_core = [
    "torch==2.0",
    "numpy",
    "munch",
    "wandb",
    "sentencepiece",
    "tensorboard",
    "tqdm",
    "pandas",
    "jupyter",
    "deepspeed==0.10.3",
    "seaborn",
]

setup(
    name='languini',
    version='0.0.2',    
    description='LM Research Environment',
    url='none',
    author='Imanol Schlag',
    author_email='imanol.schlag@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=install_requires_core,
    classifiers=[],
    keywords='Languini'
)