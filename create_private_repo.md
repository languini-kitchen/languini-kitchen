## Create a private languini repo
Instead of a public fork you may want to use the languini codebase but keep your research in a private GitHub repository. 

1. [Create a new private repository](https://github.com/new) with your personal GitHub account. E.g. named ```my_languini_model```.

2. Clone the languini repo and push the code to your private repo
```
git clone --bare https://github.com/languini-kitchen/languini-kitchen.git
cd languini-kitchen.git
git push --mirror https://github.com/ischlag/my_private_languini_repo.git
cd ..
rm -rf languini-kitchen.git
git clone https://github.com/ischlag/my_private_languini_repo.git
cd my_private_languini_repo
```

3. Proceed with the installation
```
python3 -m venv venv
source venv/bin/activate
```

```
pip install --upgrade pip setuptools
pip install -e . --upgrade
```

4. Download or link to the data as explained in the [README](README.md).