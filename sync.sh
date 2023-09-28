#!/bin/bash

# Syncs your current languini-kitchen folder to the server's home directory without data, logs, venv, etc.

# Check if an argument is provided
if [ "$#" -eq 0 ]; then
    # If no argument is provided, display a help message
    echo "Usage: ./sync.sh <server_or_username@server>"
    echo "Please provide a server or username@server as an argument."
    exit 1
fi

# Get the argument
SERVER=$1

# Get the current folder's absolute path
SOURCE_FOLDER="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

# Run the rsync command
rsync -av --progress --delete --force --delete-before "$SOURCE_FOLDER" $SERVER:~/. --exclude logs  --exclude __pycache__ --exclude wandb --exclude data --exclude venv
