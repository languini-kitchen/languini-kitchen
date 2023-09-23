#!/usr/bin/env bash

known_manifest_hash="56c3e00f8b9cf2d3ecaec145923757453b9f82df39f36b78dc974d6344142913"
local_manifest_hash="$(du -ab ./data/books/books_16384/files/ | sort -n | sha256sum | cut -d' ' -f1)"

if [ "$local_manifest_hash" != "$known_manifest_hash" ]; then
	echo "Tokenised file size manifest does not match! Dataset is broken or tokenisation was interrupted. Your hash is $local_manifest_hash, expected $known_manifest_hash" >&2
	exit 1
else
	echo "Tokenised file size manifest matches! Your data looks good." >&2
fi
