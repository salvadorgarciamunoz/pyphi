#!/bin/bash
# rebuild_docs.sh
# Run from pyphi-master/docs/ to rebuild and publish documentation to GitHub Pages

set -e  # stop on any error

echo ">>> Cleaning previous build..."
make clean

echo ">>> Building HTML..."
make html

echo ">>> Copying to docs/ root for GitHub Pages..."
cp -r build/html/. .

echo ">>> Staging docs..."
cd ..
git add docs/

echo ">>> Committing..."
git commit -m "Rebuild docs"

echo ">>> Pushing to GitHub..."
git push origin main

echo ">>> Done! Site will update at https://salvadorgarciamunoz.github.io/pyphi/ in ~1 minute."
