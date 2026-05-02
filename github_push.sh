#!/bin/bash

# Exit on error
set -e

# Configuration
REPO_URL=$1
COMMIT_MSG=${2:-"Initial commit: End-to-End Smart Waste Classifier"}

if [ -z "$REPO_URL" ]; then
  echo "Usage: ./github_push.sh <github_repository_url> [commit_message]"
  echo "Example: ./github_push.sh https://github.com/yourusername/waste-classifier.git"
  exit 1
fi

echo "♻️ Initializing Git repository..."
# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    git init
    # Add sensible gitignore
    echo ".venv/" > .gitignore
    echo "__pycache__/" >> .gitignore
    echo "data/" >> .gitignore
    echo ".env" >> .gitignore
    echo "*.onnx" >> .gitignore
    echo "*.tar" >> .gitignore
    echo "*.rar" >> .gitignore
    echo "*.zip" >> .gitignore
fi

echo "♻️ Staging files..."
git add .

echo "♻️ Committing files..."
git commit -m "$COMMIT_MSG" || echo "No changes to commit."

echo "♻️ Setting branch to main..."
git branch -M main

echo "♻️ Adding remote origin..."
git remote add origin "$REPO_URL" 2>/dev/null || git remote set-url origin "$REPO_URL"

echo "♻️ Pushing to GitHub..."
git push -u origin main

echo "✅ Successfully pushed to $REPO_URL"
