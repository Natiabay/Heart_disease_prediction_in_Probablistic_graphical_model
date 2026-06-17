#!/bin/bash
# Push heart-disease-bn to https://github.com/Natiabay/Heart_disease_prediction_in_Probablistic_graphical_model
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ ! -d .git ]]; then
  echo "ERROR: Run this script from heart-disease-bn (this folder must contain .git)."
  exit 1
fi

TOP="$(git rev-parse --show-toplevel)"
if [[ "$TOP" != "$ROOT" ]]; then
  echo "ERROR: git root is $TOP but expected $ROOT"
  exit 1
fi

PARENT_GIT="$(dirname "$ROOT")/.git"
if [[ -d "$PARENT_GIT" ]]; then
  echo "WARNING: Found extra git repo in parent folder PGM!!!!"
  echo "  That empty repo causes: 'src refspec main does not match any'"
  echo "  Remove it once (safe — heart-disease-bn has its own .git):"
  echo "    rm -rf \"$(dirname "$ROOT")/.git\""
  echo
fi

BRANCH="$(git branch --show-current)"
if [[ "$BRANCH" != "main" ]]; then
  echo "ERROR: Expected branch 'main', got '$BRANCH'"
  exit 1
fi

if ! git rev-parse HEAD >/dev/null 2>&1; then
  echo "ERROR: No commits yet. Run: python run.py  (optional) then git add . && git commit"
  exit 1
fi

echo "Pushing $ROOT (branch: $BRANCH) → origin"
echo "Remote: $(git remote get-url origin)"
echo

if ! git push -u origin main; then
  echo
  echo "Push failed — authenticate with GitHub first:"
  echo "  1. Install GitHub CLI:  sudo apt install gh"
  echo "  2. Login:               gh auth login"
  echo "  3. Retry:               ./push_to_github.sh"
  echo
  echo "Or use a Personal Access Token:"
  echo "  git push https://YOUR_TOKEN@github.com/Natiabay/Heart_disease_prediction_in_Probablistic_graphical_model.git main"
  exit 1
fi

echo
echo "Success! Live app: https://heartdiseasepredictiondemo.streamlit.app/"
echo "  Repo: Natiabay/Heart_disease_prediction_in_Probablistic_graphical_model"
echo "  Main file: streamlit_app.py"
