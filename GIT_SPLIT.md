# Go back to main
git checkout main

# Delete the existing (incorrect) branch
git branch -D deploy/rag-eval-dataset

# Now split properly – this will create a fresh branch
git subtree split -P 05-projects/02-synthesize-rag-eval-dataset -b deploy/rag-eval-dataset

# Checkout the new branch
git checkout deploy/rag-eval-dataset

# Verify – you should ONLY see the contents of 02-synthesize-rag-eval-dataset
ls