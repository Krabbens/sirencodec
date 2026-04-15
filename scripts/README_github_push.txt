GitHub push (after git-lfs install if you track .pt/.npy):

  gh repo create sirencodec --private --source=. --remote=origin --push
  # or:
  git remote add origin git@github.com:YOUR_USER/sirencodec.git
  git lfs install
  git push -u origin main

Replace YOUR_USER/sirencodec with your repository.
