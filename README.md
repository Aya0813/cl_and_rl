
env setup: 
uv venv --python 3.11
source .venv/bin/activate

Commands run: 
python3 -m pip install datasets


TO DO: 
- setup an env 
- setup hugging face 


git init
touch .gitignore (create an empty file instantly if it does not exist yet...)

.venv/
__pycache__/
*.pyc
data/
*.pt
*.bin

git add .


git commit -m "Initial commit: PoC curriculum learning pipeline"

Create a git repo (from the git website)

git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git


git branch -M main
git push -u origin main

