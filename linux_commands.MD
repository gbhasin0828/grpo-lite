Local Machine Commands
Activate virtual environment:
source /Users/gauravbhasin/Desktop/rl-takehome-lz/.venv/bin/activate


Test imports locally:
python -c "import main; import evaluator; import utils; import llms; import rl_datasets; print('All imports OK')"


Push to GitHub:
cd /Users/gauravbhasin/Desktop/rl-takehome-lz && git add . && git commit -m "message" && git push


Copy single file to Vast.ai:
scp -P 42617 -i ~/.ssh/id_ed25519 /Users/gauravbhasin/Desktop/rl-takehome-lz/FILENAME root@38.57.2.194:/workspace/rl-takehome-lz/

Copy entire folder to Vast.ai:
rsync -avz --exclude='.venv' --exclude='.git' --exclude='__pycache__' \
  -e "ssh -p 42617 -i ~/.ssh/id_ed25519" \
  /Users/gauravbhasin/Desktop/rl-takehome-lz/ \
  root@38.57.2.194:/workspace/rl-takehome-lz/

Vast.ai Commands
SSH into instance:
ssh -p 42617 root@38.57.2.194 -i ~/.ssh/id_ed25519 -L 8080:localhost:8080
Then once connected:
cd /workspace/rl-takehome-lz && source .venv/bin/activate

Navigate to project:
cd /workspace/rl-takehome-lz

Activate virtual environment:
source .venv/bin/activate


Test imports:
python -c "import main; import evaluator; import utils; import llms; import rl_datasets; print('All imports OK')"


Install dependencies:
uv sync

Run 2 iteration test:
python main.py --num_train_iters 2 --eval_iterations 0 --save_steps 0 --verbose


Run full training:
python main.py --num_train_iters 1000 --eval_iterations 20 --save_steps 100 --verbose


Generate plots:
python plotter.py --output_dir output