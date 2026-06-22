Local Machine Commands
Activate virtual environment:
bashsource /Users/gauravbhasin/Desktop/rl-takehome-lz/.venv/bin/activate


Test imports locally:
bashpython -c "import main; import evaluator; import utils; import llms; import rl_datasets; print('All imports OK')"


Push to GitHub:
bashcd /Users/gauravbhasin/Desktop/rl-takehome-lz && git add . && git commit -m "message" && git push


Copy single file to Vast.ai:
bashscp -P 42617 -i ~/.ssh/id_ed25519 /Users/gauravbhasin/Desktop/rl-takehome-lz/FILENAME root@38.57.2.194:/workspace/rl-takehome-lz/

Copy entire folder to Vast.ai:
bashscp -P 42617 -i ~/.ssh/id_ed25519 -r /Users/gauravbhasin/Desktop/rl-takehome-lz/ root@38.57.2.194:/workspace/

Vast.ai Commands
SSH into instance:
bashssh -p 42617 root@38.57.2.194 -i ~/.ssh/id_ed25519 -L 8080:localhost:8080

Navigate to project:
bashcd /workspace/rl-takehome-lz

Activate virtual environment:
bashsource .venv/bin/activate


Test imports:
bashpython -c "import main; import evaluator; import utils; import llms; import rl_datasets; print('All imports OK')"


Install dependencies:
bashuv sync

Run 2 iteration test:
bashpython main.py --num_train_iters 2 --eval_iterations 0 --save_steps 0 --verbose


Run full training:
bashpython main.py --num_train_iters 1000 --eval_iterations 20 --save_steps 100 --verbose


Generate plots:
bashpython plotter.py --output_dir output