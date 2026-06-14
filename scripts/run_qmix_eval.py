# scripts/run_qmix_eval.py
import sys, os
sys.path.insert(0, '.')
from algos.qmix.eval_qmix import run_eval

ckpt = 'results/qmix/checkpoints_10agents/qmix_step_500000.pt'

for scenario in ['easy', 'medium', 'hard']:
    print(f"\n{'='*50}")
    print(f"Evaluating QMIX — {scenario}")
    print('='*50)
    run_eval(
        checkpoint_path = ckpt,
        scenario        = scenario,
        n_episodes      = 30,
        n_agents        = 10,
        results_dir     = 'results/qmix',
        record_weights  = False,
    )