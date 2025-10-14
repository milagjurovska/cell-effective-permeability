import argparse, numpy as np
from niapy.task import Task, OptimizationType
from niapy.algorithms.basic import ArtificialBeeColonyAlgorithm

from .space import decode_vector
from .problem import build_problem
from ._finalize import train_with_best_to_summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gat","gcn","gin"], required=True)
    ap.add_argument("--csv", default="data/caco2_processed.csv")
    ap.add_argument("--target", default="target")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_dir", default="runs")
    args = ap.parse_args()

    algo_name = "abc"

    problem = build_problem(args.model, args.csv, args.target, args.batch_size,
                            args.epochs, args.patience, args.fold, args.seed)
    task = Task(problem=problem, max_evals=args.trials, optimization_type=OptimizationType.MINIMIZATION)
    algo = ArtificialBeeColonyAlgorithm(population_size=args.pop, limit=args.limit, seed=args.seed)
    best_x, best_f = algo.run(task)
    best = decode_vector(np.array(best_x, float), args.model)

    out_path = train_with_best_to_summary(
        model_name=args.model, best=best, epochs=args.epochs, patience=args.patience,
        csv=args.csv, target=args.target, batch_size=args.batch_size, seed=args.seed,
        out_dir=args.run_dir, algo_name=algo_name, fold=args.fold,
        best_val_rmse_from_search=float(best_f), trials=args.trials,
    )
    print(f"[HPO] Single-file summary saved → {out_path}")

if __name__ == "__main__":
    main()
