#!/usr/bin/env python3
"""
レベル1: 推論テキストビューアー
条件A/B/Cのresults.jsonlから正解・不正解の推論テキストを表示する
"""
import json, sys, os, argparse

def view_results(jsonl_path, show_correct=True, show_incorrect=True, max_show=3):
    if not os.path.exists(jsonl_path):
        print(f"ファイルが見つかりません: {jsonl_path}")
        return

    correct_cases = []
    incorrect_cases = []

    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if r["is_correct"]:
                correct_cases.append(r)
            else:
                incorrect_cases.append(r)

    total = len(correct_cases) + len(incorrect_cases)
    print(f"\n{'='*70}")
    print(f"条件 {correct_cases[0]['condition'] if correct_cases else incorrect_cases[0]['condition']} | "
          f"正解: {len(correct_cases)}/{total} = {len(correct_cases)/total:.1%}")
    print(f"{'='*70}")

    if show_correct and correct_cases:
        print(f"\n【正解した問題】（最大{max_show}件）")
        for r in correct_cases[:max_show]:
            print(f"\n--- {r['question_id']} | type: {r['task_type']} | level: {r['task_level']} ---")
            print(f"[問題]\n{r['question']}\n")
            print(f"[モデルの推論]\n{r['model_output']}\n")
            print(f"[予測] {r['predicted_answer']}  [正解] {r['normalized_gt']}")

    if show_incorrect and incorrect_cases:
        print(f"\n【不正解の問題】（最大{max_show}件）")
        for r in incorrect_cases[:max_show]:
            print(f"\n--- {r['question_id']} | type: {r['task_type']} | level: {r['task_level']} ---")
            print(f"[問題]\n{r['question']}\n")
            print(f"[モデルの推論]\n{r['model_output']}\n")
            print(f"[予測] {r['predicted_answer']}  [正解] {r['normalized_gt']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="推論テキストビューアー")
    parser.add_argument("--results_dir", type=str, default="./results_inference_350",
                        help="results_inferenceディレクトリのパス")
    parser.add_argument("--condition", type=str, choices=["A","B","C","all"], default="all")
    parser.add_argument("--max_show", type=int, default=2, help="表示する最大件数")
    parser.add_argument("--correct_only", action="store_true")
    parser.add_argument("--incorrect_only", action="store_true")
    args = parser.parse_args()

    show_correct  = not args.incorrect_only
    show_incorrect = not args.correct_only

    conditions = ["A","B","C"] if args.condition == "all" else [args.condition]
    for cond in conditions:
        path = os.path.join(args.results_dir, f"condition_{cond}", f"condition_{cond}_results.jsonl")
        view_results(path, show_correct=show_correct, show_incorrect=show_incorrect, max_show=args.max_show)