#!/usr/bin/env python3
"""
レベル2: Hidden States 可視化スクリプト
========================================

Senderがテスト問題を処理したときの hidden states を収集し、
PCA / t-SNE で2次元に次元削減して可視化する。

可視化の種類:
  1. 正解 vs 不正解 の hidden states 分布
  2. 条件B vs 条件C の hidden states 分布（同一問題で比較）
  3. 科目 (task_type) ごとの分布
  4. 難易度 (task_level) ごとの分布

使い方:
  python visualize_hidden_states.py \
    --results_dir ./results_inference_350 \
    --sender_model_path Qwen/Qwen2.5-0.5B-Instruct \
    --output_dir ./visualizations \
    --num_samples 50 \
    --torch_dtype bfloat16
"""

import os
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Sender の hidden states 収集（inference_math.py と同一ロジック）
# ──────────────────────────────────────────────
SENDER_PROMPT_TEMPLATE = (
    "You are a mathematical problem-solving planner.\n\n"
    "When you receive a math problem (Question), your task is to output a high-level solution plan (Plan)\n"
    "that guides another model to solve the problem in detail.\n\n"
    "IMPORTANT RULES:\n"
    "1. Provide a plan only, not the final answer.\n"
    "2. Keep the plan abstract and general.\n"
    "3. Do not copy or reference any existing solution steps.\n"
    "4. Use the exact output format specified.\n\n"
    "Question:\n{question}"
)


def load_sender(model_path: str, device: torch.device, dtype: torch.dtype):
    logger.info(f"Loading Sender: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map={"": device}
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


@torch.no_grad()
def get_hidden_states(
    model, tokenizer, question: str, device: torch.device,
    max_new_tokens: int = 300, max_hidden_states: int = 10000
) -> np.ndarray:
    """
    Sender で問題を処理し、hidden states を [T, H] の numpy 配列で返す。
    inference_math.py の SenderModel.get_hidden_states と同一ロジック。
    """
    prompt = (
        "<|im_start|>user\n"
        + SENDER_PROMPT_TEMPLATE.format(question=question)
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_beams=1,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        return_dict_in_generate=True,
        output_hidden_states=True,
    )

    steps = outputs.hidden_states
    start_index = max(0, len(steps) - max_hidden_states)
    step_hiddens = []
    for i in range(start_index, len(steps)):
        last_layer = steps[i][-1]
        h_last = last_layer[:, -1, :]  # [1, H]
        step_hiddens.append(h_last.squeeze(0))

    hidden_seq = torch.stack(step_hiddens, dim=0)  # [T, H]
    return hidden_seq.cpu().float().numpy()


def mean_pool(hidden: np.ndarray) -> np.ndarray:
    """[T, H] → [H] に平均プーリング"""
    return hidden.mean(axis=0)


# ──────────────────────────────────────────────
# results.jsonl からメタデータをロード
# ──────────────────────────────────────────────
def load_results(results_dir: str, condition: str) -> List[Dict]:
    path = os.path.join(results_dir, f"condition_{condition}",
                        f"condition_{condition}_results.jsonl")
    if not os.path.exists(path):
        logger.warning(f"見つかりません: {path}")
        return []
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


# ──────────────────────────────────────────────
# 可視化ユーティリティ
# ──────────────────────────────────────────────
PALETTE = {
    "correct":   "#2ecc71",
    "incorrect": "#e74c3c",
    "cond_B":    "#3498db",
    "cond_C":    "#e67e22",
}

LEVEL_COLORS = {
    "Level 1": "#1abc9c",
    "Level 2": "#3498db",
    "Level 3": "#9b59b6",
    "Level 4": "#e67e22",
    "Level 5": "#e74c3c",
}

TYPE_MARKERS = {
    "Algebra": "o",
    "Counting & Probability": "s",
    "Geometry": "^",
    "Intermediate Algebra": "D",
    "Number Theory": "P",
    "Prealgebra": "X",
    "Precalculus": "*",
}


def reduce_dim(vectors: np.ndarray, method: str = "pca") -> np.ndarray:
    """[N, H] → [N, 2] に次元削減"""
    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        return reducer.fit_transform(vectors)
    elif method == "tsne":
        n = len(vectors)
        perplexity = min(30, max(5, n // 3))
        reducer = TSNE(n_components=2, perplexity=perplexity,
                       random_state=42, max_iter=1000)
        return reducer.fit_transform(vectors)
    else:
        raise ValueError(f"Unknown method: {method}")


def save_fig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"保存: {path}")


# ──────────────────────────────────────────────
# 可視化 1: 正解 vs 不正解
# ──────────────────────────────────────────────
def plot_correct_vs_incorrect(
    vectors: np.ndarray, labels: List[bool],
    method: str, condition: str, output_dir: str
):
    coords = reduce_dim(vectors, method)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"Hidden States: 正解 vs 不正解 [条件{condition}] ({method.upper()})",
                 fontsize=13)

    for is_correct, color, label in [
        (True,  PALETTE["correct"],   "正解"),
        (False, PALETTE["incorrect"], "不正解"),
    ]:
        mask = np.array(labels) == is_correct
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, label=label, alpha=0.7, s=60, edgecolors="white", linewidths=0.5)

    ax.legend(fontsize=11)
    ax.set_xlabel(f"{method.upper()} dim 1")
    ax.set_ylabel(f"{method.upper()} dim 2")
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, f"01_correct_vs_incorrect_{condition}_{method}.png")
    save_fig(fig, path)


# ──────────────────────────────────────────────
# 可視化 2: 条件B vs 条件C（同一問題）
# ──────────────────────────────────────────────
def plot_B_vs_C(
    vectors_B: np.ndarray, vectors_C: np.ndarray,
    labels_B: List[bool], labels_C: List[bool],
    method: str, output_dir: str
):
    # 全ベクトルを合わせて次元削減（共通空間に投影）
    all_vectors = np.vstack([vectors_B, vectors_C])
    coords = reduce_dim(all_vectors, method)
    n_B = len(vectors_B)
    coords_B = coords[:n_B]
    coords_C = coords[n_B:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Hidden States: 条件B vs 条件C ({method.upper()})", fontsize=13)

    for ax, coords_cond, labels, cond, color in [
        (axes[0], coords_B, labels_B, "B", PALETTE["cond_B"]),
        (axes[1], coords_C, labels_C, "C", PALETTE["cond_C"]),
    ]:
        ax.set_title(f"条件{cond}（未学習{'あり' if cond=='B' else '→学習済み'}+hidden states）")
        mask_c = np.array(labels) == True
        mask_w = ~mask_c
        ax.scatter(coords_cond[mask_c, 0], coords_cond[mask_c, 1],
                   c=PALETTE["correct"], label="正解", alpha=0.8, s=70,
                   edgecolors="white", linewidths=0.5)
        ax.scatter(coords_cond[mask_w, 0], coords_cond[mask_w, 1],
                   c=PALETTE["incorrect"], label="不正解", alpha=0.5, s=50,
                   edgecolors="white", linewidths=0.5)
        ax.legend(fontsize=10)
        ax.set_xlabel(f"{method.upper()} dim 1")
        ax.set_ylabel(f"{method.upper()} dim 2")
        ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, f"02_B_vs_C_{method}.png")
    save_fig(fig, path)


# ──────────────────────────────────────────────
# 可視化 3: 難易度別分布
# ──────────────────────────────────────────────
def plot_by_level(
    vectors: np.ndarray, levels: List[str],
    method: str, condition: str, output_dir: str
):
    coords = reduce_dim(vectors, method)
    unique_levels = sorted(set(levels))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"Hidden States: 難易度別分布 [条件{condition}] ({method.upper()})", fontsize=13)

    for level in unique_levels:
        mask = np.array(levels) == level
        color = LEVEL_COLORS.get(level, "#95a5a6")
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, label=level, alpha=0.75, s=60,
                   edgecolors="white", linewidths=0.5)

    ax.legend(fontsize=10, title="難易度")
    ax.set_xlabel(f"{method.upper()} dim 1")
    ax.set_ylabel(f"{method.upper()} dim 2")
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, f"03_by_level_{condition}_{method}.png")
    save_fig(fig, path)


# ──────────────────────────────────────────────
# 可視化 4: 科目別分布
# ──────────────────────────────────────────────
def plot_by_type(
    vectors: np.ndarray, types: List[str],
    method: str, condition: str, output_dir: str
):
    coords = reduce_dim(vectors, method)
    unique_types = sorted(set(types))
    cmap = plt.cm.get_cmap("tab10", len(unique_types))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(f"Hidden States: 科目別分布 [条件{condition}] ({method.upper()})", fontsize=13)

    for i, t in enumerate(unique_types):
        mask = np.array(types) == t
        marker = TYPE_MARKERS.get(t, "o")
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(i)], label=t, marker=marker,
                   alpha=0.75, s=70, edgecolors="white", linewidths=0.5)

    ax.legend(fontsize=9, title="科目", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xlabel(f"{method.upper()} dim 1")
    ax.set_ylabel(f"{method.upper()} dim 2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, f"04_by_type_{condition}_{method}.png")
    save_fig(fig, path)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Hidden States 可視化スクリプト",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_dir", type=str, default="./results_inference_350",
                        help="inference_math.py の output_dir")
    parser.add_argument("--sender_model_path", type=str,
                        default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./visualizations")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="可視化に使う問題数（多いほど精度が上がるが時間がかかる）")
    parser.add_argument("--max_new_tokens", type=int, default=300,
                        help="Sender の生成トークン数（推論時と揃える）")
    parser.add_argument("--method", type=str, choices=["pca", "tsne", "both"],
                        default="both", help="次元削減手法")
    parser.add_argument("--conditions", type=str, nargs="+",
                        default=["B", "C"], choices=["A", "B", "C"])
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def main():
    args = parse_args()

    # デバイス・dtype
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.torch_dtype]

    logger.info(f"Device: {device}, dtype: {dtype}")

    methods = ["pca", "tsne"] if args.method == "both" else [args.method]
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Sender のロード ──────────────────────────────────────────────
    sender_model, sender_tokenizer = load_sender(
        args.sender_model_path, device, dtype
    )

    # ── 条件ごとにデータ収集 ─────────────────────────────────────────
    # key: condition, value: {"vectors": np.ndarray [N,H], "labels": [...], ...}
    condition_data: Dict[str, Dict] = {}

    for condition in args.conditions:
        records = load_results(args.results_dir, condition)
        if not records:
            continue

        # num_samples に絞る
        records = records[:args.num_samples]
        logger.info(f"条件{condition}: {len(records)}問の hidden states を収集")

        vectors = []
        correct_labels = []
        task_types = []
        task_levels = []

        for r in tqdm(records, desc=f"Condition {condition} hidden states"):
            try:
                hs = get_hidden_states(
                    sender_model, sender_tokenizer,
                    r["question"], device,
                    max_new_tokens=args.max_new_tokens,
                )
                vec = mean_pool(hs)  # [H]
                vectors.append(vec)
                correct_labels.append(r["is_correct"])
                task_types.append(r.get("task_type", "unknown"))
                task_levels.append(r.get("task_level", "unknown"))
            except Exception as e:
                logger.error(f"Error on {r['question_id']}: {e}")

        condition_data[condition] = {
            "vectors": np.array(vectors),     # [N, H]
            "labels": correct_labels,
            "types": task_types,
            "levels": task_levels,
        }
        logger.info(f"条件{condition}: {len(vectors)}件収集完了")

    # ── 可視化 ───────────────────────────────────────────────────────
    for method in methods:
        logger.info(f"=== 次元削減: {method.upper()} ===")

        # 可視化1・3・4: 条件ごと
        for cond, data in condition_data.items():
            if len(data["vectors"]) < 3:
                logger.warning(f"条件{cond}: サンプル数が少なすぎるためスキップ")
                continue

            plot_correct_vs_incorrect(
                data["vectors"], data["labels"],
                method, cond, args.output_dir
            )
            plot_by_level(
                data["vectors"], data["levels"],
                method, cond, args.output_dir
            )
            plot_by_type(
                data["vectors"], data["types"],
                method, cond, args.output_dir
            )

        # 可視化2: 条件B vs 条件C（両方あるとき）
        if "B" in condition_data and "C" in condition_data:
            n = min(len(condition_data["B"]["vectors"]),
                    len(condition_data["C"]["vectors"]))
            plot_B_vs_C(
                condition_data["B"]["vectors"][:n],
                condition_data["C"]["vectors"][:n],
                condition_data["B"]["labels"][:n],
                condition_data["C"]["labels"][:n],
                method, args.output_dir,
            )

    logger.info(f"すべての可視化を {args.output_dir} に保存しました")


if __name__ == "__main__":
    main()