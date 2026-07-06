#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MATH Inference Script - Condition B & C
========================================

条件B: 未学習モデル (Qwen2.5-0.5B-Instruct) + hidden states prefix
条件C: 学習済みモデル (trained_model_full)    + hidden states prefix

推論フロー:
  1. Sender (Qwen2.5-0.5B-Instruct) がテスト問題を解き、hidden statesを取得
  2. hidden statesをprocessしてprefixとしてembeddingsに挿入
  3. Studentモデル（条件B: 未学習 / 条件C: 学習済み）で generate()
  4. \boxed{} から回答を抽出して正解と比較
"""

import os
import re
import json
import math
import random
import argparse
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets
from tqdm.auto import tqdm

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("inference_math.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# AdaptiveProjection / ModelWithInsertedHiddenState
# （custom_model.py から必要部分を転記）
# ──────────────────────────────────────────────
class AdaptiveProjection(nn.Module):
    """Adaptive numerical range projection layer (custom_model.py と同一)"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2))
        self.output_scale = nn.Parameter(torch.tensor(0.1))
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.proj[0].weight, mean=0, std=0.02)
        nn.init.zeros_(self.proj[0].bias)
        nn.init.xavier_uniform_(self.proj[3].weight, gain=1e-2)
        nn.init.zeros_(self.proj[3].bias)

    def forward(self, x):
        residual = x * self.scale
        x = self.proj(residual)
        return (residual + x) * self.output_scale


class HiddenStateProcessor(nn.Module):
    """
    custom_model.py の ModelWithInsertedHiddenState から
    推論に必要な部分（process_hidden_states）だけを切り出したクラス。

    学習済みの MHA / LayerNorm / AdaptiveProjection の重みを
    hidden_mha_state.pt からロードして使う。
    """

    def __init__(self, hidden_size: int, num_heads: int = 8,
                 prepended_input_dim: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size

        # input projector（Senderの hidden_size が異なる場合に対応）
        if prepended_input_dim is not None and prepended_input_dim != hidden_size:
            self.input_projector = nn.Linear(prepended_input_dim, hidden_size, bias=True)
        else:
            self.input_projector = None

        self.hidden_mha = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.pre_ln = nn.LayerNorm(hidden_size, eps=1e-6)
        self.post_ln = nn.LayerNorm(hidden_size, eps=1e-6)
        self.adaptive_proj = AdaptiveProjection(hidden_size)

    def load_from_checkpoint(self, mha_state_path: str, device: torch.device):
        """hidden_mha_state.pt から重みをロード"""
        state = torch.load(mha_state_path, map_location=device)
        self.hidden_mha.load_state_dict(state["hidden_mha"])
        self.pre_ln.load_state_dict(state["pre_ln"])
        self.post_ln.load_state_dict(state["post_ln"])
        self.adaptive_proj.load_state_dict(state["adaptive_proj"])
        logger.info(f"Loaded MHA weights from {mha_state_path}")

    def process_hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        """custom_model.py の process_hidden_states と同一ロジック"""
        dev = self.pre_ln.weight.device
        dtyp = self.pre_ln.weight.dtype
        x = x.to(device=dev, dtype=dtyp, non_blocking=True).contiguous()

        if self.input_projector is not None:
            x = self.input_projector(x)

        normed = self.pre_ln(x).contiguous()

        # MHA の weight dtype を取得
        w_dtype = None
        ipw = getattr(self.hidden_mha, "in_proj_weight", None)
        if isinstance(ipw, torch.Tensor):
            w_dtype = ipw.dtype
        else:
            opw = getattr(getattr(self.hidden_mha, "out_proj", None), "weight", None)
            if isinstance(opw, torch.Tensor):
                w_dtype = opw.dtype
        if w_dtype is None:
            w_dtype = dtyp

        with torch.cuda.amp.autocast(enabled=False):
            q = normed.to(dtype=w_dtype).contiguous()
            k = normed.to(dtype=w_dtype).contiguous()
            v = normed.to(dtype=w_dtype).contiguous()
            attn_out, _ = self.hidden_mha(q, k, v, need_weights=False)

        attn_out = attn_out.to(dtyp)
        out = self.post_ln(normed + attn_out)
        projected = self.adaptive_proj(out)
        return projected

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=torch.cuda.is_available()):
            processed = self.process_hidden_states(x)
            return torch.clamp(processed, -10.0, 10.0)


# ──────────────────────────────────────────────
# Answer Extraction（math_evaluator.py から転記）
# ──────────────────────────────────────────────
class AnswerExtractor:

    @staticmethod
    def extract_boxed_answer(text: str) -> Optional[str]:
        def extract_balanced_braces(text, start_pos):
            if start_pos >= len(text) or text[start_pos] != "{":
                return None
            brace_count = 1
            pos = start_pos + 1
            while pos < len(text) and brace_count > 0:
                if text[pos] == "{":
                    brace_count += 1
                elif text[pos] == "}":
                    brace_count -= 1
                pos += 1
            if brace_count == 0:
                return text[start_pos + 1 : pos - 1]
            return None

        patterns = [r"\\boxed", r"\\fbox"]
        matches = []
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                start = match.end()
                if start < len(text) and text[start] == "{":
                    content = extract_balanced_braces(text, start)
                    if content is not None:
                        matches.append((match.start(), content))
        return matches[-1][1].strip() if matches else None

    @staticmethod
    def normalize(ans: Optional[str]) -> Optional[str]:
        if ans is None:
            return None
        a = ans.strip()
        a = re.sub(r"\\\\,", "", a)
        a = re.sub(r"\s+", "", a)
        return a

    @classmethod
    def evaluate(cls, model_output: str, ground_truth: str) -> Tuple[bool, Optional[str], Optional[str]]:
        pred = cls.normalize(cls.extract_boxed_answer(model_output))
        gold_extracted = cls.extract_boxed_answer(ground_truth)
        gold = cls.normalize(gold_extracted if gold_extracted is not None else ground_truth)
        is_correct = pred is not None and pred == gold
        return is_correct, pred, gold


# ──────────────────────────────────────────────
# Sender: hidden states を収集する
# ──────────────────────────────────────────────
class SenderModel:
    """
    math_collection.py の agent_generate と同じロジックで
    テスト問題の hidden states を取得する。
    """

    PROMPT_TEMPLATE = (
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

    def __init__(self, model_path: str, device: torch.device,
                 torch_dtype: torch.dtype = torch.float32,
                 max_new_tokens: int = 1500,
                 max_hidden_states: int = 10000,
                 temperature: float = 0.8,
                 top_p: float = 0.9,
                 top_k: int = 50):
        logger.info(f"Loading Sender model: {model_path}")
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_hidden_states = max_hidden_states
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map={"": device},
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Senderの hidden_size を保持
        self.hidden_size = self.model.config.hidden_size
        logger.info(f"Sender hidden_size: {self.hidden_size}")

    def _build_prompt(self, question: str) -> str:
        """math_collection.py の build_plan_prompt と同一"""
        return self.PROMPT_TEMPLATE.format(question=question)

    @torch.no_grad()
    def get_hidden_states(self, question: str) -> torch.Tensor:
        """
        math_collection.py の agent_generate と同一ロジック。
        返り値: [T, H]  (T = 収集したステップ数, H = hidden_size)
        """
        prompt = "<|im_start|>user\n" + self._build_prompt(question) + "<|im_end|>\n<|im_start|>assistant\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=1,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=1.0,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

        # math_collection.py の agent_generate と同一の hidden states 収集処理
        steps = outputs.hidden_states
        start_index = max(0, len(steps) - self.max_hidden_states)
        step_hiddens = []
        for i in range(start_index, len(steps)):
            last_layer = steps[i][-1]   # 最終層
            h_last = last_layer[:, -1, :]  # 最後のトークン位置
            step_hiddens.append(h_last)

        hidden_seq = torch.stack(step_hiddens, dim=1)  # [1, T, H]
        if hidden_seq.size(0) == 1:
            hidden_seq = hidden_seq.squeeze(0)          # [T, H]

        return hidden_seq


# ──────────────────────────────────────────────
# Student: hidden states prefix で generate する
# ──────────────────────────────────────────────
class StudentModel:
    """
    条件B（未学習）または条件C（学習済み）の Student モデル。

    学習時（custom_model.py）の構造を推論でも再現する：
        <|im_start|>user
        {問題文}
        <|im_end|>          ← ここに hidden states を挿入（human_end_positions 相当）
        <|im_start|>assistant
        {生成開始}

    hidden states は <|im_end|> の直後、<|im_start|>assistant の直前に挿入する。
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        torch_dtype: torch.dtype,
        hidden_state_processor: HiddenStateProcessor,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        do_sample: bool = True,
    ):
        logger.info(f"Loading Student model: {model_path}")
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.processor = hidden_state_processor

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map={"": device},
            trust_remote_code=True,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.hidden_size = self.model.config.hidden_size
        logger.info(f"Student hidden_size: {self.hidden_size}")

    def _build_inputs_embeds_with_hidden(
        self, question: str, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        学習時（custom_model.py）と同じ挿入位置を再現する。

        トークン列のイメージ:
            [<|im_start|>user\n {問題文} <|im_end|>]   ← before_ids
            [hidden_states (processed)]                  ← prefix (embedding空間)
            [\n<|im_start|>assistant\n]                ← after_ids（生成の起点）

        返り値:
            inputs_embeds : [1, before_len + T + after_len, H]
            attention_mask: [1, before_len + T + after_len]
        """
        model_dtype = self.model.get_input_embeddings().weight.dtype
        emb = self.model.get_input_embeddings()

        # ── ① <bop> までの "before" 部分（オリジナル hidden_agent.py と同じ位置に <bop> を含める）
        before_text = (
            "<|im_start|>user\n"
            "Solve the following math problem step by step. "
            "Show your work clearly and put your final answer in \\boxed{}."
            "\n\nProblem: " + question
            + "\nNow, you are given a step-by-step plan to complete this task as follow: <bop>"
        )
        before_tok = self.tokenizer(
            before_text, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        before_embeds = emb(before_tok["input_ids"]).to(model_dtype)  # [1, L_b, H]

        # ── ② hidden states を process ──────────────────────────────────
        hidden_processed = self.processor(
            hidden_states.unsqueeze(0)
        ).squeeze(0).to(model_dtype)                                   # [T, H]
        prefix_embeds = hidden_processed.unsqueeze(0)                  # [1, T, H]

        # ── ③ <eop> + <|im_end|> + assistant ヘッダ（オリジナルと同じ分割・順序）────
        eop_tok = self.tokenizer(
            "<eop>", return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        eop_embeds = emb(eop_tok["input_ids"]).to(model_dtype)

        end_tok = self.tokenizer(
            "<|im_end|>\n", return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        end_embeds = emb(end_tok["input_ids"]).to(model_dtype)

        after_text = "<|im_start|>assistant\n"
        after_tok = self.tokenizer(
            after_text, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        after_embeds = emb(after_tok["input_ids"]).to(model_dtype)    # [1, L_a, H]

        # ── ④ 結合 ────────────────────────────────────────────────────────
        inputs_embeds = torch.cat(
            [before_embeds, prefix_embeds, eop_embeds, end_embeds, after_embeds], dim=1
        )  # [1, L_b + T + L_eop + L_end + L_a, H]

        total_len = inputs_embeds.size(1)
        attention_mask = torch.ones(
            1, total_len, dtype=torch.long, device=self.device
        )

        return inputs_embeds, attention_mask

    @torch.no_grad()
    def generate_with_hidden_states(
        self, question: str, hidden_states: torch.Tensor
    ) -> str:
        """
        hidden_states を <|im_end|> 直後に挿入し、base_model.generate() を呼ぶ。
        返り値: 生成テキスト
        """
        inputs_embeds, attention_mask = self._build_inputs_embeds_with_hidden(
            question, hidden_states
        )

        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()



# ──────────────────────────────────────────────
# BaselineModel: 条件A（hidden statesなし・通常生成）
# ──────────────────────────────────────────────
class BaselineModel:
    """
    条件A: 未学習モデル（Qwen2.5-0.5B-Instruct）を
    hidden states なしで通常の chat フォーマットで generate する。

    プロンプト構造（条件B・Cと揃えた Qwen chat 形式）:
        <|im_start|>user
        Solve... Problem: {問題文}
        <|im_end|>
        <|im_start|>assistant
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        torch_dtype: torch.dtype,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        do_sample: bool = True,
    ):
        logger.info(f"Loading Baseline model: {model_path}")
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map={"": device},
            trust_remote_code=True,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Baseline hidden_size: {self.model.config.hidden_size}")

    def _build_prompt(self, question: str) -> str:
        return (
            "<|im_start|>user\n"
            "Solve the following math problem step by step. "
            "Show your work clearly and put your final answer in \\boxed{}."
            "\n\nProblem: " + question + "\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @torch.no_grad()
    def generate(self, question: str) -> str:
        """hidden states なしで通常の input_ids から generate する"""
        prompt = self._build_prompt(question)
        tok = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.device)

        outputs = self.model.generate(
            input_ids=tok["input_ids"],
            attention_mask=tok["attention_mask"],
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # 入力部分を除いた生成トークンのみデコード
        generated_ids = outputs[0][tok["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()


def run_baseline(
    baseline: BaselineModel,
    dataset,
    output_dir: str,
    condition: str = "A",
) -> Dict[str, Any]:
    """条件A・D（hidden statesなし）の推論を実行する。
    condition="A": 未学習モデル / condition="D": 学習済みモデル"""
    os.makedirs(output_dir, exist_ok=True)
    extractor = AnswerExtractor()

    results = []
    correct = 0
    total = 0

    pbar = tqdm(total=len(dataset), desc=f"Condition {condition}")

    for idx, item in enumerate(dataset):
        question   = item["problem"]
        solution   = item["solution"]
        task_type  = item.get("type", "unknown")
        task_level = item.get("level", "unknown")
        question_id = f"MATH_test_{idx}"

        try:
            model_output = baseline.generate(question)
            is_correct, pred, norm_gt = extractor.evaluate(model_output, solution)

            result = InferenceResult(
                condition=condition,
                question_id=question_id,
                question=question,
                ground_truth=solution,
                model_output=model_output,
                predicted_answer=pred,
                normalized_gt=norm_gt,
                is_correct=is_correct,
                task_type=task_type,
                task_level=task_level,
                timestamp=datetime.now().isoformat(),
            )
            results.append(result)
            total += 1
            if is_correct:
                correct += 1

        except Exception as e:
            logger.error(f"[Condition {condition}] Error on {question_id}: {e}")
            total += 1

        pbar.update(1)
        pbar.set_postfix({
            "Acc": f"{correct/total:.3f}" if total > 0 else "-",
            "Correct": correct,
            "Total": total,
        })

    pbar.close()

    accuracy = correct / total if total > 0 else 0.0
    summary = {
        "condition": condition,
        "total_questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "timestamp": datetime.now().isoformat(),
    }

    results_path = os.path.join(output_dir, f"condition_{condition}_results.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    summary_path = os.path.join(output_dir, f"condition_{condition}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(
        f"[Condition {condition}] Done: {correct}/{total} = {accuracy:.3f} → {output_dir}"
    )
    return summary


# ──────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────
@dataclass
class InferenceResult:
    condition: str          # "B" or "C"
    question_id: str
    question: str
    ground_truth: str
    model_output: str
    predicted_answer: Optional[str]
    normalized_gt: Optional[str]
    is_correct: bool
    task_type: str
    task_level: str
    timestamp: str


def load_math_test_dataset(subjects: List[str], num_samples: Optional[int]) -> List[Dict]:
    """MATHテストデータセットをロード（math_collection.py と同一の subjects/split）"""
    logger.info(f"Loading MATH test dataset, subjects: {subjects}")
    dataset = concatenate_datasets([
        load_dataset("EleutherAI/hendrycks_math", config, split="test")
        for config in subjects
    ])
    if num_samples is not None and num_samples > 0:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    logger.info(f"Loaded {len(dataset)} test problems")
    return dataset


def run_inference(
    condition: str,
    sender: SenderModel,
    student: StudentModel,
    dataset,
    output_dir: str,
) -> Dict[str, Any]:
    """
    条件B または 条件C の推論を実行する。

    Args:
        condition: "B"（未学習）または "C"（学習済み）
        sender:    Sender モデル（hidden states 生成）
        student:   Student モデル（回答生成）
        dataset:   MATHテストデータセット
        output_dir: 結果保存先
    """
    os.makedirs(output_dir, exist_ok=True)
    extractor = AnswerExtractor()

    results = []
    correct = 0
    total = 0

    pbar = tqdm(total=len(dataset), desc=f"Condition {condition}")

    for idx, item in enumerate(dataset):
        question  = item["problem"]
        solution  = item["solution"]
        task_type = item.get("type", "unknown")
        task_level = item.get("level", "unknown")
        question_id = f"MATH_test_{idx}"

        try:
            # Step 1: Sender が hidden states を取得
            hidden_states = sender.get_hidden_states(question)  # [T, H]

            # Step 2: Student が hidden states prefix で回答を生成
            model_output = student.generate_with_hidden_states(question, hidden_states)

            # Step 3: 回答を評価
            is_correct, pred, norm_gt = extractor.evaluate(model_output, solution)

            result = InferenceResult(
                condition=condition,
                question_id=question_id,
                question=question,
                ground_truth=solution,
                model_output=model_output,
                predicted_answer=pred,
                normalized_gt=norm_gt,
                is_correct=is_correct,
                task_type=task_type,
                task_level=task_level,
                timestamp=datetime.now().isoformat(),
            )
            results.append(result)
            total += 1
            if is_correct:
                correct += 1

        except Exception as e:
            logger.error(f"[Condition {condition}] Error on {question_id}: {e}")
            total += 1

        pbar.update(1)
        pbar.set_postfix({
            "Acc": f"{correct/total:.3f}" if total > 0 else "-",
            "Correct": correct,
            "Total": total,
        })

    pbar.close()

    accuracy = correct / total if total > 0 else 0.0
    summary = {
        "condition": condition,
        "total_questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "timestamp": datetime.now().isoformat(),
    }

    # 保存
    results_path = os.path.join(output_dir, f"condition_{condition}_results.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    summary_path = os.path.join(output_dir, f"condition_{condition}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(
        f"[Condition {condition}] Done: {correct}/{total} = {accuracy:.3f} "
        f"→ {output_dir}"
    )
    return summary


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="MATH Inference Script - Condition A, B & C",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 条件選択
    parser.add_argument(
        "--condition", type=str,
        choices=["A", "B", "C", "D", "AB", "BC", "AC", "AD", "CD", "all"], default="all",
        help=(
            "実行する条件: "
            "A（未学習+hidden statesなし）/ "
            "B（未学習+hidden states）/ "
            "C（学習済み+hidden states）/ "
            "D（学習済み+hidden statesなし）/ "
            "AB / BC / AC / AD / CD / all（全条件）"
        ),
    )

    # モデルパス
    parser.add_argument(
        "--sender_model_path", type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Senderモデルのパス（hidden states生成用。学習時と同じモデルを指定）",
    )
    parser.add_argument(
        "--base_model_path", type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="条件B用: 未学習ベースモデルのパス",
    )
    parser.add_argument(
        "--trained_model_path", type=str,
        default="./data/experiment_math_20260622_213422/trained_model_full",
        help="条件C用: 学習済みモデルのパス",
    )
    parser.add_argument(
        "--mha_state_path", type=str,
        default="./data/experiment_math_20260622_213422/trained_model_full/hidden_mha_state.pt",
        help="custom_model.py の save_pretrained() が出力した hidden_mha_state.pt のパス",
    )

    # データセット
    parser.add_argument(
        "--subjects", type=str, nargs="+",
        default=["algebra", "counting_and_probability", "geometry",
                 "intermediate_algebra", "number_theory", "prealgebra", "precalculus"],
        help="評価するMATHサブジェクト",
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="評価問題数（デフォルト: 全問）。動作確認には 10 など小さい値を推奨",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="乱数シード（Sender/Studentの生成を再現可能にする）",
    )

    # Sender 生成パラメータ（math_collection.py のデフォルトに合わせた）
    parser.add_argument("--sender_max_new_tokens", type=int, default=1500)
    parser.add_argument("--sender_max_hidden_states", type=int, default=10000)
    parser.add_argument("--sender_temperature", type=float, default=0.8)
    parser.add_argument("--sender_top_p", type=float, default=0.9)
    parser.add_argument("--sender_top_k", type=int, default=50)

    # Student 生成パラメータ（math_evaluator.py のデフォルトに合わせた）
    parser.add_argument("--student_max_new_tokens", type=int, default=2048)
    parser.add_argument("--student_temperature", type=float, default=0.1)
    parser.add_argument(
        "--student_do_sample", action="store_true", default=True,
        help="Studentでsampling生成を使う（デフォルト: True）",
    )
    parser.add_argument(
        "--student_no_sample", dest="student_do_sample", action="store_false",
        help="Studentでgreedy decoding を使う",
    )

    # MHA の heads 数（prepended_config.json があれば自動読み込み）
    parser.add_argument("--num_heads", type=int, default=8,
                        help="HiddenStateProcessorのMHAヘッド数")

    # デバイス・dtype
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument(
        "--torch_dtype", type=str, default="float32",
        choices=["float32", "float16", "bfloat16"],
    )

    # 出力
    parser.add_argument(
        "--output_dir", type=str, default="./results_inference",
        help="結果の保存先ディレクトリ",
    )

    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def resolve_dtype(dtype_str: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_str]


def load_prepended_config(trained_model_path: str) -> Optional[Dict]:
    """trained_model_full/prepended_config.json を読み込む"""
    config_path = os.path.join(trained_model_path, "prepended_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        logger.info(f"Loaded prepended_config: {cfg}")
        return cfg
    return None


def set_seed(seed: int) -> None:
    """Sender/Studentのdo_sample生成を再現可能にするためのシード固定"""
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.torch_dtype)

    set_seed(args.seed)
    logger.info(f"Seed固定: {args.seed}")
    logger.info(f"Device: {device}, dtype: {dtype}")
    logger.info(f"Condition: {args.condition}")

    # ── prepended_config.json から num_heads を自動取得 ──────────────
    prepended_cfg = load_prepended_config(args.trained_model_path)
    num_heads = args.num_heads
    if prepended_cfg is not None and "mha_num_heads" in prepended_cfg:
        num_heads = int(prepended_cfg["mha_num_heads"])
        logger.info(f"Using num_heads={num_heads} from prepended_config.json")

    # ── Sender（条件B・Cのみ必要。A・Dはhidden statesを使わないので不要）───
    needs_sender = args.condition not in ("A", "D", "AD")
    sender_hidden_size = 0
    if needs_sender:
        sender = SenderModel(
            model_path=args.sender_model_path,
            device=device,
            torch_dtype=dtype,
            max_new_tokens=args.sender_max_new_tokens,
            max_hidden_states=args.sender_max_hidden_states,
            temperature=args.sender_temperature,
            top_p=args.sender_top_p,
            top_k=args.sender_top_k,
        )
        sender_hidden_size = sender.hidden_size

    # ── HiddenStateProcessor（条件B・Cのみ必要）────────────────────────
    needs_hidden = args.condition not in ("A", "D", "AD")
    processor = None

    if needs_hidden:
        if prepended_cfg is None or "hidden_size" not in prepended_cfg:
            raise FileNotFoundError(
                f"prepended_config.json が見つからないか hidden_size キーがありません:\n"
                f"  {args.trained_model_path}/prepended_config.json\n"
                "trained_model_path が正しいか確認してください。"
            )
        student_hidden_size = int(prepended_cfg["hidden_size"])
        logger.info(f"Student hidden_size={student_hidden_size} (from prepended_config.json)")

        prepended_input_dim = sender_hidden_size if sender_hidden_size != student_hidden_size else None

        processor = HiddenStateProcessor(
            hidden_size=student_hidden_size,
            num_heads=num_heads,
            prepended_input_dim=prepended_input_dim,
        ).to(device)

        if os.path.exists(args.mha_state_path):
            processor.load_from_checkpoint(args.mha_state_path, device)
        else:
            logger.warning(
                f"hidden_mha_state.pt not found at {args.mha_state_path}. "
                "Using randomly initialized processor weights."
            )
        processor.eval()

    # ── データセット ─────────────────────────────────────────────────
    dataset = load_math_test_dataset(args.subjects, args.num_samples)

    summaries = {}

    # ── 条件A: 未学習 + hidden states なし（ベースライン）──────────────
    if args.condition in ("A", "AB", "AC", "AD", "all"):
        logger.info("=" * 60)
        logger.info("Condition A: Base (untrained) model, no hidden states [baseline]")
        logger.info("=" * 60)

        baseline_A = BaselineModel(
            model_path=args.base_model_path,
            device=device,
            torch_dtype=dtype,
            max_new_tokens=args.student_max_new_tokens,
            temperature=args.student_temperature,
            do_sample=args.student_do_sample,
        )

        summary_A = run_baseline(
            baseline=baseline_A,
            dataset=dataset,
            output_dir=os.path.join(args.output_dir, "condition_A"),
        )
        summaries["A"] = summary_A

        del baseline_A
        torch.cuda.empty_cache()

    # ── 条件B: 未学習 + hidden states ────────────────────────────────
    if args.condition in ("B", "AB", "BC", "all"):
        logger.info("=" * 60)
        logger.info("Condition B: Base (untrained) model + hidden states")
        logger.info("=" * 60)

        student_B = StudentModel(
            model_path=args.base_model_path,
            device=device,
            torch_dtype=dtype,
            hidden_state_processor=processor,
            max_new_tokens=args.student_max_new_tokens,
            temperature=args.student_temperature,
            do_sample=args.student_do_sample,
        )

        summary_B = run_inference(
            condition="B",
            sender=sender,
            student=student_B,
            dataset=dataset,
            output_dir=os.path.join(args.output_dir, "condition_B"),
        )
        summaries["B"] = summary_B

        del student_B
        torch.cuda.empty_cache()

    # ── 条件C: 学習済み + hidden states ──────────────────────────────
    if args.condition in ("C", "BC", "AC", "CD", "all"):
        logger.info("=" * 60)
        logger.info("Condition C: Trained model + hidden states")
        logger.info("=" * 60)

        student_C = StudentModel(
            model_path=args.trained_model_path,
            device=device,
            torch_dtype=dtype,
            hidden_state_processor=processor,
            max_new_tokens=args.student_max_new_tokens,
            temperature=args.student_temperature,
            do_sample=args.student_do_sample,
        )

        summary_C = run_inference(
            condition="C",
            sender=sender,
            student=student_C,
            dataset=dataset,
            output_dir=os.path.join(args.output_dir, "condition_C"),
        )
        summaries["C"] = summary_C

        del student_C
        torch.cuda.empty_cache()

    # ── 条件D: 学習済み + hidden statesなし ──────────────────────────
    if args.condition in ("D", "AD", "CD", "all"):
        logger.info("=" * 60)
        logger.info("Condition D: Trained model, no hidden states")
        logger.info("=" * 60)

        baseline_D = BaselineModel(
            model_path=args.trained_model_path,
            device=device,
            torch_dtype=dtype,
            max_new_tokens=args.student_max_new_tokens,
            temperature=args.student_temperature,
            do_sample=args.student_do_sample,
        )

        summary_D = run_baseline(
            baseline=baseline_D,
            dataset=dataset,
            output_dir=os.path.join(args.output_dir, "condition_D"),
            condition="D",
        )
        summaries["D"] = summary_D

        del baseline_D
        torch.cuda.empty_cache()

    # ── 最終サマリー ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Final Results")
    logger.info("=" * 60)
    for cond, s in summaries.items():
        logger.info(
            f"Condition {cond}: {s['correct']}/{s['total_questions']} "
            f"= {s['accuracy']:.3f}"
        )

    combined_path = os.path.join(args.output_dir, "combined_summary.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    logger.info(f"Combined summary saved to {combined_path}")


if __name__ == "__main__":
    main()