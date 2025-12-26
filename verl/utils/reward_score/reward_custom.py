# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from mathruler.grader import extract_boxed_content  


def format_reward(solution_str: str) -> float:
    """
    Checks if the prediction string follows the <think>...</think><answer>...</answer> format.
    Allows optional whitespace around tags and content, and is case-insensitive.
    """
    # DOTALL makes '.' match newlines, IGNORECASE ignores case
    pattern = re.compile(r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL | re.IGNORECASE)
    # Check if solution_str is None or empty before matching
    if solution_str and re.fullmatch(pattern, solution_str.strip()):
        return 1.0
    else:
        return 0.0


def parse_prediction_format(pred_str: str):
    """
    Parses the prediction format: direction(e.g., r or n).
    Returns a dict {'direction': 'l'|'r'|'n'or None if invalid.
    Allows whitespace and ignores case for direction and option.
    """
    if not pred_str:
        return None
    match = re.match(r"^\s*([ABC])\s*$", pred_str.strip(), re.IGNORECASE)
    if match:
        direction = match.group(1)
        return {"direction": direction.upper()}
    else:
        return None


def extract_answer_content(solution_str: str) -> str:
    """
    Extracts the content inside the <answer>...</answer> tags from the solution string.
    Returns the content or empty string if not found.
    """
    if not solution_str:
        return "None"
    
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
    match = pattern.search(solution_str)
    if match:
        return match.group(1).strip()
    return "None"


def accuracy_reward_translation(solution_str: str, ground_truth: str) -> float:
    """
    Calculates accuracy score based on matching direction (l/r/n), option (A-G),
    and consistency between predicted direction and GT angle sign (checked only if options match).
    Score is normalized to 0.0-1.0.
    """
    predicted_core_answer = extract_answer_content(solution_str)
    if not predicted_core_answer:
        return 0.0
        

    parsed_gt = ground_truth
    # parsed_pred = parse_prediction_format(predicted_core_answer)
    parsed_pred = predicted_core_answer

    if not parsed_gt or not parsed_pred:
        print(f"Debug: Parsing failed. GT: '{ground_truth}' -> {parsed_gt}, Pred: '{predicted_core_answer}' -> {parsed_pred}")
        print(f"Debug: Predicted core answer: {solution_str}")
        return 0.0

    raw_score = 0.0

   
    if parsed_pred == parsed_gt:
        raw_score += 1.0
    
        # print(f"Debug: Direction match (+1.0)")
 
    normalized_score = raw_score
    # print(f"Debug: Raw Score: {raw_score}, Normalized: {normalized_score}")
    return normalized_score


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    """
    Computes the final score by combining accuracy and format scores.
    Accuracy weight 0.5, Format weight 0.5.
    """
    acc_score = accuracy_reward_translation(solution_str, ground_truth)
    fmt_score = format_reward(solution_str)
    final_score = 0.5 * acc_score + 0.5 * fmt_score
    # print(f"Debug: Acc:{acc_score:.2f}, Fmt:{fmt_score:.2f}, Final:{final_score:.2f}")
    return final_score


