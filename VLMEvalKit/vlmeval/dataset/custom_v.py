import os
import re
import pandas as pd
import os.path as osp
import json
import numpy as np
import warnings
from tqdm import tqdm
# from .image_mcq import ImageMCQDataset
from .utils import DEBUG_MESSAGE, build_judge
from ..smp import (LMUDataRoot, file_size, load, dump, decode_base64_to_image_file,
                   listinstr, gpt_key_set)
import string
import glob
from .video_base import VideoMCQDataset

class CustomDataset(VideoMCQDataset):
    """
    CustomDataset class for multiple-choice questions with multiple images.
    支持多图片的多选题评测数据集，图片以JSON数组格式存储在image字段中。
    """
    TYPE = 'MCQ'

    DATASET_URL = {
        'MMSI_Bench': 'https://huggingface.co/datasets/RunsenXu/MMSI-Bench/resolve/main/MMSI_bench.tsv',
        'ViewSpatial-Bench': 'https://huggingface.co/datasets/warmsnow/ViewSpatial-Bench-vlmeval/resolve/main/ViewSpatial-Bench.tsv'
    }
    DATASET_MD5 = {
        'MMSI_Bench': 'c473f72a345f616fa68a628580b573b6'
    }

    def __init__(self, dataset='MMSI_Bench', skip_noimg=True, extract_options=False):
        """
        初始化CustomDataset
        
        Args:
            dataset (str): 数据集名称
            skip_noimg (bool): 是否跳过没有图片的数据
            extract_options (bool): 是否从选项列（A、B、C、D）中提取并拼接prompt
                                   True: 像MCQ类一样从选项列提取并拼接
                                   False: 直接使用question字段（适合question已包含选项的数据集）
        """
        # 调用父类初始化方法
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)
        
        # 保存extract_options参数
        if dataset == 'ViewSpatial-Bench':
            extract_options = True
        self.extract_options = extract_options

    @classmethod
    def supported_datasets(cls):
        return ['MMSI_Bench','ViewSpatial-Bench']

    def dump_image(self, line):
        """
        处理图片字段，支持多张图片。
        如果image字段是JSON数组格式，则解析并处理每张图片。
        """
        # 处理image_path字段
        if 'image_path' in line and isinstance(line['image_path'], str):
            tgt_path = line['image_path']
            if not isinstance(tgt_path, list):
                tgt_path = [tgt_path]
            return tgt_path

        # 处理image字段
        if 'image' in line:
            # 获取image字段的值，确保它是一个字符串
            img_field = line['image']
            if isinstance(img_field, (pd.Series, np.ndarray)):
                # 如果是Series或数组，取第一个元素
                if len(img_field) > 0:
                    img_str = (img_field.iloc[0] if hasattr(img_field, 'iloc')
                               else img_field[0])
                else:
                    return None
            else:
                img_str = img_field

            # 处理已经是列表类型的图片数据
            if isinstance(img_str, list):
                paths = []
                # 处理每张图片
                for i, img_base64 in enumerate(img_str):
                    img_path = os.path.join(self.img_root, f"{line['index']}_{i}.jpg")
                    decode_base64_to_image_file(img_base64, img_path)
                    paths.append(img_path)
                return paths

            # 确保img_str是字符串
            if not isinstance(img_str, str):
                return None

            # 检查是否是JSON数组格式的多图片
            if img_str.startswith('[') and img_str.endswith(']'):
                try:
                    # 尝试解析JSON数组
                    img_list = json.loads(img_str)
                    paths = []

                    # 处理每张图片
                    for i, img_base64 in enumerate(img_list):
                        img_path = os.path.join(self.img_root, f"{line['index']}_{i}.jpg")
                        decode_base64_to_image_file(img_base64, img_path)
                        paths.append(img_path)

                    return paths
                except json.JSONDecodeError:
                    # 如果解析失败，按单图片处理
                    pass

            # 单图片处理
            img_path = os.path.join(self.img_root, f"{line['index']}.jpg")
            decode_base64_to_image_file(img_str, img_path)
            return img_path

        return None

    def build_prompt(self, line):
        """
        构建提示，支持多图片输入。
        支持两种模式：
        1. extract_options=True: 从选项列（A、B、C、D）中提取并拼接prompt（类似MCQ）
        2. extract_options=False: 直接使用question字段（适合question已包含选项的数据集）
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        # 处理图片（支持多图）
        tgt_path = self.dump_image(line)

        if self.extract_options:
            # 模式1：从选项列提取并拼接（类似image_mcq的方式）
            question = line['question']
            
            # 提取选项
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            
            # 构建选项提示
            options_prompt = 'Options:\n'
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'
            
            # 处理hint（如果有）
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            
            # 构建完整提示
            prompt = ''
            if hint is not None:
                prompt += f'Hint: {hint}\n'
            prompt += f'Question: {question}\n'
            if len(options):
                prompt += options_prompt
                prompt += 'Please select the correct answer from the options above. \n'
        else:
            # 模式2：直接使用question字段（当前的行为）
            question = line['question']
            # 添加post_prompt，引导模型以正确格式回答
            post_prompt = ("Answer with the option's letter from the given choices directly. "
                           "Enclose the option's letter within ``.")
            prompt = f'{question}\n{post_prompt}'

        # 构建多模态消息
        msgs = []
        if isinstance(tgt_path, list):
            # 处理多张图片
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            # 处理单张图片
            msgs = [dict(type='image', value=tgt_path)]

        # 添加文本提示
        msgs.append(dict(type='text', value=prompt))
        return msgs

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """
        评估模型预测结果。
        使用extract_single_choice_with_word_boundary函数提取预测的选项。
        """
        data = load(eval_file)

        # 确保预测值和答案都是字符串类型
        data['prediction'] = [str(x) if x is not None else None for x in data['prediction']]
        data['answer'] = [str(x) if x is not None else None for x in data['answer']]

        # 计算准确率
        correct = 0
        total = 0

        # 添加预测结果列
        data['extracted_pred'] = None
        data['score'] = 0.0

        for idx, row in data.iterrows():
            gt = row['answer']
            pred = row['prediction']

            # 使用提供的函数提取选项
            extracted_pred = cls.extract_single_choice_with_word_boundary(pred)

            # 记录提取的预测结果
            data.at[idx, 'extracted_pred'] = extracted_pred

            # 如果提取到了有效选项，进行得分计算
            if extracted_pred is not None:
                answer = gt.lower().replace("\n", " ").strip()
                predict = extracted_pred.lower().replace("\n", " ").strip()
                try:
                    if answer == predict[0]:
                        data.at[idx, 'score'] = 1.0
                        correct += 1
                    elif predict[0] == "(" and answer == predict[1]:
                        data.at[idx, 'score'] = 1.0
                        correct += 1
                    elif predict[0:7] == "option " and answer == predict[7]:
                        data.at[idx, 'score'] = 1.0
                        correct += 1
                    elif predict[0:14] == "the answer is " and answer == predict[14]:
                        data.at[idx, 'score'] = 1.0
                        correct += 1
                except Exception:
                    pass

            total += 1

        accuracy = correct / total if total > 0 else 0
        print("MMSI_Bench 评测结果：")
        print(f"总样本数: {total}")
        print(f"正确样本数: {correct}")
        print(f"准确率: {accuracy:.2%}")

        # 分类别计算准确率
        category_acc = {}
        if 'category' in data.columns:
            for category in data['category'].unique():
                cat_data = data[data['category'] == category]
                cat_correct = sum(cat_data['score'] == 1.0)
                cat_total = len(cat_data)

                category_acc[category] = cat_correct / cat_total if cat_total > 0 else 0

        results = {
            'overall': accuracy,
            'categories': category_acc
        }

        # 保存详细评测结果
        score_file = eval_file.replace('.xlsx', '_score.xlsx')
        data.to_excel(score_file)

        return pd.DataFrame([results])

    @staticmethod
    def extract_single_choice_with_word_boundary(pred):
        """
        从预测文本中提取选项，并与正确答案比较。
        返回提取到的选项，如果没有找到则返回None。
        """
        if pred is None:
            return None

        # 确保pred是字符串类型
        try:
            pred = str(pred)
        except Exception:
            return None

        pattern_1 = r'``([^`]*)``'
        match = re.search(pattern_1, pred)
        if match:
            pred = match.group(1)  # 提取反引号之间的内容

        pattern_2 = r'`([^`]*)`'
        match = re.search(pattern_2, pred)
        if match:
            pred = match.group(1)  # 提取双反引号之间的内容

        pattern_3 = r'\b[A-D]\b(?!\s[a-zA-Z])'
        match = re.search(pattern_3, pred)
        if match:
            pred = match.group()  # 提取孤立的大写字母（排除"A bike"，不定冠词+空格+单词的情况）
        else:
            return None  # 如果没有匹配，返回 None

        return pred

    @staticmethod
    def extract_single_choice_with_llm(pred, question=None, cache_dir=None):
        """
        Extract a single choice answer from a prediction using an LLM.
        Uses concurrency and caching to improve performance.

        Args:
            pred (str): The prediction text to extract a choice from
            question (str, optional): The question text containing options. Default is None.
            cache_dir (str, optional): Directory for caching results. Default is '.cache'.

        Returns:
            str: The extracted choice (A, B, C, D, or Z for no match)
        """
        import hashlib
        import os
        import json
        from concurrent.futures import ProcessPoolExecutor

        # Setup cache directory
        if cache_dir is None:
            cache_dir = os.environ.get('.cache', '.cache')

        os.makedirs(cache_dir, exist_ok=True)

        # Create cache key from question and prediction
        def get_cache_path(question, pred):
            combined = (question or "") + (pred or "")
            hash_key = hashlib.md5(combined.encode()).hexdigest()
            return os.path.join(cache_dir, f"choice_cache_{hash_key}.json")

        cache_path = get_cache_path(question, pred)

        # Check cache first
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                    return cached_data.get('choice', 'Z')
            except Exception:
                # If there's any error with the cache, proceed without it
                pass

        # Build the model using build_judge
        model = build_judge(model='chatgpt-0125')

        # Build the prompt for the LLM
        prompt = (
            'You are an AI assistant who will help me to match '
            'an answer with several options of a single-choice question. '
            'You are provided with a question and an answer, '
            'and you need to find which option is most similar to the answer. '
            'If the meaning of all options are significantly different from the answer, output Z. '
            'Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. '
            'Do not explain your reasoning, just output the letter directly.\n'
            'Example 1: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
            'Answer: a cute teddy bear\nYour output: A\n'
            'Example 2: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
            'Answer: Spider\nYour output: Z\n'
            'Example 3: \n'
            f'Question: {question}\nAnswer: {pred}\nYour output: '
        )

        retry = 3
        while retry:
            try:
                ans = model.generate(prompt)
                # Try to extract the choice from the answer
                choice = None

                # First look for a single letter answer
                match = re.search(r'\b([A-DZ])\b', ans)
                if match:
                    choice = match.group(1)

                # Also look for patterns like "(A)" or "A."
                if not choice:
                    match = re.search(r'[\(\s]([A-DZ])[\)\.\s]', ans)
                    if match:
                        choice = match.group(1)

                if choice and choice in "ABCDZ":
                    # Save to cache
                    try:
                        with open(cache_path, 'w') as f:
                            json.dump({
                                'choice': choice,
                                'full_response': ans,
                                'prompt': prompt
                            }, f)
                    except Exception:
                        # If caching fails, just continue
                        pass

                    return choice
            except Exception:
                pass

            retry -= 1

        # If all attempts failed, return Z
        # Save failure to cache too
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    'choice': 'Z',
                    'full_response': 'Failed to extract',
                    'prompt': prompt
                }, f)
        except Exception:
            pass

        return "Z"

    @staticmethod
    def _process_single_item(row_dict, cache_dir='.cache'):
        """
        处理单个项目的工作函数，用于并行处理

        Args:
            row_dict (dict): 数据行字典
            cache_dir (str): 缓存目录

        Returns:
            str: 提取的选项
        """
        import time
        import random
        import sys
        import os
        import hashlib
        import json
        import re

        try:
            pred = row_dict['prediction']
            question = row_dict['question']

            # 构建缓存路径
            def get_cache_path(question, pred):
                # 规范化输入，删除所有空白字符，确保每次生成相同的哈希值
                combined = ((question or "") + (pred or "")).strip()
                # 仅使用前1000个字符计算哈希，避免超长文本
                combined = combined[:1000]
                hash_key = hashlib.md5(combined.encode('utf-8', errors='ignore')).hexdigest()
                return os.path.join(cache_dir, f"choice_cache_{hash_key}.json")

            cache_path = get_cache_path(question, pred)

            # 检查缓存 - 添加详细日志
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                        choice = cached_data.get('choice', None)
                        if choice and choice in "ABCDZ":
                            print(f"✅ 缓存命中: {cache_path}")
                            return choice
                        else:
                            print(f"❌ 缓存格式无效: {cache_path}")
                except Exception as e:
                    print(f"❌ 读取缓存出错 {cache_path}: {e}")
            else:
                print(f"⚠️ 缓存不存在: {cache_path}")

            # 添加随机延迟以避免API速率限制
            time.sleep(random.uniform(0, 0.2))

            # 直接调用静态方法处理
            from .utils import build_judge

            # 构建模型
            model = build_judge(model='chatgpt-0125')

            # 构建提示
            prompt = (
                'You are an AI assistant who will help me to match '
                'an answer with several options of a single-choice question. '
                'You are provided with a question and an answer, '
                'and you need to find which option is most similar to the answer. '
                'If the meaning of all options are significantly different from the answer, output Z. '
                'Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. '
                'Do not explain your reasoning, just output the letter directly.\n'
                'Example 1: \n'
                'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
                'Answer: a cute teddy bear\nYour output: A\n'
                'Example 2: \n'
                'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
                'Answer: Spider\nYour output: Z\n'
                'Example 3: \n'
                f'Question: {question}\nAnswer: {pred}\nYour output: '
            )

            # 调用模型
            retry = 3
            while retry:
                try:
                    ans = model.generate(prompt)
                    # 提取选项
                    choice = None

                    # 首先查找单个字母答案
                    match = re.search(r'\b([A-DZ])\b', ans)
                    if match:
                        choice = match.group(1)

                    # 还查找类似"(A)"或"A."的模式
                    if not choice:
                        match = re.search(r'[\(\s]([A-DZ])[\)\.\s]', ans)
                        if match:
                            choice = match.group(1)

                    if choice and choice in "ABCDZ":
                        # 保存到缓存
                        try:
                            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                            with open(cache_path, 'w', encoding='utf-8') as f:
                                cache_data = {
                                    'choice': choice,
                                    'full_response': ans,
                                    'prompt': prompt
                                }
                                json.dump(cache_data, f, ensure_ascii=False)
                                print(f"✅ 缓存已保存: {cache_path}")
                        except Exception as e:
                            print(f"❌ 保存缓存失败 {cache_path}: {e}")

                        return choice
                except Exception as e:
                    print(f"❌ 调用模型出错: {e}")

                retry -= 1

            # 如果所有尝试都失败，返回Z
            # 保存失败结果到缓存
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'choice': 'Z',
                        'full_response': '调用失败',
                        'prompt': prompt
                    }, f, ensure_ascii=False)
                    print(f"⚠️ 保存默认缓存(Z): {cache_path}")
            except Exception as e:
                print(f"❌ 保存默认缓存失败: {e}")

            return "Z"
        except Exception as e:
            print(f"❌ 处理项目时出错: {e}")
            return "Z"  # 出错时返回默认值

    @staticmethod
    def batch_extract_choices_with_llm(data_rows, num_processes=32, cache_dir='.cache',
                                       use_single_thread=False):
        """
        并发处理多个预测

        Args:
            data_rows: 包含预测和问题的数据行
            num_processes: 要使用的并发进程数
            cache_dir: 缓存结果的目录
            use_single_thread: 是否强制使用单线程处理

        Returns:
            list: 提取的选项列表
        """
        import os
        import multiprocessing as mp
        from functools import partial
        import time

        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)

        # 使用进程池来并发处理项目
        results = []
        total = len(data_rows)

        # 为每行创建简单的字典，只包含必要信息
        row_dicts = []
        for row in data_rows:
            row_dicts.append({
                'prediction': row['prediction'],
                'question': row['question']
            })

        # 如果指定了单线程或进程数为1，则使用单线程处理
        if use_single_thread or num_processes <= 1:
            print(f"使用单线程处理 {total} 个样本...")
            results = []
            start_time = time.time()
            for i, row in enumerate(row_dicts):
                result = CustomDataset._process_single_item(row, cache_dir=cache_dir)
                results.append(result)
                if (i + 1) % max(1, total // 20) == 0:
                    elapsed = time.time() - start_time
                    remaining = (elapsed / (i + 1)) * (total - (i + 1))
                    print(f"单线程进度: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%) - "
                          f"已用时: {elapsed:.1f}秒, 剩余时间: {remaining:.1f}秒")
            return results

        # 创建进程共享的处理函数
        process_func = partial(CustomDataset._process_single_item, cache_dir=cache_dir)

        # 对于Windows，需要使用if __name__=='__main__'来避免子进程递归创建
        # 但在当前环境中，我们直接使用multiprocessing
        print(f"启动 {num_processes} 个进程处理 {total} 个样本")

        try:
            # 设置启动方法为 forkserver 或 spawn 以提高兼容性
            if hasattr(mp, 'get_context'):
                try:
                    # 优先使用 forkserver，它在Linux上通常更可靠
                    ctx = mp.get_context('forkserver')
                    pool = ctx.Pool(processes=num_processes)
                    print("使用 forkserver 方式启动进程池")
                except ValueError:
                    try:
                        # 如果不支持 forkserver，则尝试 spawn
                        ctx = mp.get_context('spawn')
                        pool = ctx.Pool(processes=num_processes)
                        print("使用 spawn 方式启动进程池")
                    except ValueError:
                        # 如果都不支持，则使用默认方式
                        pool = mp.Pool(processes=num_processes)
                        print("使用默认方式启动进程池")
            else:
                # 如果无法设置上下文，则使用默认池
                pool = mp.Pool(processes=num_processes)
                print("使用标准进程池")

            # 使用 imap 可以按顺序得到结果，同时支持并行处理
            chunksize = max(1, total // (num_processes * 4))
            for i, result in enumerate(pool.imap(process_func, row_dicts, chunksize=chunksize)):
                results.append(result)
                processed = i + 1
                if processed % max(1, total // 20) == 0:  # 每5%更新一次进度
                    print(f"进度: {processed}/{total} ({processed / total * 100:.1f}%)")

            pool.close()
            pool.join()

        except Exception as e:
            print(f"并行处理出错: {e}")
            # 发生错误时，尝试单线程处理
            print("尝试单线程处理...")
            results = []
            for i, row in enumerate(row_dicts):
                result = CustomDataset._process_single_item(row, cache_dir=cache_dir)
                results.append(result)
                if (i + 1) % max(1, total // 20) == 0:
                    print(f"单线程进度: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%)")

        print(f"完成所有处理: {len(results)}/{total}")
        return results

