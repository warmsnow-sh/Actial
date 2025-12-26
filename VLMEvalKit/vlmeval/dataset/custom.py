import os
import re
import pandas as pd
import os.path as osp
import json
import numpy as np
import warnings
from tqdm import tqdm
from .image_mcq import ImageMCQDataset
from .utils import DEBUG_MESSAGE, build_judge
from .utils.multiple_choice import mcq_vanilla_eval, mcq_circular_eval
from ..smp import (LMUDataRoot, file_size, load, dump, decode_base64_to_image_file,
                   listinstr, gpt_key_set)
import string
import glob


class CustomDataset(ImageMCQDataset):
    """
    CustomDataset class for multiple-choice questions with multiple images.
    支持多图片的多选题评测数据集，图片以JSON数组格式存储在image字段中。
    """
    TYPE = 'MCQ'

    DATASET_URL = {
        'MMSI_Bench': 'https://huggingface.co/datasets/RunsenXu/MMSI-Bench/resolve/main/MMSI_bench.tsv',
        'ViewSpatial-Bench': 'https://huggingface.co/datasets/warmsnow/ViewSpatial-Bench-vlmeval/resolve/main/ViewSpatial-Bench.tsv',
        'omni-spatial_bench': 'https://huggingface.co/datasets/INno0o/Omni-Spatial-Bench/blob/main/omni_spatial_bench.tsv',
        'view_spatial_bench' : 'https://huggingface.co/datasets/INno0o/View-Spatial-Bench/blob/main/view_spatial_bench.tsv',
        'FAVOR-bench-resize' : 'https://huggingface.co/datasets/INno0o/View-Spatial-Bench/blob/main/FAVOR-bench-resize.tsv',
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
        
        extract_dataset = ['ViewSpatial-Bench','FAVOR-bench-resize']
        
        # 保存extract_options参数
        if dataset in extract_dataset:
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
        
        # print("--------------------------------")
        # print(line['image_path'])
        # print("--------------------------------")
        
        # 处理image_path字段
        if 'image_path' in line:
            tgt_path = line['image_path']
            # 如果是字符串，转换为列表
            if isinstance(tgt_path, str):
                tgt_path = [tgt_path]
            # 如果已经是列表，直接使用
            elif isinstance(tgt_path, list):
                pass  # 保持原样
            else:
                # 如果既不是字符串也不是列表，尝试转换
                tgt_path = [str(tgt_path)]
            
            # 验证路径并修正
            validated_paths = []
            for path in tgt_path:
                # 首先检查原始路径是否存在
                if os.path.exists(path):
                    validated_paths.append(path)
                else:
                    # 如果不存在，尝试与self.img_root进行join
                    joined_path = os.path.join(self.img_root, path)
                    if os.path.exists(joined_path):
                        validated_paths.append(joined_path)
                    else:
                        # 两种路径都不存在，报错
                        raise FileNotFoundError(f"图片路径不存在: 原始路径 '{path}' 和拼接路径 '{joined_path}' 都无法找到")
            
            return validated_paths

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

        # print("--------tgt--------------")
        # print(tgt_path)
        # print("--------line--------------")
        # # print(line)
        # print("--------final-------------------")
        
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
            if self.dataset_name == 'MMSI_Bench':
                post_prompt = ("Answer with the option's letter from the given choices directly. "
                           "Enclose the option's letter within ``.")
                prompt = f'{question}\n{post_prompt}'
            else:
                prompt = question

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
        # print("--------------------------------")
        #print(msgs)
        
        return msgs

    
    def evaluate(self, eval_file, **judge_kwargs):
        if judge_kwargs.get('use_verifier', False):
            print("not supported")
        else:
            return self.evaluate_heuristic(eval_file, **judge_kwargs)

    def evaluate_heuristic(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import (
            report_acc, report_acc_MMT, report_acc_MMSci, mcq_circular_eval, mcq_vanilla_eval
        )
        
        # 添加调试信息
        print(f"=== 调试信息 ===")
        print(f"数据集名称: {self.dataset_name}")
        print(f"self.data 是否为 None: {self.data is None}")
        if self.data is not None:
            print(f"self.data 形状: {self.data.shape}")
            print(f"self.data 列名: {list(self.data.columns)}")
            print(f"是否包含 answer 列: {'answer' in self.data.columns}")
            if 'answer' in self.data.columns:
                print(f"answer 列的前5个值: {self.data['answer'].head().tolist()}")
            if 'index' in self.data.columns:
                print(f"index 列的前5个值: {self.data['index'].head().tolist()}")
        print(f"=== 调试信息结束 ===")
        
        
        if self.dataset_name == 'ViewSpatial-Bench':
            # 处理ViewSpatial-Bench数据集：将answer内容映射回选项字母
            print("=== 处理ViewSpatial-Bench数据集 ===")
            
            # 为每一行创建answer内容到选项字母的映射
            def map_answer_to_option(row):
                answer_content = row['answer']
                # 在A、B、C、D选项中寻找匹配的内容
                for option_letter in ['A', 'B', 'C', 'D']:
                    if option_letter in row and str(row[option_letter]).strip() == str(answer_content).strip():
                        return option_letter
                # 如果没有找到完全匹配，记录警告并返回原值
                print(f"警告: 无法为答案 '{answer_content}' 找到对应的选项字母，行索引: {row['index']}")
                return answer_content
            
            # 创建新的answer列，映射到选项字母
            self.data['answer_mapped'] = self.data.apply(map_answer_to_option, axis=1)
            
            # 用映射后的答案替换原来的answer列
            self.data['answer'] = self.data['answer_mapped']
            
            # 删除临时列
            self.data = self.data.drop(columns=['answer_mapped'])
            
            print(f"答案映射完成，前5个映射后的答案: {self.data['answer'].head().tolist()}")
            print("=== ViewSpatial-Bench处理完成 ===")
        
        
        # assert dataset is not None
        dataset_map = {
            'MMBench_TEST_EN': 'MMBench', 'MMBench_TEST_EN_V11': 'MMBench_V11',
            'MMBench_TEST_CN': 'MMBench_CN', 'MMBench_TEST_CN_V11': 'MMBench_CN_V11'
        }
        dataset = self.dataset_name
        if dataset in dataset_map:
            dataset = dataset_map[dataset]
        nproc = judge_kwargs.pop('nproc', 16)

        suffix = eval_file.split('.')[-1]
        model = judge_kwargs.get('model', 'exact_matching')
        assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model
        
        model = 'exact_matching'
        if model == 'exact_matching':
            model = None
        elif gpt_key_set():
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None
        else:
            warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
            model = None

        result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')
        
        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        
        print(f"=== 预测数据调试信息 ===")
        print(f"预测数据形状: {data.shape}")
        print(f"预测数据列名: {list(data.columns)}")
        print(f"预测数据的前5个prediction: {data['prediction'].head().tolist()}")
        print(f"=== 预测数据调试信息结束 ===")
        
        # 对prediction数据使用启发式提取
        print("=== 开始对prediction进行启发式提取 ===")
        original_predictions = data['prediction'].tolist()
        extracted_predictions = []
        
        for i, pred in enumerate(original_predictions):
            extracted = self.extract_single_choice_with_word_boundary(pred)
            extracted_predictions.append(extracted if extracted is not None else pred)
            
            # # 打印所有示例的提取结果
            # print(f"原始prediction[{i}]: '{pred}' -> 提取结果: '{extracted}'")
        
        data['prediction'] = extracted_predictions
        print(f"启发式提取完成，前5个提取后的prediction: {data['prediction'].head().tolist()}")
        # print("=== prediction启发式提取完成 ===")
        
        # If not choice label, then use lower case
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        meta = self.data
        
        # 检查meta数据
        if meta is None:
            raise ValueError("self.data (meta) 为 None，数据加载失败！")
        
        if 'answer' not in meta.columns:
            raise ValueError(f"meta数据中没有'answer'列，可用列: {list(meta.columns)}")
            
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

        # 添加GT与prediction的详细对比调试信息
        print("=== GT与Prediction对比调试 ===")
        answer_map = {i: c for i, c in zip(meta['index'], meta['answer'])}
        print(f"answer_map示例: {dict(list(answer_map.items())[:5])}")
        
        # 确保data中所有index都在answer_map中
        data = data[data['index'].isin(answer_map)]
        data['GT'] = [answer_map[idx] for idx in data['index']]
        
        # 打印前10个样本的详细对比
        for i in range(min(10, len(data))):
            row = data.iloc[i]
            print(f"样本{i}: index={row['index']}, GT='{row['GT']}', prediction='{row['prediction']}', 匹配={row['GT'] == row['prediction']}")
        
        # print("=== GT与Prediction对比调试结束 ===")
        
        # # 添加meta数据结构调试
        # print("=== meta数据结构调试 ===")
        # print(f"meta数据形状: {meta.shape}")
        # print(f"meta列名: {list(meta.columns)}")
        # if len(meta) > 0:
        #     sample_meta = meta.iloc[0]
        #     print(f"meta样本0:")
        #     print(f"  所有列: {list(sample_meta.index)}")
        #     print(f"  A列: {sample_meta.get('A', 'NOT_FOUND')}")
        #     print(f"  B列: {sample_meta.get('B', 'NOT_FOUND')}")
        #     print(f"  C列: {sample_meta.get('C', 'NOT_FOUND')}")
        #     print(f"  D列: {sample_meta.get('D', 'NOT_FOUND')}")
        # print("=== meta数据结构调试结束 ===")
        
        # 测试can_infer函数的表现
        # print("=== 测试can_infer函数表现 ===")
        # from ..utils import can_infer
        
        # # 模拟extract_answer_from_item的逻辑
        # def build_choices(item):
        #     choices = {}
            
        #     # 首先尝试从A、B、C、D列获取选项（适用于ViewSpatial-Bench）
        #     for c in string.ascii_uppercase:
        #         if c in item and not pd.isna(item[c]):
        #             choices[c] = item[c]
            
        #     # 如果没有找到A、B、C、D列，尝试从question字段提取（适用于MMSI_Bench等）
        #     if not choices and 'question' in item:
        #         question = item['question']
        #         # 从question中提取选项 - 支持多种格式
        #         option_patterns = [
        #             r'Options:\s*(.*)',
        #             r'Given the options:\s*(.*)',
        #             r'Choose from:\s*(.*)',
        #             r'Select from:\s*(.*)',
        #             r'Choices:\s*(.*)'
        #         ]
                
        #         for pattern in option_patterns:
        #             match = re.search(pattern, question, re.IGNORECASE | re.DOTALL)
        #             if match:
        #                 options_text = match.group(1).strip()
        #                 # 匹配 A: content, B: content 格式（支持A-E）
        #                 choice_pattern = r'([A-E])\s*[.:]\s*(.*?)(?=\s+[A-E]\s*[.:]|$)'
        #                 matches = re.findall(choice_pattern, options_text, re.DOTALL)
        #                 if matches:
        #                     choices = {m[0]: m[1].strip().rstrip(',').strip() for m in matches}
        #                     break
                
        #         # 如果上面的模式都没匹配到，尝试直接在整个question中寻找A. B. C. D.格式的选项
        #         if not choices:
        #             # 匹配整个question中的选项格式
        #             choice_pattern = r'([A-E])\s*[.:]\s*([^\n\r]+)'
        #             matches = re.findall(choice_pattern, question)
        #             if len(matches) >= 2:  # 至少要有2个选项才认为是有效的
        #                 choices = {m[0]: m[1].strip().rstrip(',').strip() for m in matches}
        
        #     return choices
        
        # # 测试前5个样本
        # for i in range(min(5, len(data))):
        #     row = data.iloc[i]
            
        #     # 添加数据结构调试
        #     print(f"样本{i}数据结构调试:")
        #     print(f"  所有列: {list(row.index)}")
        #     print(f"  A列: {row.get('A', 'NOT_FOUND')}")
        #     print(f"  B列: {row.get('B', 'NOT_FOUND')}")
        #     print(f"  C列: {row.get('C', 'NOT_FOUND')}")
        #     print(f"  D列: {row.get('D', 'NOT_FOUND')}")
        #     print(f"  question字段: {row.get('question', 'NOT_FOUND')[:200]}...")  # 只显示前200字符
            
        #     choices = build_choices(row)
        #     inferred = can_infer(row['prediction'], choices)
        #     print(f"  choices: {choices}")
        #     print(f"  prediction: '{row['prediction']}', can_infer结果: '{inferred}', GT: '{row['GT']}'")
        #     print()
        
        # print("=== can_infer测试结束 ===")
        
        circular = False
        # if listinstr(['mmbench', 'ccbench', 'circular', 'mmcr'], dataset.lower()):
        #     # 特殊处理：ViewSpatial-Bench不应该进入circular模式
        #     if self.dataset_name != 'ViewSpatial-Bench':
        #         data = load(eval_file)
        #         data['index'] = [int(x) for x in data['index']]
        #         dump(data, eval_file)
        #         circular = True

        # # 特殊处理：确保ViewSpatial-Bench使用vanilla评估模式
        # if self.dataset_name == 'ViewSpatial-Bench':
        #     circular = False
        #     print("强制ViewSpatial-Bench使用vanilla评估模式")
        
        # # 添加调试信息
        # print(f"=== 评估模式选择 ===")
        # print(f"数据集: {self.dataset_name}")
        # print(f"dataset.lower(): {dataset.lower()}")
        # print(f"listinstr结果: {listinstr(['mmbench', 'ccbench', 'circular', 'mmcr'], dataset.lower())}")
        # print(f"最终circular模式: {circular}")
        # print(f"将使用: {'mcq_circular_eval' if circular else 'mcq_vanilla_eval'}")
        # print(f"=== 评估模式选择结束 ===")
        # circular = False
        # # 添加mcq_vanilla_eval调试信息
        # print(f"=== mcq_vanilla_eval调试 ===")
        # print(f"result_file: {result_file}")
        # print(f"result_file是否存在: {osp.exists(result_file)}")
        # if osp.exists(result_file):
        #     old_result = load(result_file)
        #     print(f"旧result文件大小: {len(old_result)}")
        #     print(f"旧result示例: {dict(list(old_result.items())[:3])}")
        # print(f"传入mcq_vanilla_eval的data形状: {data.shape}")
        # print(f"传入mcq_vanilla_eval的meta形状: {meta.shape}")
        # print(f"=== mcq_vanilla_eval调试结束 ===")
        
        # 强制重新评估：删除旧的result_file
        if osp.exists(result_file):
            os.remove(result_file)
            print(f"已删除旧的result_file: {result_file}")
            
            
        data = self.mcq_vanilla_eval_simple(model, data, meta, nproc, result_file, self.dataset_name)
        
        # if circular:
        #     data = mcq_circular_eval(model, data, meta, nproc, result_file, self.dataset_name)
        # else:
        #     # 检查数据是否包含A、B、C、D列，如果没有则使用自定义评估函数
        #     has_option_columns = any(col in meta.columns for col in ['A', 'B', 'C', 'D'])
            
        #     if not has_option_columns:
        #         print(f"数据集 {self.dataset_name} 没有A、B、C、D选项列，使用简化评估函数")
        #         data = self.mcq_vanilla_eval_simple(model, data, meta, nproc, result_file, self.dataset_name)
        #     else:
        #         print(f"数据集 {self.dataset_name} 包含选项列，使用标准评估函数")
        #         data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)

        # # 添加评估后的调试信息
        # print(f"=== 评估后调试信息 ===")
        # print(f"返回data形状: {data.shape}")
        # print(f"返回data列名: {list(data.columns)}")
        # if 'hit' in data.columns:
        #     print(f"hit列的前10个值: {data['hit'].head(10).tolist()}")
        #     print(f"hit列的统计: 总数={len(data['hit'])}, 成功={data['hit'].sum()}, 准确率={data['hit'].mean():.4f}")
        # else:
        #     print("警告: 返回的data中没有hit列!")
        # print(f"=== 评估后调试信息结束 ===")
        
        # load split
        eval_record = eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}')
        dump(data, eval_record)
        data = load(eval_record)

        # May have different report acc functions for different datasets
        if 'MMT' in dataset:
            acc = report_acc_MMT(data)
        elif 'MMSci' in dataset:
            acc = report_acc_MMSci(data)
        else:
            acc = report_acc(data)

        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(acc, score_file)





        return acc


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
 
    def extract_answer_from_item_custom(self, model, item, dataset_name=None):
        """
        自定义的答案提取函数，支持从question字段中提取选项（适用于MMSI_Bench）
        """
        # 构建选项字典，优先从A、B、C、D列，其次从question字段
        choices = {}
        
        # 首先尝试从A、B、C、D列获取选项（适用于ViewSpatial-Bench）
        for c in string.ascii_uppercase:
            if c in item and not pd.isna(item[c]):
                choices[c] = item[c]
        
        # 如果没有找到A、B、C、D列，尝试从question字段提取（适用于MMSI_Bench等）
        if not choices and 'question' in item:
            question = item['question']
            # 从question中提取选项 - 支持多种格式
            option_patterns = [
                r'Options:\s*(.*)',
                r'Given the options:\s*(.*)',
                r'Choose from:\s*(.*)',
                r'Select from:\s*(.*)',
                r'Choices:\s*(.*)'
            ]
            
            for pattern in option_patterns:
                match = re.search(pattern, question, re.IGNORECASE | re.DOTALL)
                if match:
                    options_text = match.group(1).strip()
                    # 匹配 A: content, B: content 格式（支持A-E）
                    choice_pattern = r'([A-E])\s*[.:]\s*(.*?)(?=\s+[A-E]\s*[.:]|$)'
                    matches = re.findall(choice_pattern, options_text, re.DOTALL)
                    if matches:
                        choices = {m[0]: m[1].strip().rstrip(',').strip() for m in matches}
                        break
        
        # 使用can_infer直接提取答案
        from ..utils import can_infer
        inferred = can_infer(item['prediction'], choices)
        
        if inferred:
            return dict(opt=inferred, log=f'Custom extraction: {item["prediction"]} -> {inferred}')
        else:
            return dict(opt='Z', log=f'Failed to extract from: {item["prediction"]}')
 
    def eval_vanilla_simple(self, model, item, dataset_name=None):
        """
        简化的vanilla评估函数，直接比较提取后的选项与GT
        """
        # 使用启发式提取获得选项字母
        extracted = self.extract_single_choice_with_word_boundary(item['prediction'])
        
        # 如果提取失败，尝试直接使用prediction（可能已经是选项字母）
        if extracted is None:
            extracted = str(item['prediction']).strip()
        
        # 直接与GT比较
        if extracted == item['GT']:
            return dict(hit=1, log=f'Direct match: {item["prediction"]} -> {extracted} == {item["GT"]}')
        else:
            return dict(hit=0, log=f'No match: {item["prediction"]} -> {extracted} != {item["GT"]}')
    
    def mcq_vanilla_eval_simple(self, model, data, meta, nproc, result_file, dataset_name=None):
        """
        简化的mcq_vanilla_eval，直接比较选项字母
        """
        result = {}
        if osp.exists(result_file):
            result = load(result_file)
        answer_map = {i: c for i, c in zip(meta['index'], meta['answer'])}

        data = data[data['index'].isin(answer_map)]
        data['GT'] = [answer_map[idx] for idx in data['index']]
        items = []

        for i in range(len(data)):
            item = data.iloc[i]
            if item['index'] not in result:
                items.append(item)

        # 简单的直接评估
        for i, item in enumerate(items):
            result[item['index']] = self.eval_vanilla_simple(model, item, dataset_name)
            
        # 保存结果
        dump(result, result_file)
            
        data['hit'] = [result[i]['hit'] for i in data['index']]
        data['log'] = [result[i]['log'] for i in data['index']]
        if 'GT' in data:
            data.pop('GT')
        return data
 