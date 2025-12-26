from __future__ import annotations

import os
import torch
import re
import logging
import warnings
import math
import sys
import tempfile
import cv2
from PIL import Image
import base64
from io import BytesIO
from mimetypes import guess_type

from .base import BaseModel
from .qwen2_vl.prompt import Qwen2VLPromptMixin
from .qwen2_vl.model import ensure_image_url, ensure_video_url
from ..smp import get_rank_and_world_size, get_gpu_memory, listinstr
from ..dataset import DATASET_MODALITY


from openai import APIConnectionError, OpenAI
from openai.pagination import SyncPage
from openai.types.model import Model


def get_first_model(client: OpenAI) -> str:
    """
    Get the first model from the vLLM server.
    """
    try:
        models: SyncPage[Model] = client.models.list()
    except APIConnectionError as e:
        raise RuntimeError(
            "Failed to get the list of models from the vLLM server at "
            f"{client.base_url} with API key {client.api_key}. Check\n"
            "1. the server is running\n"
            "2. the server URL is correct\n"
            "3. the API key is correct"
        ) from e

    if len(models.data) == 0:
        raise RuntimeError(f"No models found on the vLLM server at {client.base_url}")

    return models.data[0].id
# VLLM maximum image input number
VLLM_MAX_IMAGE_INPUT_NUM = 24


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def create_image_content(image_path, min_pixels, max_pixels):
    base64_image, mime_type = encode_image(image_path)
    return {
        "type": "image",
        "image": f"data:{mime_type};base64,{base64_image}",
        'min_pixels': min_pixels,
        'max_pixels': max_pixels
    }


def _encode_image(image, image_format):
    with BytesIO() as output:
        image.convert("RGB").save(output, format=image_format)
        base64_encoded_data = base64.b64encode(output.getvalue()).decode("utf-8")
    return base64_encoded_data


def _rgba_to_rgb(image):
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, image).convert("RGB")


def _resize_image(image, max_side):
    resize_scale = max_side / max(image.size)
    new_size = (
        int(image.size[0] * resize_scale),
        int(image.size[1] * resize_scale),
    )
    return image.resize(new_size)



def encode_image(image_path, max_side=None):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    image_format = mime_type.split("/")[-1].upper() if mime_type else "JPEG"

    image = Image.open(image_path)
    # Handle the alpha channel
    if image.mode == "RGBA":
        image = _rgba_to_rgb(image)
    if max_side:
        image = _resize_image(image, max_side)
    encoded_image = _encode_image(image, image_format)

    return encoded_image, mime_type





def process_video(video_path, num_frames, min_pixels, max_pixels):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

    # the sampling rate using max number of frames
    sampling_gap_maxframe = (
        1 if not num_frames else math.ceil(frame_count / num_frames)
    )
    sampling_gap = max(math.ceil(fps / 5), sampling_gap_maxframe)

    frame_number = 0
    images = []

    while True:
        success, frame = cap.read()
        if not success:
            break
        # Sample frames based on the dynamic sampling rate
        if frame_number % sampling_gap == 0:
            # Create a temporary file for the frame
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as temp_frame:
                cv2.imwrite(temp_frame.name, frame)
                images.append(create_image_content(temp_frame.name, min_pixels, max_pixels))
                os.remove(temp_frame.name)
        frame_number += 1
    if frame_number == 0:
        raise ValueError(f"Failed to read video from {video_path}, check data...")
    logging.info(
        f"Sampled {len(images)}/{frame_number} frames from video {video_path}"
    )
    cap.release()
    return images


def extract_answer_tag(s: str, verbose=False) -> str:
    # Regular expression to match content between <answer> and </answer>
    matches = re.findall(r'<answer>(.*?)</answer>', s, re.DOTALL)
    if len(matches) == 0:
        if verbose:
            print("No <answer>...</answer> blocks found.")
        return None
    elif len(matches) > 1:
        if verbose:
            print("Multiple <answer>...</answer> blocks found.")
        return None
    else:
        return matches[0].strip()


def extract_response_for_eval(s: str, verbose=False):
    ret = None
    # <answer> {}</answer>
    if ret is None:
        ret = extract_answer_tag(s, verbose=verbose)
    # </think>
    elif '</think>' in s:
        ret = s.split('</think>')[-1]
    if ret is None:
        ret = s
    return ret


def setup_visible_devices_per_rank():
    total_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    assert world_size == 1, "Only support world_size == 1 for vLLM inference"
    num_gpus = total_gpus // world_size
    start_idx = rank * num_gpus
    assigned_devices = list(range(start_idx, start_idx + num_gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in assigned_devices)
    logging.info(f"[Rank {rank}] Visible GPUs: {assigned_devices}")
    return num_gpus


class ActialChat(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens=8192,
        
        # top_p=0.95,
        top_p=0.95,
        top_k=1,
        # temperature=0.6,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,  
        verbose: bool = False,
        save_raw_output: bool = False,
        output_dir: str = "./outputs",
        use_openai_client: bool = True,
        **kwargs,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            use_cache=True
        )
        self.system_prompt = system_prompt
        # print("--------------------------------")
        # print(f"system_prompt: {self.system_prompt}")
        # print("--------------------------------")
        
        
        self.verbose = verbose
        self.post_process = post_process
        self.save_raw_output = save_raw_output
        self.output_dir = output_dir
        self.fps = 2.0
        self.nframe = 64
        self.FRAME_FACTOR = 2
        rank, world_size = get_rank_and_world_size()
        assert model_path is not None
        self.model_path = model_path

        MODEL_CLS = None


        
        
        if listinstr(['2.5', '2_5', 'qwen25'], model_path.lower()):
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            MODEL_CLS = Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
            MODEL_CLS = Qwen2VLForConditionalGeneration
            self.processor = Qwen2VLProcessor.from_pretrained(model_path)

        print(f"now testing.....{self.model_path}")
        # gpu_mems = get_gpu_memory()
        # max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        # assert max_gpu_mem > 0

        self.use_vllm = kwargs.get('use_vllm', False)
        self.limit_mm_per_prompt = 100
        self.use_openai_client = kwargs.get('use_openai_client', True)
        
        # self.use_openai_client = True
        
        
        
        print(f"use_openai_client: {self.use_openai_client}")
        
        if self.use_openai_client:
            import os
            self.base_ip = os.environ.get('LOCAL_IP',"0.0.0.0")
            self.base_port = os.environ.get('VLLM_LOCAL_PORT',"8000")
            self.base_url = kwargs.get('base_url',"http://"+self.base_ip+":"+self.base_port+"/v1")
            self.api_key = kwargs.get('api_key',"EMPTY")
            
        
        
        if self.use_vllm and not self.use_openai_client:
            from vllm import LLM
            gpu_count = setup_visible_devices_per_rank()
            if gpu_count >= 8:
                tp_size = 8
            elif gpu_count >= 4:
                tp_size = 4
            elif gpu_count >= 2:
                tp_size = 2
            else:
                tp_size = 1
            logging.info(
                f'Using vLLM for {self.model_path} inference with {tp_size} GPUs (available: {gpu_count})'
            )
            import os
            if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
                logging.warning(
                    'VLLM_WORKER_MULTIPROC_METHOD is not set to spawn.'
                    'Use \'export VLLM_WORKER_MULTIPROC_METHOD=spawn\' to avoid potential multi-process issues'
                )

            self.llm = LLM(
                model=self.model_path,
                max_num_seqs=128,
                max_model_len=64000,
                limit_mm_per_prompt={"image": self.limit_mm_per_prompt},
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=kwargs.get("gpu_utils", 0.9),
            )
        else:
            
            
            if self.use_openai_client and self.use_vllm:
                self.client = OpenAI(base_url=self.base_url,api_key=self.api_key)
                self.model_name = kwargs.get('model_name', get_first_model(self.client))
                print("--------------------------------")
                print(f"self.model_name: {self.model_name}")
                print("--------------------------------")
                try:
                    self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": "Hello, how are you?"}],
                    )
                except Exception as e:
                    print("--------------------------------------------------------------------------")
                    # print(f"error: {e}")
                    print(f"self.base_url: {self.base_url}")
                    print(f"self.api_key: {self.api_key}")
                    print(f"self.model_name: {self.model_name}")
                    print("--------------------------------------------------------------------------")
                print(f"success to use openai client")
            else:
                self.model = MODEL_CLS.from_pretrained(
                    model_path,
                    torch_dtype="auto",
                    device_map="auto",
                    attn_implementation="flash_attention_2",
                )
                self.model.eval()
        torch.cuda.empty_cache()

    def _prepare_content(
        self, inputs: list[dict[str, str]], dataset: str | None = None
    ) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []

        for s in inputs:
            if s["type"] == "image":
                item = {"type": "image", "image": ensure_image_url(s["value"])}
                if dataset == "OCRBench":
                    item["min_pixels"] = 10 * 10 * 28 * 28
                    warnings.warn(
                        f"OCRBench dataset uses custom min_pixels={item['min_pixels']}"
                    )
                    if self.max_pixels is not None:
                        item["max_pixels"] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item["min_pixels"] = self.min_pixels
                    if self.max_pixels is not None:
                        item["max_pixels"] = self.max_pixels
            elif s["type"] == "video":
                item = {"type": "video", "video": ensure_video_url(s["value"])}
                if self.fps is not None:
                    item["fps"] = self.fps
                elif self.nframe is not None:
                    import cv2

                    video = cv2.VideoCapture(s["value"])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = (
                            frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        )
                        print(f"use {new_frame_count} for {s['value']}")
                        item["nframes"] = new_frame_count
                    else:
                        item["nframes"] = self.nframe
            elif s["type"] == "text":
                item = {"type": "text", "text": s["value"]}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)

        return content

    def _prepare_content_vllm(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        video_inputs = [s for s in inputs if s['type'] == 'video']
        video_count = len(video_inputs)
        cur_image_count = 0
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                if cur_image_count < self.limit_mm_per_prompt:
                    content.append(item)
                    cur_image_count += 1
                else:
                    logging.warning(
                        f"Number of images exceeds the limit of {self.limit_mm_per_prompt}. "
                        f"Only the first {self.limit_mm_per_prompt} images will be used."
                    )
            elif s['type'] == 'video':
                if video_count > 1:
                    logging.warning(
                        "Multiple videos detected. Using video frames for each video"
                    )
                    if dataset == 'OCRBench':
                        min_pixels = 10 * 10 * 28 * 28
                        warnings.warn(f"OCRBench dataset uses custom min_pixels={min_pixels}")
                        if self.max_pixels is not None:
                            max_pixels = self.max_pixels
                    else:
                        if self.min_pixels is not None:
                            min_pixels = self.min_pixels
                        if self.max_pixels is not None:
                            max_pixels = self.max_pixels

                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()

                    frames_per_video = max(1, self.limit_mm_per_prompt // video_count)
                    content.append({"type": "text", "text": "<video frames start>"})
                    content.extend(process_video(s['value'], frames_per_video, min_pixels, max_pixels))
                    content.append({"type": "text", "text": "<video frames end>"})

                else:
                    item = {
                        'type': 'video',
                        'video': ensure_video_url(s['value'])
                    }
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                    if self.total_pixels is not None:
                        item['total_pixels'] = self.total_pixels
                    if self.fps is not None:
                        item['fps'] = self.fps
                    elif self.nframe is not None:
                        import cv2
                        video = cv2.VideoCapture(s['value'])
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        video.release()
                        if frame_count < self.nframe:
                            new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                            print(f"use {new_frame_count} for {s['value']}")
                            item['nframes'] = new_frame_count
                        else:
                            item['nframes'] = self.nframe
                    content.append(item)
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
                content.append(item)
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
        return content

    def _prepare_content_openai(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        为OpenAI API准备标准格式的消息内容，参考openai_mm.py的格式
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        
        for s in inputs:
            if s['type'] == 'image':
                # 完全按照openai_mm.py的图像格式
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": ensure_image_url(s['value'])
                    }
                })
            elif s['type'] == 'video':
                # 完全按照openai_mm.py的视频格式
                content.append({
                    "type": "video_url", 
                    "video_url": {
                        "url": ensure_video_url(s['value'])
                    }
                })
            elif s['type'] == 'text':
                # 完全按照openai_mm.py的文本格式
                content.append({
                    "type": "text",
                    "text": s['value']
                })
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
        
        return content

    def generate_inner_transformers(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical(
                "qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'"
            )
            raise err

        def replace_last_dot(input_string):
            if input_string.endswith("."):
                return input_string[:-1]
            else:
                return input_string

        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {"role": "user", "content": self._prepare_content(message, dataset=dataset)}
        )
        if self.verbose:
            print(f"\033[31m{messages}\033[0m")

        text = self.processor.apply_chat_template(
            [messages], tokenize=False, add_generation_prompt=True
        )

        images, videos = process_vision_info([messages])
        inputs = self.processor(
            text=text, images=images, videos=videos, padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]
        raw_output = response  # save raw output
        if self.post_process:
            response = extract_response_for_eval(response, verbose=self.verbose)
            # response = replace_last_dot(response)

        if self.save_raw_output:
            os.makedirs(self.output_dir, exist_ok=True)
            output_file = os.path.join(
                self.output_dir, f"{self.model_path.split('/')[-1]}_{dataset}.jsonl"
            )
            if message[0]['type'] == 'image':
                id = message[0]['value'].rsplit('/')[-1].split('.')[0]
            else:
                id = None
            import jsonlines
            with jsonlines.open(output_file, mode='a') as writer:
                writer.write({"id": id, "response": raw_output})

        if self.verbose:
            print(f"\033[32m{response}\033[0m")
        return response

    def generate_inner_vllm(self, message, dataset=None):
        
        
        # print(f"message: {message}")
        
        from vllm import SamplingParams
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
            raise err

        
        # print("---------------------b-------------------------")
        
        messages = []
        
        # print(f"self.system_prompt: {self.system_prompt}")
        
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        try:
            messages.append({'role': 'user', 'content': self._prepare_content_vllm(message, dataset=dataset)})
        except Exception as e:
            print(f"error: {e}")
            print(f"message: {message}")
            print(f"dataset: {dataset}")
            print(f"self.system_prompt: {self.system_prompt}")
            print(f"self.processor: {self.processor}")
        #print("--------------------b1-------------------------")
        if self.verbose:
            print(f'[31m{messages}[0m')

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        #print("--------------------b2-------------------------")
        images, videos = process_vision_info(messages)
        

        #print("---------------------c-------------------------")
        
        # 特殊的视频处理逻辑（参考qwen2_vl）
        videos_nd = None
        if videos and DATASET_MODALITY(dataset) == 'VIDEO' and 'megabench' not in dataset.lower():
            assert len(videos) == 1
            videos_nd = [videos[0].detach().cpu().numpy().transpose(0, 2, 3, 1)]
            
            # 检查视频序列长度
            if videos_nd[0].shape[0] > VLLM_MAX_IMAGE_INPUT_NUM:
                print('video input sequence may be too long for vllm, Maybe cannot generate response for VLLM')
        
        sampling_params = SamplingParams(
            max_tokens=8192,
            temperature=self.generate_kwargs.get('temperature', 0.0),
            top_p=self.generate_kwargs.get('top_p', 1.0),
            top_k=self.generate_kwargs.get('top_k', -1),
            repetition_penalty=self.generate_kwargs.get('repetition_penalty', 1.0),
        )
        
        #print("---------------------d-------------------------")
        
        
        if self.use_openai_client:
            # 为OpenAI客户端准备标准格式的消息
            openai_messages = []
            if self.system_prompt is not None:
                openai_messages.append({'role': 'system', 'content': self.system_prompt})
            
            # 使用专门的OpenAI消息准备方法
            openai_content = self._prepare_content_openai(message, dataset=dataset)
            openai_messages.append({'role': 'user', 'content': openai_content})
            
            # print("--------------------------------")
            # print(f"system_prompt: {self.system_prompt}")
            # print(f"openai_messages: {openai_messages}")

            # print(f"openai_messages: {openai_messages}")
            
            
            try:
                outputs = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=openai_messages,
                    max_tokens=32000,
                    temperature=self.generate_kwargs.get('temperature', 0.0),
                    top_p=self.generate_kwargs.get('top_p', 1.0),
                )
                # print(f"outputs: {outputs}")
                # # 提取响应文本
                # generated_text = outputs.choices[0].message.content
                
            except Exception as e:
                print("error in openai client")
                print(f"error: {e}")
                print(f"openai_messages: {openai_messages}")
                generated_text = f"Error: {str(e)}"
        else:
            if images:
                outputs = self.llm.generate(
                    {
                        "prompt": text,
                        "multi_modal_data": {"image": images},
                    },
                    sampling_params=sampling_params,
                    use_tqdm=False
                )
            elif videos_nd:
                # 使用特殊的视频输入格式
                video_inputs = {
                    "prompt": text,
                    "multi_modal_data": {"video": videos_nd[0]},
                    "mm_processor_kwargs": {}
                }
                outputs = self.llm.generate(
                    video_inputs,
                    sampling_params=sampling_params,
                    use_tqdm=False
                )
            elif videos:
                # 原来的视频处理方式（作为后备）
                outputs = self.llm.generate(
                    {
                        "prompt": text,
                        "multi_modal_data": {"video": videos},
                    },
                    sampling_params=sampling_params,
                    use_tqdm=False
                )
            else:
                outputs = self.llm.generate(
                    {
                        "prompt": text,
                    },
                    sampling_params=sampling_params,
                    use_tqdm=False
                )

        #print(f"outputs: {outputs}")
        if self.use_openai_client:
            generated_text = outputs.choices[0].message.content
        else:
            generated_text = outputs[0].outputs[0].text
        # print("--------------------------------")
        
        raw_output = generated_text # save raw output

        # print(f"generated_text: {generated_text}")
        
        if self.post_process:
            generated_text = extract_response_for_eval(generated_text, verbose=self.verbose)

        # print(f"processed_generated_text: {generated_text}")
        
        if self.save_raw_output:
            os.makedirs(self.output_dir, exist_ok=True)
            output_file = os.path.join(
                self.output_dir, f"{self.model_path.split('/')[-1]}_{dataset}.jsonl"
            )
            if message[0]['type'] == 'image':
                id_source = message[0]['value']
                id = id_source.rsplit('/')[-1].split('.')[0] if '/' in id_source else id_source
            elif message[0]['type'] == 'video':
                id_source = message[0]['value']
                id = id_source.rsplit('/')[-1].split('.')[0] if '/' in id_source else id_source
            else:
                id = 'text_input'

            import jsonlines
            with jsonlines.open(output_file, mode='a') as writer:
                writer.write({"id": id, "response": raw_output})

        if self.verbose:
            print(f'[32m{generated_text}[0m')
        return generated_text

    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        else:
            return self.generate_inner_transformers(message, dataset=dataset)
