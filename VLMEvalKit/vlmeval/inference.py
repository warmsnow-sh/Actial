import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

FAIL_MSG = 'Failed to obtain answer via API.'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


def infer_one_data(model, data_item, dataset, dataset_name, verbose=False):
    """推理单个数据项的函数"""
    idx = data_item['index']
    
    if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
        struct = model.build_prompt(data_item, dataset=dataset_name)
    else:
        struct = dataset.build_prompt(data_item)
    # print(f"struct: {struct}")
    
    # If `SKIP_ERR` flag is set, the model will skip the generation if error is encountered
    if os.environ.get('SKIP_ERR', False) == '1':
        try:
            response = model.generate(message=struct, dataset=dataset_name)
        except RuntimeError as err:
            torch.cuda.synchronize()
            warnings.warn(f'{type(err)} {str(err)}')
            response = f'{FAIL_MSG}: {type(err)} {str(err)}'
            print(f"response: {response}")
    else:
        response = model.generate(message=struct, dataset=dataset_name)
        print(f"response: {response}")
    torch.cuda.empty_cache()

    if verbose:
        print(response, flush=True)

    return idx, response


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, index_set=None, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'

    # To reuse records in MMBench_V11
    if dataset_name in ['MMBench', 'MMBench_CN']:
        v11_pred = f'{work_dir}/{model_name}_{dataset_name}_V11.xlsx'
        if osp.exists(v11_pred):
            try:
                reuse_inds = load('http://opencompass.openxlab.space/utils/mmb_reuse.pkl')
                data = load(v11_pred)
                ans_map = {x: y for x, y in zip(data['index'], data['prediction']) if x in reuse_inds}
                dump(ans_map, out_file)
            except Exception as err:
                print(type(err), err)

    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if ignore_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    if index_set is not None:
        res = {k: v for k, v in res.items() if k in index_set}
    os.remove(out_file)
    return res


def infer_data(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4, use_vllm=False):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    kwargs = {}
    if model_name is not None and (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
    ):
        kwargs = {'use_vllm': use_vllm}

    # (25.06.05) In newer version of transformers (after 4.50), with device_map='auto' and torchrun launcher,
    # Transformers automatically adopt TP parallelism, which leads to compatibility problems with VLMEvalKit
    # (In VLMEvalKit, we use torchrun to launch multiple model instances on a single node).
    # To bypass this problem, we unset `WORLD_SIZE` before building the model to not use TP parallel.
    ws_bak = os.environ.pop('WORLD_SIZE', None)
    model = supported_VLM[model_name](**kwargs) if isinstance(model, str) else model
    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak

    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            index_set=set(indices),
            api_nproc=api_nproc)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model
    else:
        model.set_dump_image(dataset.dump_image)

    # 使用多线程异步推理
    res_lock = threading.Lock()
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    
    print("parallel inference size: ", api_nproc)
    
    # 准备任务队列
    pending_tasks = 0
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            task_queue.put((i, idx, data.iloc[i]))
            pending_tasks += 1
    
    # 工作线程函数
    def worker():
        while True:
            try:
                # 从队列中取任务，超时1秒
                i, idx, data_item = task_queue.get(timeout=1)
                # print("-------------------b---------------------------")
                # 执行推理
                _, response = infer_one_data(model, data_item, dataset, dataset_name, verbose)
                
                # 将结果放入结果队列
                result_queue.put((i, idx, response))
                
                # 标记任务完成
                task_queue.task_done()
                
            except queue.Empty:
                # 队列为空，退出线程
                break
            except Exception as e:
                # 处理异常
                result_queue.put((i, idx, f'{FAIL_MSG}: {str(e)}'))
                task_queue.task_done()
    
    # 启动工作线程
    threads = []
    for _ in range(min(api_nproc, pending_tasks)):  # 线程数不超过任务数
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)
    
    # 创建进度条
    pbar = tqdm(total=lt, desc=f'Infer {model_name}/{dataset_name}, Rank {rank}/{world_size}')
    
    # 先更新已完成的任务数
    already_completed = lt - pending_tasks
    pbar.update(already_completed)
    
    # 收集结果
    completed_count = 0
    while completed_count < pending_tasks:
        try:
            # 从结果队列中取结果
            i, idx, response = result_queue.get(timeout=1)
            
            # 线程安全地更新结果
            with res_lock:
                res[idx] = response
                completed_count += 1
                
                # 更新进度条
                pbar.update(1)
                
                # 每10个结果保存一次
                if completed_count % 10 == 0:
                    dump(res, out_file)
                    
        except queue.Empty:
            # 检查是否所有线程都已完成
            if all(not t.is_alive() for t in threads):
                break
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    pbar.close()

    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(
    model, work_dir, model_name, dataset, verbose=False, api_nproc=4, ignore_failed=False, use_vllm=False
):
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    result_file = osp.join(work_dir, f'{model_name}_{dataset_name}.xlsx')

    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if not ignore_failed:
                results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)

    
    
    print("-------------------a---------------------------")
    
    model = infer_data(
        model=model, work_dir=work_dir, model_name=model_name, dataset=dataset,
        out_file=out_file, verbose=verbose, api_nproc=api_nproc, use_vllm=use_vllm)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = dataset.data
        for x in data['index']:
            assert x in data_all
        data['prediction'] = [str(data_all[x]) for x in data['index']]
        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    if world_size > 1:
        dist.barrier()
    return model
