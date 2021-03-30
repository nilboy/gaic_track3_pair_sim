import subprocess
import time
import os
import logging

import threading
from multiprocessing import Queue

logging.basicConfig()
logger = logging.getLogger('pipeline')
fh = logging.FileHandler('pipeline.log')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(ch)
#####
num_gpus = 4
kfold_num = 8
#####

class Timer(object):
    def __init__(self):
        self.start_time = time.time()

    def get_current_time(self):
        return (time.time() - self.start_time) / 3600

class GPUManager(object):
    def __init__(self, num_gpus=4):
        self.gpu_queue = Queue()
        for device_id in range(num_gpus):
            self.gpu_queue.put(device_id)

    def require(self, timeout=60*5):
        try:
            return self.gpu_queue.get(timeout=timeout)
        except:
            return None

    def add_gpu(self, gpu_id):
        self.gpu_queue.put(gpu_id)

timer = Timer()
gpu_manager = GPUManager(num_gpus=num_gpus)

def run_gpu_model(cmd, log_file=None, max_time=80):
    if log_file:
        cmd = f"nohup {cmd} > {log_file}"
    while True:
        if timer.get_current_time() >= max_time:
            logger.warning(f'{cmd} 超时， 不执行')
            return
        gpu_id = gpu_manager.require()
        if gpu_id is not None:
            try:
                run_cmd = f"export CUDA_VISIBLE_DEVICES={gpu_id} && {cmd}"
                logger.info(f"{run_cmd} 开始时间: {timer.get_current_time()}")
                os.system(run_cmd)
                logger.info(f"{run_cmd} 结束时间: {timer.get_current_time()}")
            except:
                logger.warning(f'{cmd} failed')
            gpu_manager.add_gpu(gpu_id)
            break

# 数据处理和模型转换
logger.info(f'数据处理开始: {timer.get_current_time()} 小时')
process_data_s1 = subprocess.Popen(
    "bash data_processor/process_data_s1.sh", shell=True
)
process_data_s1.wait()
processes = [
    subprocess.Popen(
        "bash data_processor/process_data_s2.sh", shell=True
    ),
    subprocess.Popen(
        "bash model_processor/process_model.sh", shell=True
    )
]
for process in processes:
    process.wait()
logger.info(f'数据处理结束: {timer.get_current_time()} 小时')

def train_model(model_name):
    # 预训练
    logger.info(f"预训练 {model_name} 开始: {timer.get_current_time()}")
    run_gpu_model(f'bash train/train_mlm.sh {model_name}', log_file=f"{model_name}_mlm.log", max_time=70)
    logger.info(f"预训练 {model_name} 结束: {timer.get_current_time()}")

# 预训练
model_processes = []
from multiprocessing import Process
for model_name in ['bert-base-fgm', 'bert-large', 'macbert-base-fgm', 'macbert-large',
                   'bert-base', 'macbert-base', 'bert-base2', 'macbert-base2']:
    p = Process(target=train_model, args=(model_name,))
    p.start()
    time.sleep(60)
    logger.info(f'创建模型训练 {model_name}...')
    model_processes.append(p)
for p in model_processes:
    p.join()
logger.info(f'pretrain 结束: {timer.get_current_time()} 小时')
# 模型转换
processes = [
    subprocess.Popen(
        f"python model_processor/build_ensemble_model_b.py", shell=True
    )
]
for process in processes:
    process.wait()
logger.info(f'ensemble模型构造结束: {timer.get_current_time()} 小时')
# 训练分类模型
logger.info(f'训练分类模型   : {timer.get_current_time()} 小时')
cmd = f"export CUDA_VISIBLE_DEVICES=0,1,2,3 && python train/train_classification_b.py --max_run_time={72 - timer.get_current_time()}"
logger.info(cmd)
os.system(cmd)
logger.info(f'训练分类模型结束: {timer.get_current_time()} 小时')
