import time
import numpy as np
import os
import subprocess as sp



def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values




ports=np.arange(5000,6000)
stats=get_gpu_memory()
for stat_idx,stat in enumerate(stats):
    if stat>2e4:
        break
device_idx=stat_idx
idx=0
exps=[
    'TRAIN_MLT_ALL_SYNTH09_L1_M1_B8_CORRECT_BALANCE_ADVENG'
]


unet_dir='../weights/mlt/TRAIN_MLT_ALL_SYNTH09_L1_M1_B8_REC_PRETRAINED_STRICT/log_05-16-03-26-1715844400-62319538999'
steps=[50001,30001,40001]
num_processes=3
for step in steps:
    log_dir='logs/generate/mlt'
    os.makedirs(log_dir,exist_ok=True)    
    resume_unet_path=os.path.join(unet_dir,'unet_weight_s{}.pt'.format(step))
    if not os.path.exists(resume_unet_path):
        print(resume_unet_path,'does not exist')
        continue
    exp_name=unet_dir.split('/')[-2]
    exp_name+='_s{}'.format(step)
    output_dir=os.path.join('../generated/mlt')
    exp_path=os.path.join(output_dir,exp_name)
    if os.path.exists(exp_path):
        print(exp_name,'exists')
        continue
    while True:
        stats=get_gpu_memory()
        stats=np.array(stats)
        # stat=stats[stat_idx%len(stats)]
        if np.sum(stats>2e4)>=num_processes:
            available_devices=','.join(np.where(stats>2e4)[0][:num_processes].astype(str))
            break
        print('sleep waiting for {}'.format(exp_name))
        time.sleep(30)
    print(exp_name,'GPU:{}'.format(available_devices))
    log_path=os.path.join(log_dir,exp_name+'.out')
    # 
    # TRAIN_MLT_ALL_SYNTH09_L1_M1_B8_CORRECT_BALANCE_ADVENG
    # export PYTHONPATH=/home/twkim/project/azure/ConditionalT2IWithReferenceGuidance/src_mlt;
    # accelerate launch --main_process_port 7510 --num_processes=2 --gpu_ids=4,5,6 --mixed_precision no  generate_mlt.py \
    # --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
    # --use_xformers \
    # --local_rank=0 \
    # --resume_unet_ckpt_path='../weights/mlt/TRAIN_MLT_ALL_SYNTH09_L1_M1_B8_REC_PRETRAINED_STRICT/unet_weight_s20001.pt' \
    # --output_dir="../generated/mlt/TRAIN_MLT_ALL_SYNTH09_L1_M1_B8_REC_PRETRAINED_STRICT_s20K" \
    # --eval_batch_size=16 \
    # --seed=7777 \
    # --dbname="seen_mlt_v3" \
    # --instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
    # --lora_rank=32 

    
    command='export CUDA_VISIBLE_DEVICES={};'.format(device_idx)
    command+='export PYTHONPATH=/home/twkim/project/azure/ConditionalT2IWithReferenceGuidance/src_mlt;'
    command+='accelerate launch --main_process_port {} --num_processes={} --gpu_ids={} --mixed_precision no generate_mlt.py \\\n'.format(ports[idx],num_processes,available_devices)
    command+='--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \\\n'
    command+='--use_xformers \\\n'
    command+='--local_rank=0 \\\n'
    command+='--resume_unet_ckpt_path="{}/unet_weight_s{}.pt" \\\n'.format(unet_dir,step)
    command+='--output_dir="{}" \\\n'.format(exp_path)
    command+='--eval_batch_size=15 \\\n'
    command+='--seed=7777 \\\n'
    command+='--dbname="seen_mlt_v3" \\\n'
    command+='--instance_data_root="/data/twkim/diffusion/ocr-dataset/" \\\n'
    command+='--lora_rank=32 > {} 2>&1 &'.format(log_path)
    os.system(command)
    print('STARTED')
    idx+=1
    time.sleep(60)

    


