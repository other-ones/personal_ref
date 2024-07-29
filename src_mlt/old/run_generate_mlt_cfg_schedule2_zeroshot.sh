# Best Config: dec02 base3.5 const 4
export PYTHONPATH=/home/twkim/project/azure/refdiffuser/src_mlt;
accelerate launch --main_process_port 7514 --num_processes=4 --gpu_ids=0,1,2,3 --mixed_precision no  generate_mlt_cfg_schedule2.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/mlt_zeroshot/TRAIN_MLT_ZEROSHOT_NORUS_SYNTH09_L2_M1_B7_RE2/unet_weight_s50001.pt' \
--output_dir="../generated/mlt_zeroshot_russian/TRAIN_MLT_ZEROSHOT_NORUS_SYNTH09_L2_M1_B7_RE2_50K_CFG" \
--eval_batch_size=15 \
--dbname="seen_mlt_v3" \
--target_subset='russian' \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
--lora_rank=32 \
--decay_rate=0.2 --base_scale=3.5 --cfg_const=4 --coord_jitter=0





export PYTHONPATH=/home/twkim/project/azure/refdiffuser/src_mlt;
accelerate launch --main_process_port 7513 --num_processes=4 --gpu_ids=4,5,6,7 --mixed_precision no  generate_mlt_cfg_schedule2.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/mlt_zeroshot/TRAIN_MLT_ZEROSHOT_NORUS_SYNTH09_L2_M1_B7_RE2/unet_weight_s55001.pt' \
--output_dir="../generated/mlt_zeroshot_russian/TRAIN_MLT_ZEROSHOT_NORUS_SYNTH09_L2_M1_B7_RE2_55K_CFG" \
--eval_batch_size=15 \
--dbname="seen_mlt_v3" \
--target_subset='russian' \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
--lora_rank=32 \
--decay_rate=0.2 --base_scale=3.5 --cfg_const=4 --coord_jitter=0




export PYTHONPATH=/home/twkim/project/azure/refdiffuser/src_mlt;
accelerate launch --main_process_port 7514 --num_processes=4 --gpu_ids=4,5,6,7 --mixed_precision no  generate_mlt_cfg_schedule2.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/mlt_zeroshot/TRAIN_MLT_ALL_SYNTH09_L2_M1_B7_RE2_Text/unet_weight_s12501.pt' \
--output_dir="../generated/mlt_zeroshot_russian/TRAIN_MLT_ALL_SYNTH09_L2_M1_B7_RE2_Text_12P5K_CFG" \
--eval_batch_size=15 \
--dbname="seen_mlt_v3" \
--target_subset='russian' \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
--lora_rank=32 \
--decay_rate=0.2 --base_scale=3.5 --cfg_const=4 --coord_jitter=0



export PYTHONPATH=/home/twkim/project/azure/refdiffuser/src_mlt;
accelerate launch --main_process_port 7514 --num_processes=4 --gpu_ids=0,1,2,3 --mixed_precision no  generate_mlt_cfg_schedule2.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/mlt_zeroshot/TRAIN_MLT_NOGREEK_SYNTH09_L2_M1_B7/unet_weight_s15001.pt' \
--output_dir="../generated/mlt_zeroshot_greek/TRAIN_MLT_NOGREEK_SYNTH09_L2_M1_B7_15K_CFG" \
--eval_batch_size=15 \
--dbname="seen_mlt_v3" \
--target_subset='greek' \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
--lora_rank=32 \
--decay_rate=0.2 --base_scale=3.5 --cfg_const=4 --coord_jitter=0


export PYTHONPATH=/home/twkim/project/azure/refdiffuser/src_mlt;
accelerate launch --main_process_port 7513 --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 --mixed_precision no  generate_mlt_cfg_schedule2.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/mlt_zeroshot/TRAIN_MLT_NOGREEK_SYNTH09_L2_M1_B7/unet_weight_s25001.pt' \
--output_dir="../generated/mlt_zeroshot_greek/TRAIN_MLT_NOGREEK_SYNTH09_L2_M1_B7_25K_CFG" \
--eval_batch_size=15 \
--dbname="seen_mlt_v3" \
--target_subset='greek' \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
--lora_rank=32 \
--decay_rate=0.2 --base_scale=3.5 --cfg_const=4 --coord_jitter=0






















export PYTHONPATH=/home/twkim/project/azure/refdiffuser/src_mlt;
accelerate launch --main_process_port 9378 --num_processes=4 --gpu_ids=4,5,6,7 --mixed_precision no  generate_mlt_cfg_schedule2.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/mlt_zeroshot/TRAIN_MLT_ZEROSHOT_NORUS_SYNTH09_L2_M1_B7_RE/unet_weight_s10001.pt' \
--output_dir="../generated/mlt_zeroshot_russian/TRAIN_MLT_ZEROSHOT_NORUS_SYNTH09_L2_M1_B7_RE_10K_CFG" \
--eval_batch_size=15 \
--dbname="seen_mlt_v3" \
--target_subset='russian' \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
--lora_rank=32 \
--decay_rate=0.2 --base_scale=3.5 --cfg_const=4 --coord_jitter=0

export PYTHONPATH=/home/twkim/project/azure/refdiffuser/src_mlt;
accelerate launch --main_process_port 7514 --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 --mixed_precision no  generate_mlt_cfg_schedule2.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/english/LAIONFT_PEXEL8_L2_LORA32_SUFFIX_RECRESUME/unet_weight_s50001.pt' \
--output_dir="../generated/mlt_zeroshot_english/LAIONFT_PEXEL8_L2_LORA32_SUFFIX_RECRESUME_CFG" \
--eval_batch_size=15 \
--dbname="seen_mlt_v3" \
--target_subset='russian-greek' \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
--lora_rank=32 \
--decay_rate=0.2 --base_scale=3.5 --cfg_const=4 --coord_jitter=0

export PYTHONPATH=/home/twkim/project/azure/refdiffuser/src_mlt;
accelerate launch --main_process_port 7515 --num_processes=4 --gpu_ids=4,5,6,7 --mixed_precision no  generate_mlt_cfg_schedule2.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/mlt_zeroshot/TRAIN_MLT_ZEROSHOT_LATIN_SYNTH09_L2_M2_FAILED/unet_weight_s5001.pt' \
--output_dir="../generated/mlt_zeroshot_greek/TRAIN_MLT_ZEROSHOT_LATIN_SYNTH09_L2_M2_FAILED_5K_CFG" \
--eval_batch_size=1 \
--dbname="seen_mlt_v3" \
--target_subset='greek' \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
--lora_rank=32 \
--decay_rate=0.2 --base_scale=3.5 --cfg_const=4 --coord_jitter=0