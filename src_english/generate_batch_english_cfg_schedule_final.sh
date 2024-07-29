export PYTHONPATH=$PWD;
accelerate launch --main_process_port 9971 --num_processes=3 --gpu_ids=0,1 --mixed_precision no  generate_batch_english_cfg_schedule_final.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/english/LAIONFT_PEXEL8_L2_LORA32_SUFFIX_RECRESUME/unet_weight_s50001.pt' \
--output_dir="../generated/LAIONFT_PEXEL8_L2_LORA32_SUFFIX_RECRESUME_CFG_DEC05_BASE0P5_CONST7_NOSUFFIX" \
--eval_batch_size=15 \
--dbname=mario_eval2 \
--lora_rank=32 \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/' \
--decay_rate=0.5 --base_scale=0.5 --cfg_const=7 \
--seed=7777




















export PYTHONPATH=$PWD;
accelerate launch --main_process_port 9961 --num_processes=1 --gpu_ids=2 --mixed_precision no  generate_batch_english_cfg_schedule_final.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/english/LAIONFT_PEXEL8_L2_LORA32_SUFFIX_RECRESUME/unet_weight_s50001.pt' \
--output_dir="../tmp/cfg_app/seed7777_dec05_base3p5_const4" \
--eval_batch_size=3 \
--dbname=mario_eval2 \
--lora_rank=32 \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/' \
--decay_rate=0.5 --base_scale=3.5 --cfg_const=4 \
--seed=7777 

export PYTHONPATH=$PWD;
accelerate launch --main_process_port 9981 --num_processes=1 --gpu_ids=3 --mixed_precision no  generate_batch_english_cfg_schedule_final.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/english/LAIONFT_PEXEL8_L2_LORA32_SUFFIX_RECRESUME/unet_weight_s50001.pt' \
--output_dir="../tmp/cfg_app/seed7777_dec05_base2p5_const5" \
--eval_batch_size=3 \
--dbname=mario_eval2 \
--lora_rank=32 \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/' \
--decay_rate=0.5 --base_scale=2.5 --cfg_const=5 \
--seed=7777 


export PYTHONPATH=$PWD;
accelerate launch --main_process_port 1241 --num_processes=1 --gpu_ids=4 --mixed_precision no  generate_batch_english_cfg_schedule_final.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/english/LAIONFT_PEXEL8_L2_LORA32_SUFFIX_RECRESUME/unet_weight_s50001.pt' \
--output_dir="../tmp/cfg_app/seed7777_dec05_base1p5_const6" \
--eval_batch_size=3 \
--dbname=mario_eval2 \
--lora_rank=32 \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/' \
--decay_rate=0.5 --base_scale=1.5 --cfg_const=6 \
--seed=7777 


export PYTHONPATH=$PWD;
accelerate launch --main_process_port 7712 --num_processes=1 --gpu_ids=5 --mixed_precision no  generate_batch_english_cfg_schedule_final.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/english/LAIONFT_PEXEL8_L2_LORA32_SUFFIX_RECRESUME/unet_weight_s50001.pt' \
--output_dir="../tmp/cfg_app/seed7777_dec05_base0p5_const7" \
--eval_batch_size=3 \
--dbname=mario_eval2 \
--lora_rank=32 \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/' \
--decay_rate=0.5 --base_scale=0.5 --cfg_const=7 \
--seed=7777 