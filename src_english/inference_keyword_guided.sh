export PYTHONPATH=$PWD;
accelerate launch --main_process_port 1986 --num_processes=1 --gpu_ids=1 --mixed_precision no  inference_keyword_guided.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/english/LAIONFT_PEXEL8_L2_LORA32_SUFFIX_RECRESUME/unet_weight_s50001.pt' \
--output_dir="../generated/tmp_attn_mod_nointerleave" \
--eval_batch_size=1 \
--dbname="mario_eval2" \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/' \
--num_inference_steps=25 \
--lora_rank=32 --seed=7777 --guidance_scale=7.5 --treg_pos=0.3 --treg_neg=0.3


export PYTHONPATH=$PWD;
accelerate launch --main_process_port 1987 --num_processes=7 --gpu_ids=1 --mixed_precision no  inference_keyword_guided.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/english/LAIONFT_PEXEL8_L2_LORA32_SUFFIX_RECRESUME/unet_weight_s50001.pt' \
--output_dir="../generated/tmp_attn_mod_nointerleave2" \
--eval_batch_size=1 \
--dbname="mario_eval2" \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/' \
--num_inference_steps=25 \
--lora_rank=32 --seed=9999 --guidance_scale=7.5 --treg_pos=0.7 --treg_neg=0.7


export PYTHONPATH=$PWD;
accelerate launch --main_process_port 1987 --num_processes=1 --gpu_ids=7 --mixed_precision no  inference_keyword_guided.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/english/LAIONFT_PEXEL8_L2_LORA32_SUFFIX_RECRESUME/unet_weight_s50001.pt' \
--output_dir="../generated/tmp_no_mod2" \
--eval_batch_size=1 \
--dbname="mario_eval2" \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/' \
--num_inference_steps=25 \
--lora_rank=32 --seed=9999 --guidance_scale=7.5 --treg_pos=0 --treg_neg=0  
