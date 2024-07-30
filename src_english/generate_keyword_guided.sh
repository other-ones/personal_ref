export PYTHONPATH=$PWD;
accelerate launch --main_process_port 1986 --num_processes=1 --gpu_ids=7 --mixed_precision no  generate_keyword_guided.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/english/LAIONFT_PEXEL8_L2_LORA32_SUFFIX_RECRESUME/unet_weight_s50001.pt' \
--output_dir="../generated/tmp_single" \
--eval_batch_size=1 \
--dbname="mario_eval2" \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/' \
--num_inference_steps=25 \
--lora_rank=32 --seed=7777 --guidance_scale=7.5 --treg_pos=0.5 --treg_neg=0  \
--include_suffix --debug --target_file="LAIONEval4000_1001"


export PYTHONPATH=$PWD;
accelerate launch --main_process_port 1985 --num_processes=3 --gpu_ids=4,5,6 --mixed_precision no  generate_rd_keyword_guided.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/english/LAIONFT_PEXEL8_L2_LORA32_SUFFIX_RECRESUME/unet_weight_s50001.pt' \
--output_dir="../generated/LAIONFT_PEXEL8_L2_LORA32_SUFFIX_RECRESUME_s50K_NOSKIP_NOSUFFIX_attnmod_pos04_neg0_seed7777" \
--eval_batch_size=1 \
--dbname="mario_eval2" \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/' \
--num_inference_steps=100 \
--lora_rank=32 --seed=7777 --guidance_scale=7.5 --treg_pos=0.3 --treg_neg=0 

export PYTHONPATH=$PWD;
accelerate launch --main_process_port 1986 --num_processes=3 --gpu_ids=4,5,6 --mixed_precision no  generate_rd_keyword_guided.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/english/LAIONFT_PEXEL8_L2_LORA32_SUFFIX_RECRESUME/unet_weight_s50001.pt' \
--output_dir="../generated/LAIONFT_PEXEL8_L2_LORA32_SUFFIX_RECRESUME_s50K_NOSKIP_attnmod_pos02_neg02_suffix_seed7777" \
--eval_batch_size=1 \
--dbname="mario_eval2" \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/' \
--num_inference_steps=100 \
--lora_rank=32 --seed=7777 --guidance_scale=7.5 --treg_pos=0.2 --treg_neg=0.2 \
--include_suffix

export PYTHONPATH=$PWD;
accelerate launch --main_process_port 1986 --num_processes=1 --gpu_ids=7 --mixed_precision no  generate_rd_keyword_guided.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/english/LAIONFT_PEXEL8_L2_LORA32_SUFFIX_RECRESUME/unet_weight_s50001.pt' \
--output_dir="../generated/tmp_attn_mod04_pos_batch5_interleave" \
--eval_batch_size=5 \
--dbname="mario_eval2" \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/' \
--num_inference_steps=50 \
--lora_rank=32 --seed=7777 --guidance_scale=7.5 --treg=0.4

export PYTHONPATH=$PWD;
accelerate launch --main_process_port 1694 --num_processes=3 --gpu_ids=1,2,7 --mixed_precision no  generate_rd_keyword_guided.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/english/LAIONFT_PEXEL8_L2_LORA32_SUFFIX_RECRESUME/unet_weight_s50001.pt' \
--output_dir="../generated/tmp_run2" \
--eval_batch_size=5 \
--dbname="mario_eval2" \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/' \
--num_inference_steps=50 \
--lora_rank=32 --num_intervention_steps=20 --seed=7777 --syngen_step_size=20 --guidance_scale=7.5 \
--guidance_strength=1 --vis_strength=100 --do_attn_mod=1 --mask_bg=0.4 --mask_fg=1.5