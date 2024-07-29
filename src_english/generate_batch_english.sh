export PYTHONPATH=/home/twkim/project/azure/ConditionalT2IWithReferenceGuidance/src_english;
accelerate launch --main_process_port 9932 --num_processes=0 --gpu_ids=0 --mixed_precision no  generate_batch_english.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--local_rank=0 \
--resume_unet_ckpt_path='/home/twkim/project/azure/refdiffuser/weights/english/LAIONFT_PEXEL8_L2_LORA32_SUFFIX_RECRESUME/unet_weight_s50001.pt' \
--output_dir="../generated/tmp" \
--eval_batch_size=15 \
--dbname=mario_eval2 \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
--lora_rank=32 --seed=7777 \
--guidance_scale=7.5 --uniform_layout



export PYTHONPATH=/home/twkim/project/azure/ConditionalT2IWithReferenceGuidance/src_english;
accelerate launch --main_process_port 9932 --num_processes=2 --gpu_ids=0,1 --mixed_precision no  generate_batch_english.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--local_rank=0 \
--resume_unet_ckpt_path='/home/twkim/project/azure/refdiffuser/weights/english/LAIONFT_PEXEL8_L2_LORA32_SUFFIX_RECRESUME/unet_weight_s50001.pt' \
--output_dir="../generated/LAIONFT_PEXEL8_L2_LORA32_SUFFIX_RECRESUME_s50K_NOSKIP_seed1234_suffix" \
--eval_batch_size=15 \
--dbname=mario_eval2 \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
--lora_rank=32 --seed=1234 \
--guidance_scale=7.5 \
--include_suffix

export PYTHONPATH=/home/twkim/project/azure/ConditionalT2IWithReferenceGuidance/src_english;
accelerate launch --main_process_port 9932 --num_processes=6 --gpu_ids=0,1,2,3 --mixed_precision no  generate_batch_english.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--local_rank=0 \
--resume_unet_ckpt_path='/home/twkim/project/azure/refdiffuser/weights/english/LAIONFT_PEXEL8_L2_LORA32_SUFFIX_RECRESUME/unet_weight_s50001.pt' \
--output_dir="../generated/tmp" \
--eval_batch_size=15 \
--dbname=mario_eval2 \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
--lora_rank=32 --seed=7777 \
--guidance_scale=7.5