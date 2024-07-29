export PYTHONPATH=$PWD;
accelerate launch --main_process_port 1986 --num_processes=2 --gpu_ids=4,5 --mixed_precision no  generate_keyword_guided_mlt.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/mlt/TRAIN_MLT_ALL_SYNTH09_L1_M1_B8_CORRECT_BALANCE_ADVENG/log_05-15-08-28-1715776092-41102646208/unet_weight_s20001.pt' \
--output_dir="../generated/tmp" \
--eval_batch_size=1 \
--dbname="seen_mlt_v3" \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/' \
--num_inference_steps=25 \
--lora_rank=32 --seed=7777 --guidance_scale=7.5 --treg_pos=0.4 --treg_neg=0 