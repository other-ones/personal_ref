# Best Config: dec02 base3.5 const 4
export PYTHONPATH=/home/twkim/project/azure/refdiffuser/src_mlt;
accelerate launch --main_process_port 7514 --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 --mixed_precision no  generate_mlt_cfg_schedule2.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/mlt_zeroshot/TRAIN_MLT_ZEROSHOT_NORUS_SYNTH09_L2_M1_B7_RE2/unet_weight_s32501.pt' \
--output_dir="../generated/mlt_zeroshot_russian/TRAIN_MLT_ZEROSHOT_NORUS_SYNTH09_L2_M1_B7_RE2_32P5K_CFG" \
--eval_batch_size=15 \
--dbname="unseen_mlt_new" \
--target_language='armenian' \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
--lora_rank=32 \
--decay_rate=0.2 --base_scale=3.5 --cfg_const=4 --coord_jitter=0


export PYTHONPATH=/home/twkim/project/azure/text_enh/src_mlt;
accelerate launch --main_process_port 7513 --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 --mixed_precision no  generate_mlt_cfg_schedule2.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/mlt_zeroshot/TRAIN_MLT_ZEROSHOT_LATIN_SYNTH09_L2_M1_B7_Text/unet_weight_s15001.pt' \
--output_dir="../generated/mlt_zeroshot_russian/TRAIN_MLT_ZEROSHOT_LATIN_SYNTH09_L2_M1_B7_Text_15K_CFG" \
--eval_batch_size=15 \
--dbname="seen_mlt_v3" \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
--target_subset='russian' \
--lora_rank=32 \
--base_scale=3.5 --decay_rate=0.2 --cfg_const=4 --coord_jitter=1
