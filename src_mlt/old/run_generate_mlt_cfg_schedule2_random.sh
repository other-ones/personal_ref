
russian_TRAIN_MLT_FONTFIXED_L2_LORA32_REC0025_SYNTHR07_SYNTHW010_Text_30K
# TRAIN_LOGOS_LORA32_REC010_SYNTH025_Text
export PYTHONPATH=/home/twkim/project/azure/text_enh/src_mlt;
accelerate launch --main_process_port 7514 --num_processes=7 --gpu_ids=0,1,2,3,4,5,6 --mixed_precision no  generate_mlt_cfg_schedule2_random.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/mlt/TRAIN_MLT_WEAK_EXP14/unet_weight_s5001.pt' \
--output_dir="../generated/MLT_EVAL_ENGLISH_CFG_WEAK4_DEC02_BASE3P5_CONST4" \
--eval_batch_size=10 \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
--lora_rank=32 \
--base_scale=3.5 --decay_rate=0.2 --cfg_const=4 --coord_jitter=1 \
--target_subset=english
