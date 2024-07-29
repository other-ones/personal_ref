
russian_TRAIN_MLT_FONTFIXED_L2_LORA32_REC0025_SYNTHR07_SYNTHW010_Text_30K
# TRAIN_LOGOS_LORA32_REC010_SYNTH025_Text
export PYTHONPATH=/home/twkim/project/azure/text_enh/src_mlt;
accelerate launch --main_process_port 7513 --num_processes=4 --gpu_ids=0,1,2,3 --mixed_precision no  generate_mlt.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/mlt/TRAIN_MLT_ALL_SYNTH09_L1_M1_B8_CORRECT_BALANCE_ADVENG/unet_weight_s40001.pt' \
--output_dir="../generated/mlt_all/TRAIN_MLT_ALL_SYNTH09_L1_M1_B8_CORRECT_BALANCE_ADVENG_40K_NOCFG" \
--eval_batch_size=15 \
--dbname="seen_mlt_v3" \
--target_subset='bengali' \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
--lora_rank=32 



export PYTHONPATH=/home/twkim/project/azure/text_enh/src_mlt;
accelerate launch --main_process_port 1231 --num_processes=4 --gpu_ids=4,5,6,7 --mixed_precision no  generate_mlt.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/mlt/TRAIN_MLT_ALL_SYNTH09_L1_M1_B8_REC_PRETRAINED_STRICT/unet_weight_s20001.pt' \
--output_dir="../generated/mlt_all/TRAIN_MLT_ALL_SYNTH09_L1_M1_B8_REC_PRETRAINED_STRICT_20K_NOCFG" \
--eval_batch_size=15 \
--dbname="seen_mlt_v3" \
--target_subset='bengali' \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
--lora_rank=32 

export PYTHONPATH=/home/twkim/project/azure/text_enh/src_mlt;
accelerate launch --main_process_port 7513 --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 --mixed_precision no  generate_mlt.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/mlt_zeroshot/TRAIN_MLT_NOGREEK_SYNTH09_L2_M1_B7/unet_weight_s25001.pt' \
--output_dir="../generated/mlt_zeroshot_greek/TRAIN_MLT_NOGREEK_SYNTH09_L2_M1_B7_25K_NOCFG" \
--eval_batch_size=15 \
--dbname="seen_mlt_v3" \
--target_subset='greek' \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
--lora_rank=32 

export PYTHONPATH=/home/twkim/project/azure/text_enh/src_mlt;
accelerate launch --main_process_port 7513 --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 --mixed_precision no  generate_mlt.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/mlt_zeroshot/TRAIN_MLT_ZEROSHOT_NORUS_SYNTH09_L2_M2/unet_weight_s30001.pt' \
--output_dir="../generated/mlt_zeroshot_russian/TRAIN_MLT_ZEROSHOT_NORUS_SYNTH09_L2_M1_B7_RE2_20K_CFG" \
--eval_batch_size=15 \
--dbname="seen_mlt_v3" \
--target_subset='russian' \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
--lora_rank=32 
