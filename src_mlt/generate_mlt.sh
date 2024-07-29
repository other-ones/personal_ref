
# TRAIN_MLT_ALL_SYNTH09_L1_M1_B8_CORRECT_BALANCE_ADVENG
export PYTHONPATH=/home/twkim/project/azure/ConditionalT2IWithReferenceGuidance/src_mlt;
accelerate launch --main_process_port 7510 --num_processes=3 --gpu_ids=4,5,6 --mixed_precision no  generate_mlt.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/mlt/TRAIN_MLT_ALL_SYNTH09_L1_M1_B8_REC_PRETRAINED_STRICT/unet_weight_s20001.pt' \
--output_dir="../generated/mlt/TRAIN_MLT_ALL_SYNTH09_L1_M1_B8_REC_PRETRAINED_STRICT_s20001" \
--eval_batch_size=15 \
--seed=7777 \
--dbname="seen_mlt_v3" \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
--lora_rank=32 



export PYTHONPATH=/home/twkim/project/azure/ConditionalT2IWithReferenceGuidance/src_mlt;
accelerate launch --main_process_port 7511 --num_processes=2 --gpu_ids=2,3 --mixed_precision no  generate_mlt.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/mlt/TRAIN_MLT_ALL_SYNTH09_L1_M1_B8_REC_PRETRAINED_STRICT/log_05-16-03-26-1715844400-62319538999/unet_weight_s40001.pt' \
--output_dir="../generated/mlt" \
--eval_batch_size=15 \
--dbname="seen_mlt_v3" \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
--lora_rank=32 



export PYTHONPATH=/home/twkim/project/azure/ConditionalT2IWithReferenceGuidance/src_mlt;
accelerate launch --main_process_port 7512 --num_processes=2 --gpu_ids=4,5 --mixed_precision no  generate_mlt.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/mlt/TRAIN_MLT_ALL_SYNTH09_L1_M1_B8_CORRECT_BALANCE_ADVENG/log_05-16-03-26-1715844400-62319538999/unet_weight_s40001.pt' \
--output_dir="../generated/mlt" \
--eval_batch_size=16 \
--dbname="seen_mlt_v3" \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
--lora_rank=32 



export PYTHONPATH=/home/twkim/project/azure/ConditionalT2IWithReferenceGuidance/src_mlt;
accelerate launch --main_process_port 7513 --num_processes=2 --gpu_ids=6,7 --mixed_precision no  generate_mlt.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/mlt/TRAIN_MLT_ALL_SYNTH09_L1_M1_B8_CORRECT_BALANCE_ADVENG/log_05-15-08-28-1715776092-41102646208/unet_weight_s50001.pt' \
--output_dir="../generated/mlt" \
--eval_batch_size=16 \
--dbname="seen_mlt_v3" \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
--lora_rank=32 
