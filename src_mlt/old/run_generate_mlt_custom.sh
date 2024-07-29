# TRAIN_LOGOS_LORA32_REC010_SYNTH025_Text
export PYTHONPATH=/home/twkim/project/azure/text_enh/src_mlt;
accelerate launch --main_process_port 1121 --num_processes=1 --gpu_ids=7 --mixed_precision no  generate_mlt_custom.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/mlt/TRAIN_MLT_WEAK_EXP14/unet_weight_s5001.pt' \
--output_dir="../generated/mlt_zeroshot_telugu/TRAIN_MLT_WEAK_EXP14_5K_nocfg" \
--eval_batch_size=1 \
--dbname=mlt2k \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
--lora_rank=32 \
--charset_path='ckpt/chartokenizer/char_vocab_mlt_l2r.json' 



export PYTHONPATH=/home/twkim/project/azure/text_enh/src_mlt;
accelerate launch --main_process_port 1121 --num_processes=1 --gpu_ids=6 --mixed_precision no  generate_mlt_custom.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--use_xformers \
--local_rank=0 \
--resume_unet_ckpt_path='../weights/english/LAIONFT_PEXEL8_L2_LORA32_SUFFIX_RECRESUME/unet_weight_s50001.pt' \
--output_dir="../tmp/tmp_zeroshot" \
--eval_batch_size=1 \
--dbname=mlt2k \
--instance_data_root='/data/twkim/diffusion/ocr-dataset/'  \
--lora_rank=32 \
--charset_path='ckpt/chartokenizer/char_vocab_mlt_l2r.json' 


# TRAIN_MLT_FONTFIXED_L2_LORA32_REC0025_SYNTHR07_SYNTHW010_Text
# unet_weight_s5001
# TRAIN_MLT_FONTFIXED_L2_LORA32_REC0025_SYNTHR07_SYNTHW025_Text
# unet_weight_s30001.p