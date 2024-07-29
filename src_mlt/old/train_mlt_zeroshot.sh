export PYTHONPATH=/home/twkim/project/azure/refdiffuser/src_mlt;
accelerate launch --main_process_port 9912 --num_processes=2 --gpu_ids=0,1 --mixed_precision no  train_mlt_zeroshot.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--instance_data_root="/data/twkim/diffusion/ocr-dataset/" \
--output_dir="../saved_models/tmp_all" \
--train_batch_size=5 \
--gradient_accumulation_steps=1 \
--learning_rate=1e-4 \
--lr_scheduler="constant" \
--lr_warmup_steps=100 \
--max_train_steps=200000 \
--save_steps=500 \
--unfreeze_lora_step=1500 \
--local_rank=0 \
--root_path='/data/dataset' \
--laion_batch_size=0 \
--mario_batch_size=0 \
--visualize_steps 100 \
--synth_unet_loss_weight=0.01  \
--synth_ratio=0.5 \
--charset_path='ckpt/chartokenizer/char_vocab_mlt_l2r.json' \
--lora_rank=32 \
--debug \
--mlt_config='all' \
--mario_prob=1.0


# Russian Latin
export PYTHONPATH=/home/twkim/project/azure/refdiffuser/src_mlt;
accelerate launch --main_process_port 9912 --num_processes=2 --gpu_ids=0,1 --mixed_precision no  train_mlt_zeroshot.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--instance_data_root="/data/twkim/diffusion/ocr-dataset/" \
--output_dir="../saved_models/tmp_all" \
--train_batch_size=5 \
--gradient_accumulation_steps=1 \
--learning_rate=1e-4 \
--lr_scheduler="constant" \
--lr_warmup_steps=100 \
--max_train_steps=200000 \
--save_steps=500 \
--unfreeze_lora_step=1500 \
--local_rank=0 \
--root_path='/data/dataset' \
--laion_batch_size=0 \
--mario_batch_size=0 \
--visualize_steps 100 \
--synth_unet_loss_weight=0.01  \
--synth_ratio=0.5 \
--charset_path='ckpt/chartokenizer/char_vocab_mlt_all.json' \
--lora_rank=32 \
--debug \
--mlt_config='all'

# Russian rest
export PYTHONPATH=/home/twkim/project/azure/refdiffuser/src_mlt;
accelerate launch --main_process_port 9913 --num_processes=2 --gpu_ids=0,1 --mixed_precision no  train_mlt_zeroshot.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--instance_data_root="/data/twkim/diffusion/ocr-dataset/" \
--output_dir="../saved_models/tmp_mlt_nogreek" \
--train_batch_size=5 \
--gradient_accumulation_steps=1 \
--learning_rate=1e-4 \
--lr_scheduler="constant" \
--lr_warmup_steps=100 \
--max_train_steps=200000 \
--save_steps=500 \
--unfreeze_lora_step=1500 \
--local_rank=0 \
--root_path='/data/dataset' \
--laion_batch_size=1 \
--mario_batch_size=1 \
--visualize_steps 100 \
--synth_unet_loss_weight=0.01  \
--synth_ratio=0.5 \
--charset_path='ckpt/chartokenizer/char_vocab_mlt_no_greek.json' \
--resume_recnet_ckpt_path='../weights/english/TRAIN_ENGLISH_LORA32_LAION2/recnet_weight_s100001.pt' \
--lora_rank=32 \
--debug --mlt_config='no_greek'
