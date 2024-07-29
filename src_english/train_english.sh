accelerate launch --main_process_port 1818 --num_processes=2 --gpu_ids=3,7 --mixed_precision no  train_english.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--instance_data_list="pexel5" \
--instance_data_root="/data/twkim/diffusion/ocr-dataset/" \
--output_dir="../saved_models/tmp_english" \
--train_text_encoder \
--train_batch_size=1 \
--gradient_accumulation_steps=1 \
--learning_rate=1e-4 \
--lr_scheduler="constant" \
--lr_warmup_steps=100 \
--max_train_steps=200000 \
--save_steps=500 \
--unfreeze_lora_step=1500 \
--local_rank=0 \
--root_path='/data/dataset' \
--laion_batch_size=2 \
--mario_batch_size=2 \
--rec_loss_weight=0.05 \
--visualize_steps=100

accelerate launch --main_process_port 1818 --num_processes=2 --gpu_ids=0,1 --mixed_precision no  train_english.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--instance_data_list="pexel5" \
--instance_data_root="/data/twkim/diffusion/ocr-dataset/" \
--output_dir="../saved_models/tmp_english" \
--train_text_encoder \
--train_batch_size=2 \
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
--rec_loss_weight=0.05 \
--visualize_steps=1 \
--criterion='focal_loss'