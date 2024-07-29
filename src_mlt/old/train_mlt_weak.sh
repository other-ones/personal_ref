export PYTHONPATH=/home/twkim/project/azure/refdiffuser/src_mlt;
accelerate launch --main_process_port 9912 --num_processes=2 --gpu_ids=6,7 --mixed_precision no  train_mlt_weak.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--instance_data_root="/data/twkim/diffusion/ocr-dataset/" \
--output_dir="../saved_models/tmp_mlt_weak" \
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
--laion_batch_size=0 \
--mario_batch_size=0 \
--charset_path='ckpt/chartokenizer/char_vocab_mlt_l2r.json' \
--visualize_steps 100 \
--synth_unet_loss_weight=0.01  \
--synth_ratio=0.5 \
--target_languages english-thai-greek-russian-german-italian-french-hindi-bengali --contrast_mode='none' \
--instance_data_list=icdar2019_matched-synth_pexel_thai3-synth_pexel_greek3-synth_pexel_russian3-synth_pexel_german3-synth_pexel_italian3-synth_pexel_french3-synth_pexel_hindi3-synth_pexel_bengali3 \
--lora_rank=32 \
--debug

# english-thai-greek-russian-german-italian-french-hindi-bengali