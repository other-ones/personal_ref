import re
import os
import numpy as np
import shutil
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, AutoProcessor
caption_path='/data/twkim/ocr-dataset/mario_eval2/mario_eval2_blip_caption.txt'
lroot='/data/twkim/ocr-dataset/mario_eval2/layouts'
model_name="stabilityai/stable-diffusion-2-1"
tokenizer = CLIPTokenizer.from_pretrained(
        model_name,
        subfolder="tokenizer",
        revision=None,
    )
caption="These dogs are very sociable lovable and can fit in a variety of different lifestyles This is all true of 'the toy Poodle' whose number one desire is to please their owner So if you have a this 'toy' dog at home check out these 'Toy Poodle names' for inspiration toypoodle toypoodlenames 'poodle' Small 'Poodle Best' Dog 'Names Toy' Dog Breeds Dog Toys Teddy Bear Number Fit Check Dogs"
input_ids=tokenizer(
            caption,
            padding="do_not_pad",
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).input_ids
print(len(input_ids))
input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
decodeds=[]
for tid in input_ids:
    decoded=tokenizer.decode(tid)
    print(decoded,'decoded')
    decodeds.append(decoded)
print(''.join(decodeds),len(decodeds))
