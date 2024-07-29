import shutil
import cv2
import numpy as np
from PIL import Image
import os
# /data/twkim/diffusion/ocr-dataset/merged_logos_filtered2/templates/
# /data/twkim/diffusion/ocr-dataset/synth_pexel_logos/templates/'
data_root='/data/twkim/diffusion/ocr-dataset/'
dbnames=['merged_logos_filtered2','synth_pexel_logos']
dst_root='/home/jacobwang/code/azure/text_enh/src_mlt2/logo_templates'
os.makedirs(dst_root,exist_ok=True)
for db in dbnames:
    dbroot=os.path.join(data_root,db)
    src_temp_root=os.path.join(dbroot,'templates')

    templates=os.listdir(src_temp_root)
    for template in templates:
        src_temp_path=os.path.join(src_temp_root,template)
        dst_path=os.path.join(dst_root,template.replace('.png','.jpg'))

        if not '.png' in template:
            shutil.copy(src_temp_path,dst_path)
            continue
        image=Image.open(src_temp_path)
        logo_image_np=np.array(image).astype(np.uint8)
        fg_color=np.random.randint(0,256,size=(4,))
        bg_color=np.random.randint(0,256,size=(4,))
        bg_color=bg_color
        fg_color=fg_color.reshape(1,1,4)
        fg_color[-1]=255
        bg_color[-1]=255

        # logo_image_np+=np.array((0,255,0,255)).astype(np.uint8)
        logo_image_np+=fg_color.astype(np.uint8)
        colored_logo_image=Image.fromarray(logo_image_np)
        logo_frame = Image.new("RGBA", image.size, tuple(bg_color.tolist()[:-1]))

        logo_frame.paste(colored_logo_image, (0, 0), image)  
        logo_frame.convert('RGB').save(dst_path, "JPEG")