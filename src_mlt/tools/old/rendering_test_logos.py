from PIL import Image
import os
from utils import generate_spatial_rendering_or_logo
from utils import generate_spatial_rendering
from utils import visualize_box
import numpy as np
glyph_texts = ["walmart"]
# glyph_texts = ["walmart"]
coords=np.array([[110, 160,390, 265]])
inference_lang_list=['logo']
inference_scripts_list=['logo']
logo_templates=[]
instance_data_root='/data/twkim/diffusion/ocr-dataset'
for logo in glyph_texts:
    if os.path.exists(os.path.join(instance_data_root, 'merged_logos2','templates', logo+'.png')):
        logo_path = os.path.join(instance_data_root, 'merged_logos2','templates', logo+'.png')
    elif os.path.exists(os.path.join(instance_data_root, 'merged_logos2','templates', logo+'.jpg')):
        logo_path = os.path.join(instance_data_root, 'merged_logos2','templates', logo+'.jpg')
    logo_image=Image.open(logo_path).convert('RGB')
    logo_templates.append(logo_image)
    
rendered_whole_images,_,_=generate_spatial_rendering_or_logo(width=512,height=512,words=glyph_texts,dst_coords=coords,
                                                                                diversify_font=False,
                                                                                lang_list=inference_lang_list,logo_templates=logo_templates)
rendered_whole_images=visualize_box(rendered_whole_images,coords)
rendered_whole_images.save('rendered_whole_images.jpg')