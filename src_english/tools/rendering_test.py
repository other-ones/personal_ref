# from utils import generate_spatial_rendering_or_logo
import numpy as np
from PIL import Image, ImageDraw,ImageFont
import os



def generate_spatial_rendering(width, height, words,dst_coords=None,diversify_font=False,font_path=None):
    image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    new_dst_coords=[]
    new_words=[]
    for i in range(len(words)):
        # color=tuple(np.random.randint(0,255,size=(3,)))
        color=(0,0,0)
        coords=dst_coords[i]
        word = words[i]
        assert len(word)
        x1, y1, x2, y2 = np.array(coords).astype(np.int32)#np.min(xs),np.min(ys),np.max(xs),np.max(ys)
        region_width = x2 - x1
        region_height = y2 - y1
        min_side=min(region_width,region_height)
        new_words.append(word)
        new_dst_coords.append(coords)
        if region_height>(region_width*2):
            font_size = int(max(region_width, region_height) / len(word))
            font_size=font_size*0.75
            font_size=int(font_size)
            font_size=max(1,font_size)
            font_size=min(min_side,font_size)
            font_size=max(5,font_size)
            font = ImageFont.truetype(font_path, font_size)
            text_width, text_height = draw.textsize(word, font=font)
            text_x = x1 + (region_width - text_width) // 2
            text_y = y1 + (region_height - text_height) // 2
            draw.text((text_x, text_y), word, font=font, fill=0,spacing=1)
        else: 
            font_size = int(max(region_width, region_height)*1.4 / len(word))
            font_size=int(font_size)
            font_size=min(min_side,font_size)
            font_size=max(35,font_size)
            font_size=min(65,font_size)
            print(font_size,'font_size')
            font = ImageFont.truetype(font_path, font_size)
            text_width, text_height = draw.textsize(word, font=font)
            text_x = x1 + (region_width - text_width) // 2
            text_y = y1 + (region_height - text_height) // 2
            draw.text((text_x, text_y), word, font=font, fill=color)
    return image,np.array(new_dst_coords),new_words
dst_root='english_font_viz'
os.makedirs(dst_root,exist_ok=True)
root='/home/twkim/project/azure/refdiffuser/src_english/fonts'
fonts=os.listdir(root)
glyph_texts = ["red", "apple"]

# vertical
coords1=np.array([
    [200, 200, 300, 250],
[150, 270, 350, 320]
])

# horizontal
coords2=np.array([
    [120,210,220,260],
[200, 220, 370, 270]
])

coords=coords2.astype(np.int32).tolist()
for font in fonts:
    font_name=font.split('.')[0]
    font_path=os.path.join(root,font)
    rendered_whole_image,_,_=generate_spatial_rendering(width=512,height=512,words=glyph_texts,
                                                                     dst_coords=coords,
                                                                     font_path=font_path)
    rendered_whole_image.save(dst_root+'/{}.jpg'.format(font_name))