import numpy as np
from PIL import Image
import torch
import cv2
import os
def visualize_box(image,boxes,chars=None,thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.6
    image=np.array(image)
    if chars is not None:
        for box,char in zip(boxes,chars):
            box=(box.detach().cpu().numpy()*2).astype(np.int32)
            x0,y0,x1,y1=box
            box=(box).reshape(-1,2).astype(np.int32)
            point1=box[0]
            point2=box[1]
            image=cv2.rectangle(image,tuple(point1),tuple(point2),color=(0,255,0),thickness=1)   
            image = cv2.putText(image, char, (x0,y1+5), font, fontScale, (255,0,0), 1, cv2.LINE_AA)
    else:
        for box in boxes:
            print(np.array(box).shape,'np.array(box).shape')
            if torch.is_tensor(box):
                box=(box.detach().cpu().numpy().astype(np.int32)).reshape(-1,2)
            else:
                box=np.array(box).astype(np.int32).reshape(-1,2)
            point1=box[0]
            point2=box[1]
            image=cv2.rectangle(image,tuple(point1),tuple(point2),color=(0,255,0),thickness=thickness)
    image=Image.fromarray(image)
    return image


def visualize_polygon(image,coords):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.6
    image=np.array(image)
    color=(0,255,0)
    thickness=10
    coords=coords.reshape(-1,2)
    cv2.line(image, coords[0], coords[1], color, thickness)
    cv2.line(image, coords[1], coords[2], color, thickness)
    cv2.line(image, coords[2], coords[3], color, thickness)
    cv2.line(image, coords[3], coords[0], color, thickness)
    # for point in coords:
    #     if torch.is_tensor(box):
    #         box=(box.detach().cpu().numpy().astype(np.int32)).reshape(-1,2)
    #     else:
    #         box=np.array(box).reshape(-1,2)
    #     point1=box[0]
    #     point2=box[1]
    #     image=cv2.rectangle(image,tuple(point1),tuple(point2),color=(0,255,0),thickness=1)
    image=Image.fromarray(image)
    return image



label_root='/data/twkim/diffusion/ocr-dataset/pexel/labels/train'
img_root='/data/twkim/diffusion/ocr-dataset/pexel/images/train'
caption_path='/data/twkim/diffusion/ocr-dataset/pexel/pexel_blip_caption.txt'
# pexels-photo-8406973
with open(caption_path, 'r',encoding='utf-8', errors='ignore') as f:
    caption_lines=f.readlines()
    for line in caption_lines[4:]:
        line=line.strip()
        label, caption=line.split('\t')
        name=label.split('.')[0]
        if not 'pexels-photo-8406973' in name:
            continue
        img_path=None
        for suffix in ['jpeg','jpg','png']:
            img_path=os.path.join(img_root,name+'.'+suffix)
            if os.path.exists(img_path):
                break
        assert img_path is not None
        label_path=os.path.join(label_root,name+'.txt')
        img=Image.open(img_path)
        label_lines=open(label_path,'r').readlines()
        coords_list=[]
        text_boxes=[]
        img_w,img_h=img.size
        w_ratio=img_w/img_w
        h_ratio=img_h/img_h
        for line in label_lines:
            line=line.strip()
            splits=line.split('\t')
            coords=np.array(splits[0].split(',')).astype(np.float32).reshape(-1,2)
            coords[:,0]*=w_ratio
            coords[:,1]*=h_ratio
            x1,x2=min(coords[:,0]),max(coords[:,0])
            y1,y2=min(coords[:,1]),max(coords[:,1])
            text_width=(x2-x1)
            text_height=(y2-y1)
            if max(text_width,text_height)<0.015*(max(img_w,img_h)):
                continue
            text_boxes.append([x1,y1,x2,y2])
            coords_list.append(coords.astype(np.int32).reshape(-1).tolist())
        coords_list=np.array(coords_list)
        # img=img.resize((512,512))
        img=visualize_box(img,text_boxes,thickness=5)
        img.save('drawn.jpg')
        exit()