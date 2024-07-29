import json
import numpy as np
import shutil
import os
import cv2
def generate_mask(height, width, coords,mask_type='rectangle',mask_values=None):
    mask_img = np.zeros((height, width)).astype(np.int32)
    for idx,coord in enumerate(coords):
        if mask_values is None:
            color=1
        else:
            color=mask_values[idx]+1
        coord=np.array(coord)
        if mask_type=='rectangle':
            coord=coord.astype(np.int32)
            coord=coord.reshape(-1,2)
            xs,ys=coord[:,0],coord[:,1] 
            max_x,max_y=np.max(xs),np.max(ys)
            min_x,min_y=np.min(xs),np.min(ys)
            x1,y1,x2,y2=[min_x,min_y,max_x,max_y]
            mask_img[y1:y2+1,x1:x2+1]=color
        elif mask_type=='polygon':
            mask_img = cv2.fillPoly(mask_img, pts=[np.array(coord).astype(np.int32).reshape(-1,2)], color=color) #xy order
    return mask_img





    
    
    
    

root='/data/twkim/diffusion/ocr-dataset/pexel/'
src_img_root='/data/twkim/diffusion/ocr-dataset/pexel/raw/images1/train'
ocr_label_root='/data/twkim/diffusion/ocr-dataset/pexel/raw/ocr_labels1/'
dst_label_root=os.path.join(root,'labels1/train')
dst_img_root=os.path.join(root,'images1/train')
os.makedirs(dst_label_root,exist_ok=True)
os.makedirs(dst_img_root,exist_ok=True)
dirs=os.listdir(ocr_label_root)
count=0
for dir in dirs:
    dirpath=os.path.join(ocr_label_root,dir)
    json_files=os.listdir(dirpath)
    for file in json_files:
        dstpath=os.path.join(dst_label_root,file.replace('.json','.txt'))
        file_path=os.path.join(dirpath,file)
        json_file=open(file_path)
        lines=json_file.readlines()
        if len(lines)==0:
            continue
        json_file.seek(0)
        data=json.load(open(file_path))
        img_h=data['analyze_result']['read_results'][0]['height']
        img_w=data['analyze_result']['read_results'][0]['width']
        img_h=int(img_h)
        img_w=int(img_w)
        assert len(data['analyze_result']['read_results'])==1
        num_written=0
        if len(data['analyze_result']['read_results'][0]['lines'])==0:
            continue
        dst_file=open(dstpath,'w',encoding='utf-8')
        coords_list=[]
        for line_data in data['analyze_result']['read_results'][0]['lines']:
            if len(line_data['words'])>8 or len(line_data['words'])<1:
                continue
            for word_data in line_data['words']:
                bbox=word_data['bounding_box']
                word=word_data['text']
                x1,y1,x2,y2,x3,y3,x4,y4=np.array(bbox).astype(np.int32)
                max_x,min_x=max([x1,x2,x3,x4]),min([x1,x2,x3,x4])
                max_y,min_y=max([y1,y2,y3,y4]),min([y1,y2,y3,y4])
                text_width=max_x-min_x
                text_height=max_y-min_y
                if max(text_width,text_height)<0.015*max(img_h,img_w):
                    continue
                coords_list.append([x1,y1,x2,y2,x3,y3,x4,y4])
                dst_file.write('{},{},{},{},{},{},{},{}\tUNK\t{}\n'.format(x1,y1,x2,y2,x3,y3,x4,y4,str(word.encode('utf-8').decode())))
                num_written+=1
        coords_list=np.array(coords_list)
        mask=generate_mask(img_h,img_w,coords=coords_list,mask_type='polygon')
        area_ratio=np.sum(mask)/(img_h*img_w)
        if not ((num_written>=1 and num_written<=8) and area_ratio>0.07):
            os.remove(dstpath)
        else:
            src_imgpath=None
            for suffix in ['.jpg','.jpeg','png']:
                if os.path.exists(os.path.join(src_img_root,file.replace('.json', suffix))):
                    src_imgpath=os.path.join(src_img_root,file.replace('.json', suffix))
                    break
            dst_imgpath=os.path.join(dst_img_root,file.replace('.json',suffix))
            if not os.path.exists(dst_imgpath) and src_imgpath is not None:
                os.symlink(src_imgpath,dst_imgpath)
        count+=1
        if(count%100)==0:
            print(count)