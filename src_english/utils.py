import os

from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as TF
import numpy as np
import random
from PIL import Image,ImageDraw,ImageFont
import random
import cv2
import string
Image.MAX_IMAGE_PIXELS = 1000000000

charset=list(string.ascii_letters+string.punctuation+string.digits)
def generate_gaussian_map(width,height, box):
    # box_x0,box_y0,box_x1,box_y1,box_x2,box_y2,box_x3,box_y3=box
    box=np.array(box)
    box=box.reshape(-1,2)
    xs,ys=box[:,0],box[:,1]
    box_minx,box_miny=np.min(xs),np.min(ys)
    box_maxx,box_maxy=np.max(xs),np.max(ys)
    
    center_x=int((box_minx+box_maxx)/2)
    center_y=int((box_miny+box_maxy)/2)
    box_width=int(box_maxx-box_minx)
    box_height=int(box_maxy-box_miny)
    covariance_matrix = np.array([[box_width**1.5, 0], [0, box_height**1.5]])  # Covariance matrix
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)
    x, y = np.meshgrid(x, y)
    
    x0, y0 = int(center_x),int(center_y)
    xy = np.stack((x - x0, y - y0), axis=-1)
    inv_cov_matrix = np.linalg.inv(covariance_matrix)
    # Adjusted matrix multiplication
    exponent = np.sum(xy @ inv_cov_matrix * xy, axis=-1)
    gauss_map = np.exp(-0.5 * exponent)
    return gauss_map
def generate_pos_neg_masks(height, width, 
                           coords_list,
                           is_keyword_token,
                           tidx_to_crd_idx,
                           fg_mask,
                           eot_idx,mask_type='rectangle'):
    # mask_list=[np.zeros((height, width)).astype(np.float32)] #indexed by token_idx
    # bin_mask:512,512 (fg:1 bg:0)
    pos_mask_list=np.zeros((len(is_keyword_token),height,width))
    neg_mask_list=np.zeros((len(is_keyword_token),height,width))
    # if it's a non-keyword token,
    # then it's corresponding negative mask is foreground_mask
    # first token is special token and non-keyword
    # nonkey_ratio=0.1
    # pos_mask_list[0]=(1-fg_mask) # for "<|startoftext|>" (it;s non-keyword)
    # neg_mask_list[0]=fg_mask # for "<|startoftext|>" (it's non-keyword)
    for tidx in range(1,len(is_keyword_token)):
        # 1) end of text token
        if tidx ==eot_idx: 
            # attention maps for padding tokens are not altered
            # pos_mask_list[tidx]=(1-fg_mask)
            # neg_mask_list[0]=fg_mask
            break
        # 2) non-keyword
        if not tidx in tidx_to_crd_idx: # None Keyword
            # if it's not a keyword token,
            # the it's corresponding negative mask is background_mask
            # i.e., (1-fg_mask)
            # neg_mask_list[tidx]=fg_mask #non-keyword's negative mask is fg_mask
            pos_mask_list[tidx]=(1-fg_mask)
            neg_mask_list[0]=fg_mask
            continue

            
        # 2) keyword
        corresp_crd_idx=tidx_to_crd_idx[tidx]
        coords=coords_list[corresp_crd_idx]
        if mask_type=='gaussian':
            # mask_img=generate_mask(512,512,[coords],mask_type='gaussian') #normalized [min_value,1]
            mask_img=generate_gaussian_map(512,512,coords)
        else:
            mask_img=generate_mask(512,512,[coords],mask_type='rectangle')
        neg_mask_list[tidx]=(1-mask_img)
        pos_mask_list[tidx]=mask_img
    

    return pos_mask_list,neg_mask_list    
def get_uniform_layout(words_list,ar_list):
    dev_ratio=np.random.uniform(low=0.15,high=0.25) # deviation from the border
    print(dev_ratio)
    xdev=int(dev_ratio*512)
    ydev=int(dev_ratio*512)
    # inner width and height after deviation from borders
    inner_size=int(512*(1-dev_ratio*2))
    num_words=len(words_list)
    coords_list=[]
    if num_words==1:
        section_height=inner_size
        section_width=inner_size
        # first sample the width and x 
        # then corresponding height and y
        # limit value for x
        left_lim=xdev
        right_lim=xdev+(section_width)
        # limit value for y
        top_lim=ydev
        bot_lim=ydev+(section_height)
        # sample width
        if ar_list[0]>1: #width>height
            sampled_width=np.random.randint(low=int(section_width*0.45),high=int(0.8*section_width))
        else: # width<height
            sampled_width=np.random.randint(low=int(section_width*0.3),high=int(0.5*section_width))
        # get x range and sample coordinate value
        sample_x_max=(right_lim-sampled_width)
        sample_x_min=left_lim
        sampled_x=np.random.randint(low=sample_x_min,high=sample_x_max)
        aspect_ratio=ar_list[0]
        aspect_ratio=min(2.8,aspect_ratio)
        # get height corresponding to sampled width
        sampled_height=int(sampled_width/aspect_ratio)
        # get y range corresponding to sampled x and 
        sample_y_max=bot_lim-sampled_height
        sample_y_min=top_lim
        # if corresponding y coordinate is out of range, set it as limit value
        if sample_y_max<=0 or (sample_y_min>=sample_y_max):
            sampled_y=top_lim
        else:
            sampled_y=np.random.randint(low=sample_y_min,high=sample_y_max)
        coords=[sampled_x,sampled_y,
                sampled_x+sampled_width,sampled_y,
                sampled_x+sampled_width,sampled_y+sampled_height,
                sampled_x,sampled_y+sampled_height,
                ]
        coords_list.append(coords)
    else:
        row_size=2 # number of items per row
        num_rows=np.ceil(num_words/row_size)
        section_height=int(inner_size/num_rows)
        section_width=int(inner_size/2)
        min_height=(section_height*0.3)
        min_height=max(min_height,45)
        for idx in range(num_words):
            row_idx=idx//2
            col_idx=int((idx%2)!=0)
            left_lim=xdev+section_width*col_idx
            right_lim=xdev+(section_width*(col_idx+1))
            top_lim=ydev+(section_height*row_idx)
            bot_lim=ydev+(section_height*(row_idx+1))
            # get height -> get y coord
            if ar_list[idx]>1:#ar: width/heigth
                sampled_height=np.random.randint(low=(section_height*0.4),high=int(0.5*section_height))
            else:
                sampled_height=np.random.randint(low=(section_height*0.45),high=int(0.75*section_height))
            if (num_words%2)!=0:
                sampled_height=int(sampled_height*0.9)
            sample_y_max=(bot_lim-sampled_height)
            sample_y_min=top_lim
            sampled_y=np.random.randint(low=sample_y_min,high=sample_y_max)
            # get x coordinate
            word=words_list[idx]
            aspect_ratio=ar_list[idx]
            aspect_ratio=min(2.8,aspect_ratio)
            sampled_width=int(sampled_height*aspect_ratio)
            sampled_width=min(int(section_width*1.4),sampled_width) 
            sample_x_max=right_lim-sampled_width
            sample_x_min=left_lim
            if sample_x_max<=0 or (sample_x_min>=sample_x_max):
                sampled_x=left_lim
            else:
                sampled_x=np.random.randint(low=sample_x_min,high=sample_x_max)

            coords=[sampled_x,sampled_y,
                    sampled_x+sampled_width,sampled_y,
                    sampled_x+sampled_width,sampled_y+sampled_height,
                    sampled_x,sampled_y+sampled_height,
                    ]
            coords_list.append(coords)
    return coords_list


# def get_uniform_layout(words_list,leng2ar):
#     dev_ratio=np.random.uniform(low=0.1,high=0.25) # deviation from the border
#     xdev=int(dev_ratio*512)
#     ydev=int(dev_ratio*512)
#     inner_size=int(512*(1-dev_ratio*2))
#     row_size=2 # number of items per row
#     num_words=len(words_list)
#     num_rows=np.ceil(num_words/row_size)
#     # print(num_words,'num_words',row_size,'row_size',num_rows,'num_rows')
#     section_height=int(inner_size/num_rows)
#     section_width=int(inner_size/2)
#     min_height=(section_height*0.25)
#     min_height=max(min_height,45)
#     coords_list=[]
#     for idx in range(num_words):
#         row_idx=idx//2
#         col_idx=int((idx%2)==0)
#         left_lim=xdev+section_width*col_idx
#         right_lim=xdev+(section_width*(col_idx+1))
#         top_lim=ydev+(section_height*row_idx)
#         bot_lim=ydev+(section_height*(row_idx+1))
#         # get height -> get y coord
#         sampled_height=np.random.randint(low=min_height,high=int(0.6*section_height))
#         sample_y_max=(bot_lim-sampled_height)
#         sample_y_min=top_lim
#         sampled_y=np.random.randint(low=sample_y_min,high=sample_y_max)

#         # get x coordinate
#         word=words_list[idx]
#         word_length=len(word)
#         aspect_ratio=leng2ar[word_length]
#         aspect_ratio=min(2.5,aspect_ratio)
#         sampled_width=int(sampled_height*aspect_ratio)
#         sampled_width=min(section_width,sampled_width)     

#         sample_x_max=right_lim-sampled_width
#         sample_x_min=left_lim
#         if sample_x_max<=0 or (sample_x_min>=sample_x_max):
#             sampled_x=left_lim
#         else:
#             sampled_x=np.random.randint(low=sample_x_min,high=sample_x_max)

#         coords=[sampled_x,sampled_y,
#                 sampled_x+sampled_width,sampled_y,
#                 sampled_x+sampled_width,sampled_y+sampled_height,
#                 sampled_x,sampled_y+sampled_height,
#                 ]
#         coords_list.append(coords)

#     return coords_list
def random_crop_with_roi(image, word_coordinates):
    image=np.array(image)
    # Extract ROI coordinates
    word_coordinates=np.array(word_coordinates).reshape(-1,4,2)
    xs,ys=word_coordinates[:,:,0].reshape(-1),word_coordinates[:,:,1].reshape(-1)
    tx1,ty1=min(xs),min(ys)
    tx2,ty2=max(xs),max(ys)
    text_roi=[tx1,ty1,tx2,ty2]
    x1, y1, x2, y2 = text_roi

    # Calculate ROI width and height
    roi_width = x2 - x1
    roi_height = y2 - y1
    max_roi_side=max(roi_width,roi_height)
    # min_roi_side=min(roi_width,roi_height)

    # Calculate the center of the ROI
    roi_center_x = (x1 + x2) // 2
    roi_center_y = (y1 + y2) // 2

    # Define the allowable range for the resulting square image size
    # min_size = max(max(500,max_roi_side), max(roi_height+50, roi_width+50))
    # min_size=max(500,max_roi_side+50)
    min_size=max_roi_side+50
    max_size = min(image.shape[1], image.shape[0])  # Limit max size to image dimensions
    max_size=min(1.5*max_roi_side,max_size)
    if min_size>=max_size:
        # cropped_image = image[y1:y2, x1:x2]
        # word_coordinates[:,:,0]-=x1
        # word_coordinates[:,:,1]-=y1
        # cropped_image=Image.fromarray(cropped_image)
        # return cropped_image, word_coordinates
        return Image.fromarray(image),word_coordinates
    # min_size=min(min_size,max_size)
    # print('cropped')
    side_length = random.randint(min_size, int(max_size))
    # Calculate the new ROI coordinates based on the center and square size
    new_x1 = max(0, roi_center_x - side_length // 2)
    new_y1 = max(0, roi_center_y - side_length // 2)
    new_x2 = min(image.shape[1], new_x1 + side_length)
    new_y2 = min(image.shape[0], new_y1 + side_length)

    # # Adjust the coordinates to make the resulting region square
    cropped_image = image[new_y1:new_y2, new_x1:new_x2]
    word_coordinates[:,:,0]-=new_x1
    word_coordinates[:,:,1]-=new_y1
    cropped_image=Image.fromarray(cropped_image)
    return cropped_image, word_coordinates.reshape(-1,8)
def non_max_suppression_fast(boxes, overlapThresh=0.8):
	boxes=np.array(boxes)
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
		

	# return only the bounding boxes that were picked using the
	# integer data type
	# return boxes[pick].astype("int")
	return pick
def rotate_point(point, center, angle):
    # rotation
    angle = math.radians(angle)
    x = point[0] - center[0]
    y = point[1] - center[1]
    x1 = x * math.cos(angle) - y * math.sin(angle)
    y1 = x * math.sin(angle) + y * math.cos(angle)
    x1 += center[0]
    y1 += center[1]
    return int(x1), int(y1)


def generate_random_rectangles(image):
        # randomly generate 0~3 masks
        rectangles = []
        box_num = random.randint(0, 3)
        for i in range(box_num):
            x = random.randint(0, image.size[0])
            y = random.randint(0, image.size[1])
            w = random.randint(16, 256)
            h = random.randint(16, 96) 
            angle = random.randint(-45, 45)
            p1 = (x, y)
            p2 = (x + w, y)
            p3 = (x + w, y + h)
            p4 = (x, y + h)
            center = ((x + x + w) / 2, (y + y + h) / 2)
            p1 = rotate_point(p1, center, angle)
            p2 = rotate_point(p2, center, angle)
            p3 = rotate_point(p3, center, angle)
            p4 = rotate_point(p4, center, angle)
            rectangles.append((p1, p2, p3, p4))
        return rectangles
def calculate_iou(box1, box2):
    # Calculate the intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of intersection
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate the area of each box
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate the Union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area

    return iou
def check_overlap(word_coordinates, x, y, width, height):
    box1 = [x, y, x + width, y + height]
    for x1, y1, x2, y2 in word_coordinates:
        box2 = [x1, y1, x2, y2]
        if calculate_iou(box1, box2) > 0.1:
            return True
    return False
    
import math
def rotate_coordinates(x1, y1, x2, y2, x3, y3, x4, y4, angle):
    # Calculate the center of the rectangle
    center_x = (x1 + x2 + x3 + x4) / 4
    center_y = (y1 + y2 + y3 + y4) / 4

    # Convert the angle from degrees to radians
    angle_rad = math.radians(angle)

    # Function to rotate a point (x, y) around a center (cx, cy)
    def rotate_point(x, y, cx, cy, angle):
        rotated_x = cx + (x - cx) * math.cos(angle) - (y - cy) * math.sin(angle)
        rotated_y = cy + (x - cx) * math.sin(angle) + (y - cy) * math.cos(angle)
        return rotated_x, rotated_y

    # Rotate each point of the rectangle
    rotated_x1, rotated_y1 = rotate_point(x1, y1, center_x, center_y, angle_rad)
    rotated_x2, rotated_y2 = rotate_point(x2, y2, center_x, center_y, angle_rad)
    rotated_x3, rotated_y3 = rotate_point(x3, y3, center_x, center_y, angle_rad)
    rotated_x4, rotated_y4 = rotate_point(x4, y4, center_x, center_y, angle_rad)

    return rotated_x1, rotated_y1, rotated_x2, rotated_y2, rotated_x3, rotated_y3, rotated_x4, rotated_y4
# def create_mask_from_coords(height, width, coords,mask_type='rectangle'):
#     mask_img = np.zeros((height, width)).astype(np.int32)
#     for coord in coords:
#         coord=np.array(coord)
#         coord[np.where(coord<0)]=0
#         coord[np.where(coord>511)]=511
#         if mask_type=='rectangle':
#             coord=coord.astype(np.int32)
#             coord=coord.reshape(-1,2)
#             xs,ys=coord[:,0],coord[:,1] 
#             max_x,max_y=np.max(xs),np.max(ys)
#             min_x,min_y=np.min(xs),np.min(ys)
#             x1,y1,x2,y2=[min_x,min_y,max_x,max_y]
#             mask_img[y1:y2+1,x1:x2+1]=1
#         elif mask_type=='polygon':
#             mask_img = cv2.fillPoly(mask_img, pts=[np.array(coord).astype(np.int32).reshape(-1,2)], color=(1,1)) #xy order
#     return mask_img
def generate_mask(height, width, coords,mask_type='rectangle'):
    mask_img = np.zeros((height, width)).astype(np.int32)
    for coord in coords:
        if not len(coord):
            print(len(coord),'len(coord)')
            continue
        coord=np.array(coord)
        coord[np.where(coord<0)]=0
        coord[np.where(coord>511)]=511
        if mask_type=='rectangle':
            coord=coord.astype(np.int32)
            coord=coord.reshape(-1,2)
            xs,ys=coord[:,0],coord[:,1]
            max_x,max_y=np.max(xs),np.max(ys)
            min_x,min_y=np.min(xs),np.min(ys)
            x1,y1,x2,y2=[min_x,min_y,max_x,max_y]
            mask_img[y1:y2+1,x1:x2+1]=1
        elif mask_type=='polygon':
            mask_img = cv2.fillPoly(mask_img, pts=[np.array(coord).astype(np.int32).reshape(-1,2)], color=(1,1)) #xy order
    return mask_img
def generate_spatial_rendering(width, height, words,dst_coords=None,font_path='fonts/GoNotoCurrent.ttf'):
    image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for i in range(len(words)):
        if font_path=='random_font':
            fonts=os.listdir('fonts_')
            np.random.shuffle(fonts)
            cur_font_path=os.path.join('fonts_',fonts[0])
        else:
            cur_font_path=font_path
        coords=dst_coords[i]
        word = words[i]
        assert len(word)
        x1, y1, x2, y2 = np.array(coords).astype(np.int32)#np.min(xs),np.min(ys),np.max(xs),np.max(ys)
        region_width = x2 - x1
        region_height = y2 - y1
        min_side=min(region_width,region_height)
        if region_height>(region_width*2):
            font_size = int(max(region_width, region_height) / len(word))
            font_size=font_size*0.75
            font_size=int(font_size)
            font_size=max(1,font_size)
            font_size=min(min_side,font_size)
            font_size=max(5,font_size)
            font = ImageFont.truetype(cur_font_path, font_size)
            text_width, text_height = draw.textsize(word, font=font)
            text_x = x1 + (region_width - text_width) // 2
            text_y = y1 + (region_height - text_height) // 2
            draw.text((text_x, text_y), word, font=font, fill=0,spacing=1)
        else: 
            font_size = int(max(region_width, region_height)*1.4 / len(word))
            font_size=int(font_size)
            font_size=max(35,font_size)
            font_size=min(min_side,font_size)
            font_size=min(56,font_size)
            font = ImageFont.truetype(cur_font_path, font_size)
            text_width, text_height = draw.textsize(word, font=font)
            text_x = x1 + (region_width - text_width) // 2
            text_y = y1 + (region_height - text_height) // 2
            draw.text((text_x, text_y), word, font=font, fill=0)
    return image
    
    
# def generate_image_with_text(width, height, words):
#     min_font_size = 20
#     max_font_size =40
#     # Create a blank image with an alpha channel
#     # font_path='/home/twkim/project/glyph_rec/fonts/GoNotoCurrent.ttf'
#     # available_fonts=['/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf', '/usr/share/fonts/truetype/liberation2/LiberationMono-Italic.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSans-Italic.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSerif-BoldItalic.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSerif-Italic.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSans-BoldItalic.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf', '/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf', '/usr/share/fonts/truetype/liberation2/LiberationMono-BoldItalic.ttf', '/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf', '/usr/share/fonts/truetype/liberation/LiberationSansNarrow-BoldItalic.ttf', '/usr/share/fonts/truetype/liberation/LiberationMono-Italic.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', '/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf', '/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf', '/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf', '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf', '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf', '/usr/share/fonts/truetype/liberation/LiberationMono-BoldItalic.ttf']
#     available_fonts=[ '/usr/share/fonts/truetype/liberation2/LiberationMono-Italic.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSans-Italic.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSerif-BoldItalic.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSerif-Italic.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSans-BoldItalic.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf', '/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf', '/usr/share/fonts/truetype/liberation2/LiberationMono-BoldItalic.ttf', '/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf', '/usr/share/fonts/truetype/liberation/LiberationSansNarrow-BoldItalic.ttf', '/usr/share/fonts/truetype/liberation/LiberationMono-Italic.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', '/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf', '/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf', '/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf', '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf', '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf', '/usr/share/fonts/truetype/liberation/LiberationMono-BoldItalic.ttf']
#     font_path=np.random.choice(available_fonts)
#     image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
#     draw = ImageDraw.Draw(image)
#     word_coordinates = []
#     for word in words:
#         # Generate random font size and angle
#         font_size = random.randint(min_font_size, max_font_size)
#         angle = random.randint(-45, 45)
#         # Load the font with the specified path and size
#         font = ImageFont.truetype(font_path, font_size)
#         # Get the width and height of the current word
#         word_width, word_height = draw.textsize(word, font=font)
#         # Find a non-overlapping position for the word
#         x = random.randint(0, width - word_width)
#         y = random.randint(0, height - word_height)
#         for _ in range(5):
#         # while check_overlap(word_coordinates, x, y, word_width, word_height):
#             x = random.randint(0, width - word_width)
#             y = random.randint(0, height - word_height)
#             if not check_overlap(word_coordinates, x, y, word_width, word_height):
#                 break
#         # Create a blank image with an alpha channel for the word
#         word_image = Image.new("RGBA", (word_width, word_height), (255, 255, 255, 0))
#         word_draw = ImageDraw.Draw(word_image)
#         # Draw the word on the word image
#         word_draw.text((0, 0), word, fill=(0, 0, 0), font=font)

#         # Rotate the word image
#         rotated_word = word_image.rotate(angle, expand=True, resample=Image.BICUBIC)

#         # Calculate the center of rotation
#         center_x = x + word_width / 2
#         center_y = y + word_height / 2

#         # Adjust the rotated coordinates based on the image size
#         rotated_x = max(0, center_x - rotated_word.width / 2)
#         rotated_y = max(0, center_y - rotated_word.height / 2)
#         rotated_x = min(rotated_x, width - rotated_word.width)
#         rotated_y = min(rotated_y, height - rotated_word.height)

#         # Paste the rotated word image onto the main image
#         image.paste(rotated_word, (int(rotated_x), int(rotated_y)), mask=rotated_word)

#         # Store the coordinates of the word
#         word_coordinates.append((int(rotated_x), int(rotated_y), int(rotated_x + rotated_word.width), int(rotated_y + rotated_word.height)))

#     # Convert the image to 'L' mode (grayscale)
#     image = image.convert('RGB')
#     return image, word_coordinates
def visualize_polygon(image,coords):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.6
    image=np.array(image)
    color=(0,255,0)
    thickness=1
    coords=np.array(coords)
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

def visualize_box(image,boxes,chars=None):
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
            if not len(box):
                continue
            if torch.is_tensor(box):
                box=(box.detach().cpu().numpy().astype(np.int32)).reshape(-1,2)
            else:
                box=np.array(box).reshape(-1,2)
            point1=box[0]
            point2=box[1]
            image=cv2.rectangle(image,tuple(point1),tuple(point2),color=(0,255,0),thickness=1)
    image=Image.fromarray(image)
    return image
def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images
class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character,device):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        self.device=device
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        # print(batch_text.shape,'batch_text.shape',batch_max_length,'batch_max_length')
        
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(self.device), torch.IntTensor(length).to(self.device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


viz_count=0
def visualize_mask_image_for_debugging(input_img,instance_mask,words,dir_name,db):
    global viz_count
    # # visualize
    instance_mask_viz = instance_mask * 255
    instance_mask_viz2 =instance_mask
    h,w=instance_mask_viz2.shape
    instance_mask_viz2=np.repeat(instance_mask_viz2.reshape(h, w, 1), 3, axis=2)
    Image.fromarray(instance_mask_viz).convert('L').save('{}/{}_{}_{}_instance_mask.jpg'.format(dir_name,viz_count + 1, words[0].replace('/',''),db))
    input_img_masked=instance_mask_viz2*input_img
    input_img_masked=input_img_masked.astype(np.uint8)
    input_img_masked=Image.fromarray(input_img_masked)
    input_img_masked.save('{}/{}_{}_{}_input_img_masked.jpg'.format(dir_name,viz_count+1,words[0].replace('/',''),db))
    viz_count+=1
    

def random_crop_image_mask(image_np,mask_np,coords):
    h,w,_=image_np.shape
    # print(coords.shape,'coords.shape')
    # for item in coords:
    #     print(coords.shape,'item.shape')
    mask_x_min=np.infty
    mask_y_min=np.infty
    mask_x_max=-1
    mask_y_max=-1
    for item in coords:
        item=np.array(item).astype(np.int32).reshape(-1,2)
        cur_x_min=np.min(item[:,0])
        cur_y_min=np.min(item[:,1])
        cur_x_max=np.max(item[:,0])
        cur_y_max=np.max(item[:,1])
        if cur_x_max>mask_x_max:
            mask_x_max=cur_x_max
        if cur_y_max>mask_y_max:
            mask_y_max=cur_y_max
        if cur_y_min<mask_y_min:
            mask_y_min=cur_y_min
        if cur_x_min<mask_x_min:
            mask_x_min=cur_x_min

    if mask_x_max<w-1:
        crop_x_max=np.random.randint(low=mask_x_max,high=w)
    else:
        crop_x_max=w-1
    if mask_y_max < h - 1:
        crop_y_max = np.random.randint(low=mask_y_max, high=h)
    else:
        crop_y_max = h - 1

    if mask_x_min==0:
        crop_x_min=0
    else:
        crop_x_min=np.random.randint(low=0,high=mask_x_min+1)
    if mask_y_min==0:
        crop_y_min=0
    else:
        crop_y_min=np.random.randint(low=0,high=mask_y_min+1)
    image_cropped=image_np[crop_y_min:crop_y_max+1,crop_x_min:crop_x_max+1]
    mask_cropped=mask_np[crop_y_min:crop_y_max+1,crop_x_min:crop_x_max+1]
    return image_cropped,mask_cropped

def create_random_mask(height, width, num_regions=1, mask_transforms=None,target_ar=None):
    img = np.zeros((height, width), dtype=np.int32)
    for i in range(num_regions):
        # generate mask
        if target_ar=='landscape':
            mask_height = np.random.randint(int(0.07 * height), int(0.4 * height))
            mask_width = np.random.randint(min(int(0.4 * width)-1,int(mask_height*1.2)), int(0.4 * width))
        elif target_ar=='portrait':
            mask_width = np.random.randint(int(0.07 * width), int(0.4 * width))
            mask_height = np.random.randint(min(int(0.4 * height)-1,int(mask_width*1.2)), int(0.4 * height))
        mask_x = np.random.randint(0, width - mask_width)
        mask_y = np.random.randint(0, height - mask_height)
        img[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    img = Image.fromarray(img)#.convert('L')
    return img


# For Inpainting
import math
def rotate_point(point, center, angle):
        # rotation
        angle = math.radians(angle)
        x = point[0] - center[0]
        y = point[1] - center[1]
        x1 = x * math.cos(angle) - y * math.sin(angle)
        y1 = x * math.sin(angle) + y * math.cos(angle)
        x1 += center[0]
        y1 += center[1]
        return int(x1), int(y1)
def generate_random_rectangles(width,height):
        # randomly generate 0~3 masks
        rectangles = []
        box_num = random.randint(0, 3)
        for i in range(box_num):
            x = random.randint(0, width)
            y = random.randint(0, height)
            w = random.randint(16, 256)
            h = random.randint(16, 96) 
            angle = random.randint(-45, 45)
            p1 = (x, y)
            p2 = (x + w, y)
            p3 = (x + w, y + h)
            p4 = (x, y + h)
            center = ((x + x + w) / 2, (y + y + h) / 2)
            p1 = rotate_point(p1, center, angle)
            p2 = rotate_point(p2, center, angle)
            p3 = rotate_point(p3, center, angle)
            p4 = rotate_point(p4, center, angle)
            rectangles.append((p1, p2, p3, p4))
        return rectangles
def augment_mask(inpaint_mask):
    inpaint_mask=Image.fromarray(inpaint_mask)
    rectangles=generate_random_rectangles(512,512)
    draw_image_mask = ImageDraw.ImageDraw(inpaint_mask)
    for rectangle in rectangles:
        draw_image_mask.polygon(rectangle, fill=0)
    inpaint_mask=np.array(inpaint_mask)
    return inpaint_mask