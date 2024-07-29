import os
root1='/data/twkim/diffusion/ocr-dataset/merged_logos3/images/train'
root2='/data/twkim/diffusion/ocr-dataset/merged_logos3_ocr/images/train'
files1=os.listdir(root1)
files2=os.listdir(root2)
print(len(set(files1)-set(files2)))
print(len(set(files2)-set(files1)))
print(len(files1))
root1='/data/twkim/diffusion/ocr-dataset/merged_logos3/labels/train'
root2='/data/twkim/diffusion/ocr-dataset/merged_logos3_ocr/labels/train'
files1=os.listdir(root1)
files2=os.listdir(root2)
print(len(set(files1)-set(files2)))
print(len(set(files2)-set(files1)))
print(len(files1))
