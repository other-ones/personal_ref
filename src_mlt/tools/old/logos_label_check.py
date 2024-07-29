import os
root='/data/twkim/diffusion/ocr-dataset/flickr_logos/labels/train'
imgroot='/data/twkim/diffusion/ocr-dataset/flickr_logos/image/train'
files=os.listdir(root)
logos=[]
for file in files:
    labelpath=os.path.join(root,file)
    lines=open(labelpath).readlines()
    imgpath=os.path.join(imgroot,file.split('.')[0]+'.jpg')
    for line in lines:
        line=line.strip()
        splits=line.split('\t')
        logo=splits[-1]
        logos.append(logo)
logos=list(set(logos))
print(sorted(logos))