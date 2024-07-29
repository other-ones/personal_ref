import os
root='/data/twkim/diffusion/ocr-dataset/logos_seen_V1/logos'
lset=[]
flist=os.listdir(root)
for f in flist:
    fpath=os.path.join(root,f)
    lines=open(fpath).readlines()
    for item in lines:
        item=item.strip()
        if not item in lset:
            lset.append(item)
print(len(set(lset)))
print(lset)