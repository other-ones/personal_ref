import os
lang='french'
root='/data/twkim/diffusion/ocr-dataset/synth_pexel_{}'.format(lang)
images=os.listdir(os.path.join(root,'images/train'))
dst_file=open('sent.txt','w')
for f in images:
    dst_file.write('{}\n'.format(f))
