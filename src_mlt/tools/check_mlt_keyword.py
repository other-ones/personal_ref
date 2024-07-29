fpath='/data/twkim/ocr-dataset/seen_mlt_v3/seen_mlt_v3_blip_caption.txt'
lines=open(fpath).readlines()
for line in lines:
    line=line.strip()
    words=line.split()
    count=0
    for word in words:
        if word=='words':
            count+=1
    assert count==1,'count==1'
print('no error')

