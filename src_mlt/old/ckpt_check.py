import json
old_set=json.load(open('/home/twkim/project/azure/refdiffuser/src_mlt/ckpt/chartokenizer/char_vocab_mlt_l2r.json'))
new_set=json.load(open('/home/twkim/project/azure/refdiffuser/src_mlt/ckpt/chartokenizer/char_vocab_mlt_all.json'))
print(len(old_set),'old_set')
print(len(new_set),'new_set')
print(set(old_set)-set(new_set),'old-new')
print(set(new_set)-set(old_set),'new-old')