import json
from torch.utils.data import Dataset
import os
all_langs=['english','italian','german','french', 'arabic', 'hindi','hebrew','greek','thai','russian','bengali','logo']
class scriptInfo(Dataset):
    def __init__(self, 
                 charset_path,
                 target_languages=None,
                #  instance_data_list_real,
                #  instance_data_list_synth
                 ) -> None:
        self.lang2script={
            'english':'latin',
            'italian':'latin',
            'german':'latin',
            'french':'latin',
            'arabic':'arabic',
            'hindi':'hindi',
            'hebrew':'hebrew',
            'greek':'greek',
            'russian':'cyrillic',
            # 'latin':'latin',
            'thai':'thai',
            'logo':'logo',
            'bengali':'bengali'
            }
        self.charset_path=charset_path    
        self.target_languages=target_languages    
        self.lang_set=[]
        self.logo2idx=json.load(open('datasets/logo2idx_synth_merged.json'))
        # self.instance_data_list_real=sorted(instance_data_list_real)
        # self.instance_data_list_synth=instance_data_list_synth  

        
        
        # 1. Parse language
        # for item in self.instance_data_list_real:
        #     if 'mario_laion' in item or 'icdar2019' in item or ('logos' in item and '_ocr' in item):
        #         self.lang_set.append('english')
        #     if 'logo' in item:
        #         self.lang_set.append('logo')
        # for synth_db_name in instance_data_list_synth:
        #     for lang in all_langs:
        #         if lang in synth_db_name:
        #             self.lang_set.append(lang)
        # self.lang_set=sorted(list(set(self.target_languages)))
        # print(self.lang_set,'self.lang_set')
        self.lang_set="english-thai-greek-russian-german-italian-french-hindi-bengali".split('-')
        
        
        # 2. Parse scripts
        self.script_set=[]
        for lang in self.lang_set:
            self.script_set.append(self.lang2script[lang])
        self.script_set=sorted(list(set(self.script_set)))
        
        
        
        
        self.script2idx={}
        self.idx2script={}
        for idx,script in enumerate(self.script_set):
            self.script2idx[script]=idx
            self.idx2script[idx]=script
        # 3. Parse charset
        self.charset=json.load(open(charset_path))
        self.char2idx={}
        for char in self.charset:
            self.char2idx[char]=len(self.char2idx)
        self.idx2char={}
        for char in self.char2idx:
            charid=self.char2idx[char]
            self.idx2char[charid]=char
        