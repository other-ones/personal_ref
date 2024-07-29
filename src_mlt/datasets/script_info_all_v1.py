import json
from torch.utils.data import Dataset
import os
all_langs=['english','italian','german','french', 'arabic', 'hindi','greek','thai','russian','bengali']
class scriptInfo(Dataset):
    def __init__(self, 
                 charset_path,
                 ) -> None:
        self.lang2script={
            'english':'latin',
            'italian':'latin',
            'german':'latin',
            'french':'latin',
            'arabic':'arabic',
            'hindi':'hindi',
            'greek':'greek',
            'russian':'russian',
            'thai':'thai',
            'logo':'logo',
            'bengali':'bengali'
            }
        # self.target_languages=target_languages    
        # self.logo2idx=json.load(open('datasets/logo2idx_synth_merged.json'))
        # self.lang_set=all_langs
        
        
        # # 2. Parse scripts
        # self.script_set=[]
        # for lang in self.lang_set:
        #     self.script_set.append(self.lang2script[lang])
        # self.script_set=sorted(list(set(self.script_set)))
        
        

        # 3. Parse charset
        self.charset_path=charset_path    
        self.charset=json.load(open(charset_path))
        self.char2idx={}
        for char in self.charset:
            self.char2idx[char]=len(self.char2idx)
        self.idx2char={}
        for char in self.char2idx:
            charid=self.char2idx[char]
            self.idx2char[charid]=char
        
        self.target_scripts=['latin','hindi','thai','bengali','greek','russian']
        self.synth_script_ratios={
            'bengali':0.25,
            'thai':0.25,
            'hindi':0.2,
            'greek':0.125,
            'russian':0.125,
            'latin':0.05,
        }
        self.real_script_ratios={
            'latin':0.15,
            'hindi':0.4,
            'bengali':0.45,
        }
        self.latin_probs={
            'italian':0.35,
            'french':0.35,
            'german':0.2,
            'english':0.1,
        }