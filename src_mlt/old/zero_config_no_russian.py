class MLTConfig():
    def __init__(self):
        self.synth_script_ratios={
            'latin':0.2,
            'hindi':0.3,
            'thai':0.3,
            'bengali':0.3,
            'greek':0.3,
        }
        self.real_script_ratios={
            'latin':0.25,
            'hindi':0.35,
            'bengali':0.4,
        }
        self.latin_probs={
            'english':0.1,
            'italian':0.25,
            'german':0.3,
            'french':0.35,
        }