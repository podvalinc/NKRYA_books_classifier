from preprocess.druzhkin import DruzhkinAnalyzer
import numpy as np
import os
from tqdm import tqdm
separator = '\\sep'
analyzer = DruzhkinAnalyzer()
for idx in range(3):
    for file_name in tqdm(os.listdir('tokenized_70_'+str(idx))):
        with open(os.path.join('tokenized_70_'+str(idx), file_name), encoding='utf-8') as f:
            texts = f.read();
            texts = texts.split(separator)
            res = [analyzer.analyze(text) for text in texts]
            res = np.array(res)
            np.savetxt('analyzed_'+str(idx)+'/'+file_name, res)
