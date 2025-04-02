import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('punkt')
nltk.download('stopwords')

class PatentDataLoader:
    def __init__(self, file_path):
        
        self.file_path = file_path
        self.data = None
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self):
        df=pd.read_csv(self.file_path)
        # df=df[0:1000]
        self.data = df
        print(len(self.data))
        print(f"Loaded {len(self.data)} patents from {self.file_path}")
        return self.data
    
    
    def prepare_data(self, combine_fields=True):
        if self.data is None:
            self.load_data()

        self.data['combined_text'] = self.data['title'] + ' ' + self.data['abstract']
        return self.data