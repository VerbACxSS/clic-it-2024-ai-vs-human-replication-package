import pandas as pd

from utils import openai_utils
from utils.Phi3Model import Phi3
from utils.Llama3Model import Llama3

# Load the original corpus
corpus_df = pd.read_csv('simplified_corpora/original.csv', encoding='utf-8')

# Create and save the simplified gpt4 parallel corpus
corpus_df['simplified_text'] = openai_utils.predict(corpus_df['original_text'].tolist(), _model='gpt-4-turbo-2024-04-09')
corpus_df.to_csv('simplified_corpora/gpt4.csv', index=False)

# Create and save the simplified gpt3_5 parallel corpus
corpus_df['simplified_text'] = openai_utils.predict(corpus_df['original_text'].tolist(), _model='gpt-3.5-turbo-0125')
corpus_df.to_csv('simplified_corpora/gpt3_5.csv', index=False)

# Create and save the simplified llama3 parallel corpus
corpus_df['simplified_text'] = Llama3().predict(corpus_df['original_text'].tolist())
corpus_df.to_csv('simplified_corpora/llama3.csv', index=False)

# Create and save the simplified phi3 parallel corpus
corpus_df['simplified_text'] = Phi3().predict(corpus_df['original_text'].tolist())
corpus_df.to_csv('simplified_corpora/phi3.csv', index=False)