import pandas as pd

CORPUS_NAMES = ['human1', 'human2', 'gpt3_5', 'gpt4', 'llama3', 'phi3']


def load_corpora_dfs():
  print('Loading original corpus...')
  original_df = pd.read_csv('simplified_corpora/original.csv', encoding='utf-8')
  original_df = original_df[['original_text', 'document', 'paragraph_index']]
  original_df = original_df.sort_values(by=['document', 'paragraph_index'])
  print(original_df.shape)

  simplified_corpora_dfs = dict()
  for CORPUS_NAME in CORPUS_NAMES:
    print('Loading', CORPUS_NAME, ' corpus...')
    corpus_df = pd.read_csv(f'simplified_corpora/{CORPUS_NAME}.csv', encoding='utf-8')
    corpus_df = corpus_df[['original_text', 'document', 'paragraph_index', 'simplified_text']]
    corpus_df = corpus_df.sort_values(by=['document', 'paragraph_index'])
    print(corpus_df.shape)
    simplified_corpora_dfs[CORPUS_NAME] = corpus_df
  
  return original_df, simplified_corpora_dfs