import re
import numpy as np
import math
import pandas as pd
from collections import Counter

def tokenize(text):
  """Returns a list of words that make up the text.

  Note: for simplicity, lowercase everything.
  Requirement Use regular expressions to satisfy this function

  Params: {text: String}
  Returns: List
  """
  # fix regular expression to remove links
  return re.findall('[a-z]+', text.lower())

def tokenize_mbti(df):
  """Returns a list of tokenized words.
  Params: {df: Dataframe}
  Returns: List
  """
  tokenized = []
  for idx, row in df.iterrows():
      t = row['type']
      text = tokenize(row['posts'])
      tokenized.append((t, text))
  return tokenized

def mbti_tokenized(tokenized):
  # returns the mbti as keys with the tokenized words as values
  mbti_dict = {}
  for (a, b) in tokenized:
    if a not in mbti_dict:
        mbti_dict[a] = b
    else:
        mbti_dict[a] += b
  return mbti_dict

def build_inverted_index(mbti_dict):
  # initialization
  d = {}
 
  # updating inverted_index dict
  mbti_index = list(mbti_dict.keys())
  for mbti in mbti_dict:
      tokenized = mbti_dict[mbti]
      count_tokenized = Counter(tokenized)
      for word in count_tokenized:
          if word in d:
            d[word].append((mbti_index.index(mbti), count_tokenized[word]))
          else:
            d[word] = [(mbti_index.index(mbti), count_tokenized[word])]
  return d

def compute_idf(inv_idx, n_docs, min_df=1, max_df_ratio=1):
  # initialization
  idf_dict = {}
  
  # compute IDF
  for word in inv_idx:
      word_df = len(inv_idx[word]) 
      word_df_ratio = word_df / n_docs
      if word_df >= min_df and word_df_ratio < max_df_ratio:
          idf_dict[word] = math.log2(n_docs / (1 + word_df))          
  return idf_dict

def valid_query(input_query, idf_dict):
  # check if query is valid
  tokenize_query = tokenize(input_query.lower())
  for token in tokenize_query:
      if token not in idf_dict:
          tokenize_query.remove(token)
  return tokenize_query

def compute_doc_norms(index, idf, n_docs):
  # initialization
  norms = np.zeros(n_docs)
  
  # compute norm of each document
  for w, l in index.items():
      if w in idf:
          for mbti, tf in l:
              calculation = math.pow((tf * idf[w]), 2)
              norms[mbti] += calculation
  
  return np.sqrt(norms)

def index_search(query, index, idf, doc_norms, mbti_dict):
  # query is already tokenized
  tf_q = Counter(query) 
  
  # initializations
  tfidf_query = [] # list containing tuples of terms and their respective tf-idf score for terms in thequery
  norm_sum = 0 # sum of (tf * idf)^2 of all words in the query
  
  # calculate tfidf of terms in query and compute norm_sum
  for term in tf_q:
    if term in idf:
      tfidf_score = idf[term] * tf_q[term]
      tfidf_query.append((term, tfidf_score))
      norm_sum += math.pow(tfidf_score, 2)
      print(term)
      print(tfidf_score)
  
  # # calculate norm of query
  # q_norm = math.sqrt(norm_sum)
    

  # initialization
  # dictionary of cosine similarity numerator     
  d = {}
    
  # compute cosine similarity numerator
  for term, score in tfidf_query:
      invidx_list = index[term]
      for mbti, tf in invidx_list:
          if mbti not in d:
              d[mbti] = score * (tf * idf[term])
          else:
              d[mbti] += score * (tf * idf[term])
    
  # initialization
  # list of tuples of cosine similarity score and doc
  results = []
    
  # compute cosine similarity    
  mbti_index = list(mbti_dict.keys())        
  for mbti in d:
      d_norm = doc_norms[mbti]
      # norm_product = q_norm * d_norm
      if d_norm != 0:
          sim = d[mbti] / d_norm
          results.append((sim, mbti_index[mbti]))
        
    
  return sorted(results, key = lambda x: (-x[0], x[1]))

def precompute():
  mbti = pd.read_csv('data/mbti.csv')
  tokenized = tokenize_mbti(mbti)
  mbti_dict = mbti_tokenized(tokenized)
  inv_idx = build_inverted_index(mbti_dict)
  idf = compute_idf(inv_idx, 16)
  doc_norms = compute_doc_norms(inv_idx, idf, 16)
  return (inv_idx, idf, doc_norms, mbti_dict)

def rank_mbtis(query, inv_idx, idf, doc_norms, mbti_dict):
  q = valid_query(query, idf)
  rankings = index_search(q, inv_idx, idf, doc_norms, mbti_dict)
  return rankings



