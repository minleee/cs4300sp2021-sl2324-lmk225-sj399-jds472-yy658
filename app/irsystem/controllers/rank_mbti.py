import re
import numpy as np
import math
import pandas as pd
from collections import Counter

def tokenize(text):
  """Returns a list of words that make up the text.

  Note: for simplicity, lowercase everything.
  Requirement: Use regular expressions to satisfy this function

  Params: {text: String}
  Returns: List
  """
  # fix regular expression to remove links
  return re.findall('[a-z]+', text.lower())

def tokenize_mbti(df):
  """Returns a list of tokenized words.
  Params: {df: Datafraame}
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

def word_count(mbti_dict):
  # returns the words as keys and the mbti counts as values
  word_dict = {}
  for key in mbti_dict:
    word_set = set(mbti_dict.get(key))
    for word in word_set:
        if word not in word_dict:
            word_dict[word] = 1
        else:
            word_dict[word] += 1
  return word_dict

def compute_idf(word_dict, n_docs, min_df=1, max_df_ratio=1):
  # return idf score of each word
    # initialization
    idf_dict = {}
    
    # compute IDF
    for word in word_dict:
        word_df = word_dict[word] 
        word_df_ratio = word_df / n_docs
        if word_df >= min_df and word_df_ratio < max_df_ratio:
            idf_dict[word] = math.log2(n_docs / (1 + word_df))
            
    return idf_dict

def output_words_to_analyze(input_word_counts):
  """Returns a list of words to analyze in alphabetically sorted order
      Params: {input_word_counts: Dict}
      Returns: List
  """
  # YOUR CODE HERE
  analyze_list = []
  for word in input_word_counts:
      if input_word_counts.get(word) > 1:
          analyze_list.append(word)
  return sorted(analyze_list)

def create_tfmatrix(mbti_dict, words_to_analyze):
  # creates tf matrix of terms in dataset
  # each row is the mbti, each column is a word to be analyzed
  tf_matrix = np.zeros((len(mbti_dict), len(words_to_analyze)))
  mbti_keys = list(mbti_dict.keys())

  for mbti in mbti_dict:
    c = Counter(mbti_dict[mbti])
    for index, word in enumerate(words_to_analyze):
        count = c[word]
        tf_matrix[mbti_keys.index(mbti)][index] = count
  return tf_matrix

def valid_query(input_query, words_to_analyze):
  # check if query is valid
  tokenize_query = tokenize(input_query.lower())
  for token in tokenize_query:
      if token not in words_to_analyze:
          tokenize_query.remove(token)
  return tokenize_query

def compute_doc_norms(tf_matrix, idf_dict, words_to_analyze, n_docs=16):
  # calculate doc norms
  norms = np.zeros(n_docs)   
  # compute norm of each document
  for i in range(len(tf_matrix)):
      for j in range(len(tf_matrix[i])):
          if words_to_analyze[j] in idf_dict:
                  calculation = math.pow((tf_matrix[i][j] * idf_dict[words_to_analyze[j]]), 2)
                  norms[i] += calculation
  return np.sqrt(norms)

def cosine_score(mbti_dict, tokenized_query, idf_dict, doc_norms, tf_matrix, words_to_analyze): 
  # tf of terms in query
  tf_q = Counter(tokenized_query) 
    
  # initializations ('hi', 0.008)
  tfidf_query = [] # list containing tuples of terms and their respective tf-idf score for terms in thequery
  norm_sum = 0 # sum of (tf * idf)^2 of all words in the query
    
  # calculate tfidf of terms in query and compute norm_sum
  for term in tf_q:
      if term in idf_dict:
          tfidf_score = idf_dict[term] * tf_q[term]
          tfidf_query.append((term, tfidf_score))
          norm_sum += math.pow(tfidf_score, 2)
  
  # calculate norm of query
  q_norm = math.sqrt(norm_sum)
    

  # initialization
  # dictionary of cosine similarity numerator     
  d = {}
  
  # compute cosine similarity numerator
  for term, score in tfidf_query:
      index = words_to_analyze.index(term)
      for doc_id in range(len(tf_matrix)):
          if doc_id not in d:
              d[doc_id] = score * (tf_matrix[doc_id][index] * idf_dict[term])
          else:
              d[doc_id] += score * (tf_matrix[doc_id][index] * idf_dict[term])
  
  # initialization
  # list of tuples of cosine similarity score and doc
  results = []
  
  # compute cosine similarity
  mbti_keys = list(mbti_dict.keys())         
  for doc_id in d:
      d_norm = doc_norms[doc_id]
      norm_product = q_norm * d_norm
      if norm_product != 0:
          sim = d[doc_id] / norm_product
          results.append((sim, mbti_keys[doc_id]))
      
    
  return sorted(results, key = lambda x: (-x[0], x[1]))

def precompute():
  mbti = pd.read_csv('data/mbti.csv')
  tokenized = tokenize_mbti(mbti)
  mbti_dict = mbti_tokenized(tokenized)
  word_dict = word_count(mbti_dict)
  idf_dict = compute_idf(word_dict, 16)
  words_to_analyze = output_words_to_analyze(word_dict)
  tf_matrix = create_tfmatrix(mbti_dict, words_to_analyze)
  doc_norms = compute_doc_norms(tf_matrix, idf_dict, words_to_analyze)
  return (mbti_dict, idf_dict, doc_norms, tf_matrix, words_to_analyze)


def rank_mbtis(mbti_dict, query, idf_dict, doc_norms, tf_matrix, words_to_analyze):
  q = valid_query(query, words_to_analyze)
  rankings = cosine_score(mbti_dict, q, idf_dict, doc_norms, tf_matrix, words_to_analyze)
  return rankings



