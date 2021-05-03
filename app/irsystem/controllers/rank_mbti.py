import re
import random
import numpy as np
import math
import pandas as pd
from collections import Counter
import json

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
  result = []
  tokenize_query = tokenize(input_query.lower())
  for token in tokenize_query:
      if token in idf_dict:
          result.append(token)
  return result

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

def index_search(query, index, idf, doc_norms, mbti_keys):
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
  
  # # calculate norm of query
  # q_norm = math.sqrt(norm_sum)
    

  # initialization
  # dictionary of cosine similarity numerator     
  d = {}
  words = {}
    
  # compute cosine similarity numerator
  for term, score in tfidf_query:
      invidx_list = index[term]
      for mbti, tf in invidx_list:
          if mbti not in d:
              scores = score * (tf * idf[term])
              d[mbti] = scores
              words[mbti] = [(scores, term)]
          else:
              d[mbti] += score * (tf * idf[term])
              words[mbti] += [(scores, term)]
    
  # initialization
  # list of tuples of cosine similarity score and doc
  results = []
    
  # compute cosine similarity    
  mbti_index = mbti_keys        
  for mbti in d:
      d_norm = doc_norms[mbti]
      # norm_product = q_norm * d_norm
      if d_norm != 0:
          sim = d[mbti] / d_norm
          w = words[mbti]
          sorted_words = sorted(w, key = lambda x: (-x[0]))
          sorted_words = [i[1] for i in sorted_words]
          results.append((sim, mbti_index[mbti], sorted_words))
        
  results = sorted(results, key = lambda x: (-x[0]))

  return [ (round(a,4),b,c) for (a,b,c) in results ]

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

def rank_movies(ranks, movie_index, updated_movie, mbti_keys):
  query_scores = np.zeros(len(mbti_keys))
  # place more weight on top 3 mbtis
  len_ranks = len(ranks)
  weight = 0.3
  for i in range(len(ranks)):
  #   if 
    (a,b,c) = ranks[i]
    ranks[i] = (a + weight, b, c)
    weight -= 0.1

  # calculate movie ranks
  for (score, mbti, words) in ranks:
    query_scores[mbti_keys.index(mbti)] = score
  movie_score = np.dot(updated_movie, query_scores)
  ranking_index = np.argsort((movie_score * -1))

  # determine movies
  movie_list = []
  for i in ranking_index:
    movie_list.append(movie_index[i])
  return movie_list

def get_characters(mbti_list, movie_list, character_dict):
  final = []
  for movie in movie_list:
    characters = []
    m = {}
    c = 3
    for (score, mbti) in mbti_list:
      movie_characters_dict = character_dict[movie]
      if (mbti in movie_characters_dict):
        if (len(movie_characters_dict[mbti]) <= 3):
          for character in movie_characters_dict[mbti]:
            ch = [character[0], mbti, character[1]]
            characters.append(ch)
            c = c-1
        else:
          ch = random.sample(movie_characters_dict[mbti], k = c)
          for x in ch:
            val = [x[0], mbti, x[1]]
            characters.append(val)
            c -= 1
      if c == 0:
        break
    m["movie"] = movie
    m["characters"] = characters
    final.append(m)
  return final

def rocchio_update(query, rel, nrel, idf, tf_idf, words_index, mbti_keys, a = 0.3, b = 0.8, c = 0.3):
  # with idf instead of tf-idf for now
  q = list(set(valid_query(query, idf)))
  q_index = [ words_index[word] for word in q]

  tf_q = Counter(q)

  q_vector = np.zeros(len(tf_idf[0]))
  for idx, term in enumerate(q):
    q_vector[words_index[term]]= tf_q[term] * idf[term]
  
  updated_q = np.zeros(len(q_vector))

  first = a * q_vector

  second = np.zeros(len(q_vector))
  if len(rel) != 0:
    for r in rel:
        rel_vector = tf_idf[mbti_keys.index(r)]
        second = np.add(second, rel_vector)
    second = (b / len(rel)) * second

  third = np.zeros(len(q_vector))
  if len(nrel) != 0:
      for nr in nrel:
          nrel_vector = tf_idf[mbti_keys.index(nr)]
          third = np.add(third, nrel_vector)
      third = (c / len(nrel)) * third

  # updated query
  updated_q = np.clip(first + second - third, a_min = 0, a_max = None)
  
  result = []
  #(term,tf_idf)
  for idx, term in enumerate(q):
    result.append((term, updated_q[q_index[idx]]))

  return result

  # for idx, term in enumerate(q):
  #   idf[term] = updated_q[idx]
  # json.dump(idf, open('data/idf.json', 'w'))


def rocchio_search(updated_query, index, idf, doc_norms, mbti_keys):

  # initializations
  tfidf_query = updated_query # list containing tuples of terms and their respective tf-idf score for terms in the query

  # initialization
  # dictionary of cosine similarity numerator     
  d = {}
  words = {}
    
  # compute cosine similarity numerator
  for term, score in tfidf_query:
      invidx_list = index[term]
      for mbti, tf in invidx_list:
          if mbti not in d:
              scores = score * (tf * idf[term])
              d[mbti] = scores
              words[mbti] = [(scores, term)]
          else:
              d[mbti] += score * (tf * idf[term])
              words[mbti] += [(scores, term)]
    
  # initialization
  # list of tuples of cosine similarity score and doc
  results = []
    
  # compute cosine similarity    
  mbti_index = mbti_keys        
  for mbti in d:
      d_norm = doc_norms[mbti]
      # norm_product = q_norm * d_norm
      if d_norm != 0:
          sim = d[mbti] / d_norm
          w = words[mbti]
          sorted_words = sorted(w, key = lambda x: (-x[0]))
          sorted_words = [i[1] for i in sorted_words]
          results.append((sim, mbti_index[mbti], sorted_words))
        
  results = sorted(results, key = lambda x: (-x[0]))

  return [ (round(a,4),b,c) for (a,b,c) in results ]

