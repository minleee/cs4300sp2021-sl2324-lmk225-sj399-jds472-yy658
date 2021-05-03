import re
import random
import numpy as np
import math
import pandas as pd
from collections import Counter
import json

def tokenize(text):
  """
  Returns a list of words that make up the text.

  Params: {text: String}

  Returns: List
  """
  # lowercase text and use regular expressions
  return re.findall('[a-z]+', text.lower())

def tokenize_mbti(df):
  """
  Returns a list of MBTIs and their tokenized words.

  Params: {df: DataFrame}

  Returns: List
  """
  # initialization
  tokenized = []

  # iterate through dataframe, tokenize text
  for idx, row in df.iterrows():
    t = row['type']
    text = tokenize(row['posts'])
    tokenized.append((t, text))

  return tokenized

def mbti_tokenized(tokenized):
  """
  Returns a dictionary with MBTI keys and tokenized text values.

  Params: {tokenized: List of Tuples}

  Returns: Dictionary
  """
  # initialization
  mbti_dict = {}

  # creates dictionary with MBTIs as keys with the tokenized text as values
  for (a, b) in tokenized:
    if a not in mbti_dict:
      mbti_dict[a] = b
    else:
      mbti_dict[a] += b

  return mbti_dict

def build_inverted_index(mbti_dict):
  """
  Returns a dictionary that represents the inverted index from the dictionary of
  MBTI keys and tokenized text values.

  Params: {mbti_dict: Dictionary}

  Returns: Dictionary
  """
  # initialization
  d = {}
  mbti_index = list(mbti_dict.keys())

  # created inverted index dictionary
  for mbti in mbti_dict:
    # use Counter to get tokenized words and frequencies
    tokenized = mbti_dict[mbti]
    count_tokenized = Counter(tokenized)
    # iterate through tokenized words and add to dictionary with frequencies
    for word in count_tokenized:
      if word in d:
        d[word].append((mbti_index.index(mbti), count_tokenized[word]))
      else:
        d[word] = [(mbti_index.index(mbti), count_tokenized[word])]

  return d

def compute_idf(inv_idx, n_docs, min_df=1, max_df_ratio=1.0):
  """
  Returns a dictionary with valid words as keys and their IDFs as values.

  Params: {inv_idx: Dictionary,
           n_docs: Integer,
           min_df: Integer,
           max_df_ratio: Float
          }

  Returns: Dictionary
  """
  # initialization
  idf_dict = {}
  
  # compute IDF for each word in inverted index
  for word in inv_idx:
    word_df = len(inv_idx[word]) 
    word_df_ratio = word_df / n_docs
    if word_df >= min_df and word_df_ratio < max_df_ratio:
      idf_dict[word] = math.log2(n_docs / (1 + word_df))
       
  return idf_dict

def valid_query(input_query, idf_dict):
  """
  Returns a list of valid tokenized words from the input query.

  Params: {input_query: String,
           idf_dict: Dictionary
          }

  Returns: List
  """
  # initialization
  result = []

  # tokenize query and create a list with valid tokenized words
  tokenize_query = tokenize(input_query.lower())
  for token in tokenize_query:
    if token in idf_dict:
      result.append(token)
  return result

def compute_doc_norms(index, idf, n_docs):
  """
  Returns a numpy array of the norms of each MBTI.

  Params: {index: Dictionary,
           idf: Dictionary,
           n_docs: Integer
          }

  Returns: Numpy Array
  """
  # initialization
  norms = np.zeros(n_docs)
  
  # compute the norm of each MBTI and add to numpy array
  for w, l in index.items():
    if w in idf:
      for mbti, tf in l:
        calculation = math.pow((tf * idf[w]), 2)
        norms[mbti] += calculation
  
  return np.sqrt(norms)

def index_search(query, index, idf, doc_norms, mbti_keys):
  """
  Returns a sorted list of a 3-element tuple, consisting of cosine similarity 
  score, MBTI, and relevant keywords. The tuples are sorted in descending order 
  of cosine similarity score.

  Params: {query: List,
           index: Dictionary,
           idf: Dictionary,
           doc_norms: Numpy Array,
           mbti_keys: List  
          }

  Returns: List
  """
  # query is already tokenized
  # count term frequencies of query tokens
  tf_q = Counter(query) 
  
  # initializations
  # list containing tuples of terms and their respective tf-idf score for 
  # terms in the query
  tfidf_query = []
  # sum of (tf * idf)^2 of all words in the query
  norm_sum = 0 
  
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

  # sort results   
  results = sorted(results, key = lambda x: (-x[0]))

  return [(round(a,4),b,c) for (a,b,c) in results]

def precompute():
  """
  Computes values that only need to be calculated once

  Returns: Tuple
  """
  mbti = pd.read_csv('data/mbti.csv')
  tokenized = tokenize_mbti(mbti)
  mbti_dict = mbti_tokenized(tokenized)
  inv_idx = build_inverted_index(mbti_dict)
  idf = compute_idf(inv_idx, 16)
  doc_norms = compute_doc_norms(inv_idx, idf, 16)

  # returns inverted index, IDFs, document norms, and mbti dict
  return (inv_idx, idf, doc_norms, mbti_dict)

def rank_mbtis(query, inv_idx, idf, doc_norms, mbti_dict):
  """
  Returns a list of tuples of MBTIs sorted in descending order based on cosine
  similarity score.

  Params: {query: String,
           inv_idx: Dictionary,
           idf: Dictionary,
           doc_norms: Numpy Array
           mbti_dict: Dictionary
          }

  Returns: List of Tuples
  """
  # tokenize valid words in query
  q = valid_query(query, idf)

  # get list of rankings based on cosine similarity scores
  rankings = index_search(q, inv_idx, idf, doc_norms, mbti_dict)

  return rankings

def rank_movies(ranks, movie_index, updated_movie, mbti_keys):
  """
  Returns list of movies with most similar character MBTI distributions as 
  rankings.

  Params: {ranks: List,
           movie_index: Numpy Text,
           updated_movie: Numpy Array,
           mbti_keys: List
          }

  Returns: List
  """
  # initialization
  query_scores = np.zeros(len(mbti_keys))

  # adjust weights to place more emphasis on top 3 ranks MBTIs
  len_ranks = len(ranks)
  weight = 0.3
  for i in range(len(ranks)):
    (a,b,c) = ranks[i]
    ranks[i] = (a + weight, b, c)
    weight -= 0.1

  # calculate movie rankings
  for (score, mbti, words) in ranks:
    query_scores[mbti_keys.index(mbti)] = score
  movie_score = np.dot(updated_movie, query_scores)

  # sort movie rankings in descending order
  ranking_index = np.argsort((movie_score * -1))

  # determine actual movie titles
  movie_list = []
  for i in ranking_index:
    movie_list.append(movie_index[i])

  return movie_list

def get_characters(mbti_list, movie_list, character_dict):
  """
  Returns list of characters from movies with most MBTIs.

  Params: {mbti_list: List,
           movie_list: List,
           character_dict: Dictionary
          }
          
  Returns: List
  """
  # initialization
  final = []

  # iterate through movies to get top 3 characters for each movie
  for movie in movie_list:
    # initialization
    characters = []
    m = {}
    c = 3
    # return top 3 characters in movie with similar MBTIs
    for (score, mbti) in mbti_list:
      movie_characters_dict = character_dict[movie]
      if (mbti in movie_characters_dict):
        # return all characters in the movie with MBTI
        if (len(movie_characters_dict[mbti]) <= 3):
          for character in movie_characters_dict[mbti]:
            ch = [character[0], mbti, character[1]]
            characters.append(ch)
            c = c - 1
        else:
          # randomly select 3 characters of MBTI in movie
          ch = random.sample(movie_characters_dict[mbti], k = c)
          for x in ch:
            val = [x[0], mbti, x[1]]
            characters.append(val)
            c = c - 1
      # stop when 3 characters have been found
      if c == 0:
        break
    
    # keep movie and characters together
    m["movie"] = movie
    m["characters"] = characters
    final.append(m)

  return final

def rocchio_update(query, rel, nrel, idf, tf_idf, words_index, mbti_keys, 
  a = 0.3, b = 0.8, c = 0.3):
  """
  Returns list of updated query terms and weights after implementing the 
  Rocchio algorithm.

  Params: {query: String,
           rel: List,
           nrel: List,
           idf: Dictionary,
           tf_idf: Numpy Array,
           words_index: Dictionary,
           mbti_keys: List
          }
          
  Returns: List
  """
  # initialization
  # tokenize query and convert to set to remove duplicates
  # convert to list to get final list of tokens from query
  q = list(set(valid_query(query, idf)))

  # contains index of tokenized terms in query
  q_index = [words_index[word] for word in q]

  # compute term frequencies of terms in query
  tf_q = Counter(q)

  # calculate vector of original query
  q_vector = np.zeros(len(tf_idf[0]))
  for idx, term in enumerate(q):
    q_vector[words_index[term]]= tf_q[term] * idf[term]
  
  # initialize updated query vector
  updated_q = np.zeros(len(q_vector))

  # adjust weight of original query vector
  first = a * q_vector

  # compute second query vector to account for relevant MBTIs
  second = np.zeros(len(q_vector))
  if len(rel) != 0:
    for r in rel:
        rel_vector = tf_idf[mbti_keys.index(r)]
        second = np.add(second, rel_vector)
    second = (b / len(rel)) * second

  # compute third query vector to account for nonrelevant MBTIs
  third = np.zeros(len(q_vector))
  if len(nrel) != 0:
      for nr in nrel:
          nrel_vector = tf_idf[mbti_keys.index(nr)]
          third = np.add(third, nrel_vector)
      third = (c / len(nrel)) * third

  # compute updated query vector values
  updated_q = np.clip(first + second - third, a_min = 0, a_max = None)
  
  # initializion
  result = []
  # append to list a tuple of (term, tf_idf) for updated query vector
  for idx, term in enumerate(q):
    result.append((term, updated_q[q_index[idx]]))

  return result

def rocchio_search(updated_query, index, idf, doc_norms, mbti_keys):
  """
  Returns list of updated MBTI rankings after updating query vector from Rocchio
  algorithm.

  Params: {updated_query: List,
           index: Dictionary,
           idf: Dictionary,
           doc_norms: Numpy Array,
           mbti_keys: List
          }
          
  Returns: List
  """
  # initialization
  # list containing tuples of terms and their respective tf-idf score 
  # for terms in the query
  tfidf_query = updated_query 

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

  # sort results in descending order based on cosine similarity score   
  results = sorted(results, key = lambda x: (-x[0]))

  return [(round(a,4),b,c) for (a,b,c) in results]

