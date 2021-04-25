from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

# import
import re
import numpy as np
import pandas as pd
import random
import json
import app.irsystem.controllers.rank_mbti as m

project_name = "Mo_Bie_TI"
net_id = "Justin Sun (jds472), Lydia Kim (lmk225), Sohwi Jung (sj399), Seungmin Lee (sl2324), Grace Yim (yy658)"

inv_idx = json.load( open( "data/inv_idx.json" ) )
idf = json.load( open( "data/idf.json" ) )
character_dict = json.load( open( "data/character_dict.json" ) )
doc_norms = np.loadtxt('data/doc_norms.txt')
movie_index = np.loadtxt('data/movie_index.txt', delimiter='\n', dtype=str, comments=None)
updated_movie = np.loadtxt('data/updated_movie.txt')
mbti_keys = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP', 'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search')

	output_message = ''
	print(query)
	if not query:
		data = []
		output_message = ''
		print('no query')
		return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)
	else:
		output_message = "Your search: " + query
		rankings = m.rank_mbtis(query, inv_idx, idf, doc_norms, mbti_keys)
		# print(rankings)
		movies = m.rank_movies(rankings, movie_index, updated_movie, mbti_keys)
		if rankings != [] and rankings[0][0] !=0:
			top_mbti = rankings[:5]
			top_5 = movies[:5]
			combined = m.get_characters(top_mbti, top_5, character_dict)
			top_5 = combined[:5]
		# 	s = sum([pair[0] for pair in rankings])
		# 	for idx, (a,b) in enumerate(rankings):
		# 		a = (a / s) * 100
		# 		a = round(a, 2)
		# 		a = str(a) + '%'
		# 		rankings[idx] = (a, b)
		# 	top_5 = rankings[:5]
		else:
			top_5 = ['No results. Please try again.']


	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=top_5)



