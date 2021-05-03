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
tf_idf = np.loadtxt('data/tf_idf.txt')
words_index = json.load( open( "data/words_index.json" ) )
mbti_keys = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP', 'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search')

	output_message = ''
	if query:
		output_message = "Your search: " + query
		rankings = m.rank_mbtis(query, inv_idx, idf, doc_norms, mbti_keys)
		movies = m.rank_movies(rankings.copy(), movie_index, updated_movie, mbti_keys)
		top_5 = []
		if rankings != [] and rankings[0][0] !=0:
			top_mbti = [(i[0], i[1]) for i in rankings][:5]
			# top words in query 
			top_words = [i[2] for i in rankings][:5]
			top_5 = movies[:5]
			combined = m.get_characters(top_mbti, top_5, character_dict)
			top_5 = (rankings[:5], combined[:5], top_words)

			s = sum([pair[0] for pair in rankings])
			for idx, (a,b,c) in enumerate(rankings):
				a = (a / s) * 100
				a = round(a, 2)
				a = str(a) + '%'
				rankings[idx] = (a, b,c)
			top_5 = (rankings[:5], combined[:5], top_words)
	elif request.args.get('radio_1'):
		rel = []
		nrel = []
		for i in range(1,6):
			if 'relevant' == request.args.get('radio_' + str(i))[:8]:
				rel.append(request.args.get('radio_' + str(i))[-4:])
			else:
				nrel.append(request.args.get('radio_' + str(i))[-4:])

		query = request.args.get('rocchio-selected')
		query = query[13:]
		output_message = "Your search: " + query
		updated_query = m.rocchio_update(query, rel, nrel, idf, tf_idf, words_index, mbti_keys)
		rankings = m.rocchio_search(updated_query, inv_idx, idf, doc_norms, mbti_keys)
		movies = m.rank_movies(rankings.copy(), movie_index, updated_movie, mbti_keys)
		top_5 = []
		if rankings != [] and rankings[0][0] !=0:
			top_mbti = [(i[0], i[1]) for i in rankings][:5]
			# top words in query 
			top_words = [i[2] for i in rankings][:5]
			top_5 = movies[:5]
			combined = m.get_characters(top_mbti, top_5, character_dict)
			top_5 = (rankings[:5], combined[:5], top_words)

			s = sum([pair[0] for pair in rankings])
			for idx, (a,b,c) in enumerate(rankings):
				a = (a / s) * 100
				a = round(a, 2)
				a = str(a) + '%'
				rankings[idx] = (a, b,c)
			top_5 = (rankings[:5], combined[:5], top_words)
	else:
		data = []
		output_message = ''
		# print('no query')
		return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)


	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=top_5)


@irsystem.route('/', methods=['GET'], defaults={'path': ''})
@irsystem.route('/<path:path>')
def index(path):
    return render_template("about.html")



