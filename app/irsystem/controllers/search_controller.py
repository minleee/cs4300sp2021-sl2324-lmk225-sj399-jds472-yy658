from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

# import
import re
import numpy as np
import pandas as pd
import json
import app.irsystem.controllers.rank_mbti as m

project_name = "Mo_Bie_TI"
net_id = "Justin Sun (jds472), Lydia Kim (lmk225), Sohwi Jung (sj399), Seungmin Lee (sl2324), Grace Yim (yy658)"
# inv_idx = np.load('data/inv_idx.npy',allow_pickle='TRUE').item()
# idf = np.load('data/idf.npy',allow_pickle='TRUE').item()
# doc_norms = np.loadtxt('data/doc_norms.txt', dtype=int)
# mbti_dict = np.load('data/mbti_dict.npy',allow_pickle='TRUE').item()
# inv_idx, idf, doc_norms, mbti_dict = m.precompute()
inv_idx = json.load( open( "data/inv_idx.json" ) )
idf = json.load( open( "data/idf.json" ) )
doc_norms = np.loadtxt('data/doc_norms.txt', dtype=int)
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
		# inv_idx = json.load( open( "data/inv_idx.json" ) )
		# idf = json.load( open( "data/idf.json" ) )
		# doc_norms = np.loadtxt('data/doc_norms.txt', dtype=int)
		# mbti_dict = json.load( open( "data/mbti_dict.json" ) )
		output_message = "Your search: " + query
		rankings = m.rank_mbtis(query, inv_idx, idf, doc_norms, mbti_keys)
		if rankings != [] and rankings[0][0] !=0:
			s = sum([pair[0] for pair in rankings])
			for idx, (a,b) in enumerate(rankings):
				a = (a / s) * 100
				a = round(a, 2)
				a = str(a) + '%'
				rankings[idx] = (a, b)
			top_5 = rankings[:5]
		else:
			top_5 = ['No results. Please try again.']

	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=top_5)



