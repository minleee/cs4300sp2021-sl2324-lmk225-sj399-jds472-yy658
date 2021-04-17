from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

# import
import re
import numpy as np
import pandas as pd
import app.irsystem.controllers.rank_mbti as m

project_name = "Mo_Bie_TI"
net_id = "Justin Sun (jds472), Lydia Kim (lmk225), Sohwi Jung (sj399), Seungmin Lee (sl2324), Grace Yim (yy658)"
mbti_dict, idf_dict, doc_norms, tf_matrix, words_to_analyze = m.precompute()

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
		rankings = m.rank_mbtis(mbti_dict, query, idf_dict, doc_norms, tf_matrix, words_to_analyze)
		s = sum([pair[0] for pair in rankings])
		for idx, (a,b) in enumerate(rankings):
			a = (a / s) * 100
			a = round(a, 2)
			a = str(a) + '%'
			rankings[idx] = (a, b)
		top_5 = rankings[:5]
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=top_5)



