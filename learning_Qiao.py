from elasticsearch import Elasticsearch
import datetime
import re
import requests, os, sys
import prepare
import tensorflow as tf
import numpy as np


es = Elasticsearch([{'host': 'localhost', 'port': '9200'}])
index_name = "news_{201808}"
result = es.search(index=index_name, doc_type="type_name", body = {"query": {"term" : {"Clean_content.keyword":""}}}, size = 10000, _source = ("Raw_content"))


with tf.Session() as sess:
	# restore the CNN model
	saver = tf.train.import_meta_graph('./my_model_probability/my_model.ckpt.meta')
	saver.restore(sess, tf.train.latest_checkpoint('./my_model_probability'))
	graph = tf.get_default_graph()
	# access and create placeholders and feed-dict to feed new data
	input_x = graph.get_tensor_by_name("input_x:0")
	op_to_restore = graph.get_tensor_by_name("predictions:0")
	prob = graph.get_tensor_by_name("predictions_b:0")
	dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")

	for data in result['hits']['hits']:
		raw_news = data['_source']['Raw_content']
		ID= data['_id']
		print(ID)
		cleandata = prepare.load_data_and_labels_another(raw_news)
		if cleandata is not None:
			transdata = prepare.word_id_convert(cleandata)
			# convert the data into a (1,300) shape.
			a = transdata.reshape(-1,1)
			tdata = np.transpose(a)
			feed_dict = {input_x:tdata, dropout_keep_prob: 0.5}
			#sess.run(tf.global_variables_initializer())
			predictions = sess.run(op_to_restore, feed_dict)    
			predictions_b = sess.run(prob, feed_dict)
			# print(predictions)
			# print(predictions_b)
			date_now = datetime.datetime.now()
			query={"doc":{"updated_at":date_now,"Clean_content":cleandata}}
			es.update(index=index_name, doc_type="type_name", id=ID, ignore=400, body=query)
			# The sequence of topics strictly obey the sequence of label in clean_helper.py
			topics =["Financial","International","Legal","Social","Tech,Sci&Innov","Hemp"]
			tem = []
			for idx, topic in enumerate(topics):
				probability = predictions_b[0][idx]
				probability = round(probability,2)
				tem.append([topic, probability])
			for itm in tem:
				query = {"script": {"lang": "painless", "inline": "ctx._source.Categorization.add(params.newitem)","params":{"newitem":{"name":"insight_daily","key":'{}'.format(itm[0]),"value":'{}'.format(itm[1])}}}}
				es.update(index = index_name, doc_type = "type_name", id = ID, ignore = 400, body = query)
			
	

