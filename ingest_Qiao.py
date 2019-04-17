from elasticsearch import Elasticsearch
import datetime
import re
import requests, os, sys
es = Elasticsearch([{'host': 'localhost', 'port': '9200'}])

def clean_timestamp(date):
	clean_date = re.sub("[^0-9]","",str(date))
	return clean_date[:6]
timestamp = clean_timestamp(datetime.datetime.now())
date_now = datetime.datetime.now()
index_name = "news_{" + timestamp + "}"
source_url = "source_url.txt"

URLS = list(open(source_url, mode = 'r').readlines())

for url in URLS:
	raw_news= os.popen('python news_scraping.py '+url).read()
	if es.indices.exists(index = index_name):
		doc = {"Language":"English", "Raw_content":raw_news,"Clean_content":"", "Categorization":[], "Link":url, "Continent":"NorthAmerica","Country":"US","State":""}
		es.index(index = index_name, ignore=400, doc_type="type_name",body=doc)
	else:
		mapping = {"index_patterns":"news_*",
				"settings": {"number_of_shards": 1,"number_of_replicas": 0,	
				"index": {"query": {"default_field": "id"}}},
				"mappings": {"type_name" : {"properties": 
				{"@Language": {"type": "text"},"@Raw_content": {"type": "text"},"@Clean_content": {"type": "text"},
				"@Categorization": {"properties":{"name":{"type":"text"},"key":{"type":"text"},"value":{"type":"float"}}},
				"@created_at": {"type": "date","format": "yyyy-MM-dd HH:mm:ss"},"@updated_at": {"type": "date","format": "yyyy-MM-dd HH:mm:ss"},
				"@Link": {"type": "text"},"@Continent": {"type": "text"},"@Country": {"type": "text"},"@State": {"type": "text"}}}}}
		es.indices.create(index=index_name,ignore=400,body=mapping)
		doc = {"Language":"English", "Raw_content":raw_news,"Clean_content":"","created_at":date_now, "Categorization":[], "Link":url, "Continent":"NorthAmerica","Country":"US","State":"","updated_at":""}
		es.index(index = index_name, ignore=400, doc_type="type_name",body=doc)