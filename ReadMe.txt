Prepare File:
	news_scraping.py
			  -- Web scraping script
			  -- python news_scraping.py URL
	
	data
			  -- six folders
			  -- Store news contents scraped from URLs
			  -- One folder for each category

	glove.6B.100d_pickle 
						 -- GloVe
						 -- The pretrained vector with 100 dimension from Stanford
						 -- https://nlp.stanford.edu/projects/glove/

	glove.6B.100d.txt 
					  -- Original file of glove.6B.100d_pickle

	prepare4train.py 
					 -- Import clean_helper.py 
					 -- Call load_data_and_labels_another function
					 -- data_pickle is the output
	
	clean_helper.py 
					-- A bunch of functions
					-- Clean raw news data
					-- Label news with vectors
					-- Define batch iterator function for train.py
					-- Dump results to data_pickle
          -- For retrain, please change the folder names in line 94

Training Files:
	data 
			  -- Files are input for clean_helper.py

	word_embedding.py 
					  -- Import data_pickle
					  -- Use number to represent words in news according to data in glove.6B.100d_pickle
					  -- Return vectors
					  -- Dump results to word_index_pickle, glove.6B.100d_pickle
	
	text_cnn.py 
				 -- Define the CNN for train.py

	train.py 
			 -- training script
			 -- Please change the 'data_dir' and define your path.
			 -- Please change the directory for saved model in line 220.
			 -- Please update word_index_pickle_train, word_index_pickle_valid and word_index_pickle_test if retrain
			 -- In shell, type "python train.py"

Classify Data/Article:
	glove.6B.100d_pickle
						 -- Needed
	
	my_model_probability
						 -- Model saved from training process
             -- Clean the folder if retrain model
	
	prepare.py
			   -- Process raw data for classify_news.py
			   -- Called by classify_news.py

	classify_news.py
					 -- Take in raw news content (string)
					 -- Import prepare.py
					 -- Process news content
					 -- Predict the probability
					 -- Print out the result. (can be changed)

Retrain:
      -- Update "data"
      -- Update category names in clean_helper.py
      -- python prepare4train.py
      -- python word_embedding.py
      -- python train.py


ElasticSearch Template:
code:***
PUT /news_{201808}
{
  "index_patterns":"news_*",
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "index": {
      "query": {
        "default_field": "id"
      }
    }
  },
  "mappings": {
    "type_name" : {
      "properties": {
        "language": {
          "type": "text"
        },
        "raw_content":{
          "type": "text"
        },
        "clean_content":{
          "type": "text"
        },
        "categorization":{
          "type" : "nested",
          "properties": [{
            "classify_name":{
              "type": "text"
            },
            "prediction_value":{
              "type":"nested",
              "properties":[{
              "name":{
                "type": "text"
              },
              "value":{
                "type":"float"
              }
          }]} 
          }]
        },
        "created_at":{
          "type": "date",
          "format": "yyyy-MM-dd HH:mm:ss"
        },
        "classified_at":{
          "type": "date",
          "format": "yyyy-MM-dd HH:mm:ss"
        },
        "updated_at":{
          "type": "date",
          "format": "yyyy-MM-dd HH:mm:ss"
        },
        "link":{
          "type": "text"
        },
        "continent":{
          "type": "text"
        },
        "country":{
          "type": "text"
        },
        "state":{
          "type": "text"
        }
      }
    }
  }
}
***

