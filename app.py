from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from datetime import datetime
import re
import json
from services.preprocessing import Preprocessing
from services.etm import EmbeddedTopicModeling as ETM
from services.llm import Llm
from models.tweet import Tweet
from models.topics import Topics
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/")
def index():
	return render_template('index.html')

@app.route("/topic")
def result():
	keyword = request.args.get('keyword')
	# period = request.args.get('period').split(' - ')
	# start_date = datetime.strptime(period[0], "%m/%d/%Y")
	# start_date = start_date.strftime("%Y-%m-%d")
	# end_date = datetime.strptime(period[1], "%m/%d/%Y")
	# end_date = end_date.strftime("%Y-%m-%d")
	print("Fungsi Ambil Tanggal Dimulai")
	start_date = request.args.get('start_date')
	end_date = request.args.get('end_date')
	tweets = Tweet.getTweetByKeyword(keyword, start_date, end_date)
	
	if len(tweets) <= 0:
		return jsonify({ 
			"status" : 422, 
			"message" : "Mohon Maaf, tidak ada data untuk kata kunci dan rentang waktu tersebut",
		})

	dataTweetText = []
	dataTweet = []
	for tweet in tweets:
		dataTweet.append(tweet)
		dataTweetText.append(tweet['full_text'])
	print("Memulai Preprocessing")
	# data = Preprocessing(dataTweetText,keyword).run_preprocessing()
	print("Memulai Modeling")
	etm = ETM()
	etm_model = etm.generateTopic()
	num_topics = etm_model[0]
	topics = etm_model[2]['topics']
	
	# topic_res = []
	# for topic_id, topic in topics:
	# 	topic_res.append([word for word, _ in topic])

	context = Llm.getContext(topics, keyword, num_topics)
	documents_prob = etm.document(dataTweet, etm_model)
	
	res = { 
		"status" : 200, 
		"message" : "Data Topics",
		"data": {
			"topic": topics,
			"context": context['context'],
			"interpretation": context['interpretation'],
			"documents_topic": documents_prob,
		}, 
	} 
	
	return render_template('result.html', context=context, topic=topics)
	# return jsonify(res)

@app.route("/topic-by-project/<string:projectId>", methods=['GET'])
def get_topic_by_project(projectId):
    topic = Topics.getTopicByProjectId(projectId)
    data = {
        "status": 200,
        "message": "Data Topics",
        "data": topic
    }
    return jsonify(data)

@app.route("/document-by-project/<string:projectId>", methods=['GET'])
def get_document_by_project(projectId):
	topic = request.args.get('topic')
	document_topic = Topics.getDocumentTopicByProjectId(projectId, topic)
	data = {
        "status": 200,
        "message": "Data Documents",
        "data": document_topic
    }
	return jsonify(data)

@app.route("/rag-topic/<string:projectId>", methods=['GET'])
def rag_topic(projectId):
  
		topic = Topics.getContextTopicByProjectId(projectId)
    
		document = Topics.getDocumentTopicByProjectId(projectId)
    		
		# Menyiapkan data untuk dikembalikan
		data = {
        "status": 200,
        "message": "Data Topics",
        "data": {
            "keyword": topic[0]['keyword'],
            "topic": topic,
            "documents_topic": document,
        }
    }

		return jsonify(data)

def start_app():
    from dotenv import load_dotenv
    import os
    load_dotenv()
    app_port = int(os.getenv('APP_PORT', 6000))
    app_debug = os.getenv('APP_DEBUG', 'True') == 'True'
    app.run(debug=app_debug, port=app_port)