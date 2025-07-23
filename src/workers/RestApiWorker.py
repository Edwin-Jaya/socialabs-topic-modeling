from multiprocessing.connection import Connection
from flask import Flask, request, jsonify
from flask_classful import FlaskView, route
import threading
import uuid
import asyncio
import time
import utils.log as log
from utils.handleMessage import sendMessage, convertMessage


from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


class RestApiWorker(FlaskView):
    ###############
    # dont edit this part
    ###############
    route_base = "/"
    conn: Connection
    requests: dict = {}

    def __init__(self):
        # we'll assign these in run()
        self._port: int = None

        self.requests: dict = {}

    def run(self, conn: Connection, port: int):
        # assign here
        RestApiWorker.conn = conn
        self._port = port

        RestApiWorker.register(app)

        # start background threads *before* blocking server
        threading.Thread(target=self.listen_task, daemon=True).start()
        threading.Thread(target=self.health_check, daemon=True).start()

        app.run(debug=True, port=self._port, use_reloader=False)
        # asyncio.run(self.listen_task())
        self.health_check()

    def health_check(self):
        """Send a heartbeat every 10s."""
        while True:
            sendMessage(
                conn=RestApiWorker.conn,
                messageId="heartbeat",
                status="healthy"
            )
            time.sleep(10)

    def listen_task(self):
        while True:
            try:
                # Check for messages with 1 second timeout
                if RestApiWorker.conn.poll(1):
                    raw = RestApiWorker.conn.recv()
                    msg = convertMessage(raw)
                    self.onProcessed(raw)
            except EOFError:
                break
            except Exception as e:
                log(f"Listener error: {e}", 'error')
                break

    def onProcessed(self, msg: dict):
        """
        Called when a worker response comes in.
        msg must contain 'messageId' and 'data'.
        """
        task_id = msg.get("messageId")
        entry = RestApiWorker.requests[task_id]
        if not entry:
            return
        entry["response"] = msg.get("data")
        entry["event"].set()

    def sendToOtherWorker(self, destination: str, data):
        task_id = str(uuid.uuid4())
        evt = threading.Event()

        RestApiWorker.requests[task_id] = {
            "event": evt,
            "response": None
        }
        print(f"Sending request to {destination} with task_id: {task_id}")

        sendMessage(
            conn=RestApiWorker.conn,
            messageId=task_id,
            status="processing",
            destination=destination,
            data=data
        )
        if not evt.wait(timeout=10):
            # timeout
            return {
                "taskId": task_id,
                "status": "timeout",
                "result": None
            }

        # success
        result = RestApiWorker.requests.pop(task_id)["response"]
        return {
            "taskId": task_id,
            "status": "completed",
            "result": result
        }

    ##########################################
    # FLASK ROUTES FUNCTIONS
    ##########################################
    @route('/', methods=['GET'])
    def getData(self):
        projectId = request.args.get('projectId')
        response = self.sendToOtherWorker(
            destination=[f"DatabaseInteractionWorker/getData/{projectId}"],
            data=projectId
        )
        if response["status"] == "timeout":
            return jsonify({"error": "Request timed out"}), 504
        elif response["status"] == "completed":
            return jsonify(response["result"]), 200
        else:
            return jsonify({"error": "Unknown error"}), 500

    @route('/topic', methods=['POST'])
    def result(self):

        projectId = request.json.get('projectId')
        keyword = request.args.get('keyword')

        prompt = request.json.get('prompt')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
    #   tweets = Tweet.getTweetByKeyword(keyword, start_date, end_date)

        tweets = self.sendToOtherWorker(
            destination=[f"DatabaseInteractionWorker/getTweetByKeyword/"],
            data={
                "keyword": "asd",
                "start_date": "20250",
                "end_date": "2025-01-01"
            }
        )

        if tweets["status"] == "timeout":
            return jsonify({"error": "Request timed out"}), 504
        elif tweets["status"] == "completed":
            dataTweetText = []
            dataTweet = []
            for tweet in tweets:
                dataTweet.append(tweet)
                dataTweetText.append(tweet['full_text'])
            print("Memulai Preprocessing")
            data = self.sendToOtherWorker(
                destination=[f"Preprocessing"],
                data={
                    "dataTweetText":dataTweetText,
                    "keyword":keyword
                }
            ) 
            # Bikin data temp buat di training
            data = self.sendToOtherWorker(
                destination=[f"preprocessingWorker/run_preprocessing/"]
            )
            etm_model = self.sendToOtherWorker(
                destination=[f"etmWorker/generateTopic/"]
            )
            num_topics = etm_model[0]
            topics = etm_model[2]['topics']
            
            # topic_res = []
            # for topic_id, topic in topics:
            # 	topic_res.append([word for word, _ in topic])

            context = self.sendToOtherWorker(
                destination=[f"llmWorker/getContext/"],
                data={
                    "topics":topics,
                    "keyword":keyword,
                    "num_topics":num_topics
                }
            ) 

            documents_prob = self.sendToOtherWorker(
                destination=[f"etmWorker/document"]
                data={
                    "dataTweet":dataTweet,
                    "etm_model":etm_model
                }
            ) 
	
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
	
            # return jsonify(tweets["result"]), 200
            return jsonify(res), 200
        else:
            return jsonify({"error": "Unknown error"}), 500

    @app.route("/topic-by-project/<string:projectId>", methods=['GET'])
    def get_topic_by_project(self, projectId):
        topic = self.sendToOtherWorker(
            destination=[f"DatabaseInteractionWorker/getTopicByProjectId"],
            data = {
                "projectId":projectId
            }
        )
        data = {
            "status": 200,
            "message": "Data Topics",
            "data": topic
        }
        return jsonify(data)
    
    @app.route("/document-by-project/<string:projectId>", methods=['GET'])
    def get_document_by_project(self, projectId):
        topic = request.args.get('topic')
        document_topic = self.sendToOtherWorker(
            destination=[f"DatabaseInteractionWorker/getDocumentTopicByProjectId"],
            data = {
                "projectId":projectId,
                "topic":topic
            }
        )
        data = {
            "status": 200,
            "message": "Data Documents",
            "data": document_topic
        }
        return jsonify(data)

    @app.route("/rag-topic/<string:projectId>", methods=['GET'])
    def rag_topic(self, projectId):
    
        topic = self.sendToOtherWorker(
            destination=[f"DatabaseInteractionWorker/getContextTopicByProjectId"],
            data = {
                "projectId":projectId
            }
        )
        
        document = self.sendToOtherWorker(
            destination=[f"DatabaseInteractionWorker/getDocumentTopicByProjectId"],
            data = {
                "projectId":projectId
            }
        )
                
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

def main(conn: Connection, config: dict):

    worker = RestApiWorker()
    worker.run(conn, config.get("port", 5000))
