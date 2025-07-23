from multiprocessing.connection import Connection

from pymongo import MongoClient
import asyncio
from utils.log import log
from utils.handleMessage import sendMessage
import time
import datetime
import re

from .Worker import Worker


class DatabaseInteractionWorker(Worker):
    #################
    # dont edit this part
    ################
    _instanceId: str
    _isBusy: bool = False
    _client: MongoClient
    _db: str
    _dbtweet: str

    def __init__(self, conn: Connection, config: dict):
        self.conn = conn
        self._db = config.get("database", "mydatabase")
        self._dbtweet = config.get("database", "tweet_db")
        self.connection_string = config.get(
            "connection_string", "mongodb://localhost:27017/")

    def run(self) -> None:
        self._instanceId = "DatabaseInteractionWorker"
        self._client = MongoClient(self.connection_string)
        self._db = self._client[self._db]
        self._dbtweet = self._client[self._dbtweet]
        if not self._client:
            log("Failed to connect to MongoDB", "error")
        log(f"Connected to MongoDB at {self.connection_string}", "success")
        asyncio.run(self.listen_task())
        self.health_check()

    def health_check(self) -> None:
        while True:
            pass
            sendMessage(conn=self.conn, messageId=self._instanceId,
                        status="healthy")
            time.sleep(10)

    async def listen_task(self) -> None:
        while True:
            try:
                if self.conn.poll(1):  # Check for messages with 1 second timeout
                    message = self.conn.recv()
                    dest = [
                        d
                        for d in message["destination"]
                        if d.split("/", 1)[0] == "DatabaseInteractionWorker"
                    ]
                    # dest = [d for d in message['destination'] if d ='DatabaseInteractionWorker']
                    destSplited = dest[0].split('/')
                    method = destSplited[1]
                    param = destSplited[2]
                    instance_method = getattr(self, method)
                    result = instance_method(id=param, data=message["data"])
                    print(f"Received message: {result}")
                    sendMessage(
                        conn=self.conn,
                        status="completed",
                        destination=result["destination"],
                        messageId=message["messageId"],
                        data=convertObjectIdToStr(result.get('data', [])),
                    )
            except EOFError:
                log("Connection closed by supervisor", 'error')
                break
            except Exception as e:
                log(f"Message loop error: {e}", 'error')
                break

    #########################################
    # Methods for Database Interaction
    #########################################

    def getData(self, id):
        if not self._isBusy:
            self._isBusy = True
            collection = self._db['mycollection']
            data = list(collection.find({"project_id": id}))

            self._isBusy = False
            return {"data": data, "destination": ["RestApiWorker/onProcessed"]}

    def getTweetByKeyword(self, data):
        keyword, start_date, end_date = data
        # Match stage to filter tweets by keyword
        match_stage = {
            '$match': {
                'full_text': {'$regex': keyword, '$options': 'i'}
            }
        }

        pipeline = [match_stage]

        # Add date filtering if both start_date and end_date are provided
        if start_date and end_date:
            start_datetime = datetime.strptime(
                f"{start_date} 00:00:00 +0000", "%Y-%m-%d %H:%M:%S %z")
            end_datetime = datetime.strptime(
                f"{end_date} 23:59:59 +0000", "%Y-%m-%d %H:%M:%S %z")
            print(f"Filtering tweets from {start_datetime} to {end_datetime}")

            add_fields_stage = {
                '$addFields': {
                    'parsed_date': {'$toDate': '$created_at'}
                }
            }
            match_date_stage = {
                '$match': {
                    'parsed_date': {'$gte': start_datetime, '$lte': end_datetime}
                }
            }

            pipeline.extend([add_fields_stage, match_date_stage])

        # Project stage to include only specific fields
        project_stage = {
            '$project': {
                '_id': 0,
                'full_text': 1,
                'username': 1,
                'in_reply_to_screen_name': 1,
                'tweet_url': 1
            }
        }
        pipeline.append(project_stage)

        # Execute the aggregation pipeline
        cursor = self._dbtweet['tweets'].aggregate(pipeline)
        return {"data": list(cursor), "destination": ["RestApiWorker/onProcessed"]}

        # return list(cursor)
    def getTweetByIdStr(self, id_str_list):
        tweets = self._dbtweet['tweets'].find(
            {"id_str": {"$in": id_str_list, "$ne": None}},
            {"_id": 0, "full_text": 1, "id_str": 1, "user_id_str": 1, "username": 1,
                "conversation_id_str": 1, "tweet_url": 1, "in_reply_to_screen_name": 1}
        )
        return {"data": tweets, "destination": ["RestApiWorker/onProcessed"]}

    def classifyTweet(self, id_str, data):
        tweets = self._dbtweet["tweets"].update_many(
            {"id_str": id_str},
            {"$set": {"topic": data}}
        )
        return {"data": tweets, "destination": ["RestApiWorker/onProcessed"]}

    def createTopic(self, data):
        topics = self._db["topics"].insert_many(data)
        return {"data": topics, "destination": ["RestApiWorker/onProcessed"]}

    def createDocument(self, data):
        docs = self._db["documents"].insert_many(data)
        return {"data": docs, "destination": ["RestApiWorker/onProcessed"]}

    def createContext(self, data):
        context = self._db["context"].insert_many(data)
        return {"data": context, "destination": ["RestApiWorker/onProcessed"]}

    def getTopicByProjectId(self, projectId):
        topicProject = self._db["topics"].find(
            {"projectId": projectId},
            {"_id": 0}
        )
        return {"data": list(topicProject), "destination": ["RestApiWorker/onProcessed"]}

    def getContextTopicByProjectId(self, projectId):
        contextTopic = self._db["topics"].find(
            {"projectId": projectId},
            {"_id": 0, "context": 1, "topicId": 1, "keyword": 1}
        )
        return {"data": list(contextTopic), "destination": ["RestApiWorker/onProcessed"]}

    def getDocumentTopicByProjectId(self, projectId, topic=None):
        filter = {"projectId": projectId}
        if topic != None:
            filter["topic"] = topic

        documentTopic = self._db["documents"].find(
            filter,
            {"_id": 0, "full_text": 1, "username": 1, "tweet_url": 1, "topic": 1}
        )
        return {"data": list(documentTopic), "destination": ["RestApiWorker/onProcessed"]}

# Helper function to convert ObjectId to string in a list of documents


def convertObjectIdToStr(data: list) -> list:
    res = []
    for doc in data:
        doc["_id"] = str(doc["_id"])
        res.append(doc)
    return res
# This is the main function that the supervisor calls


def main(conn: Connection, config: dict):
    """Main entry point for the worker process"""
    worker = DatabaseInteractionWorker(conn, config)
    worker.run()
