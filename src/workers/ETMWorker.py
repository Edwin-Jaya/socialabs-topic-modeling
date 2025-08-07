from multiprocessing.connection import Connection
import threading
import traceback
import uuid
import time

import numpy as np
from  utils.log import log 
from utils.handleMessage import sendMessage, convertMessage

from octis.models.ETM import ETM
from octis.evaluation_metrics.coherence_metrics import Coherence

from octis.dataset.dataset import Dataset

from joblib import Parallel, delayed


from .Worker import Worker

class ETMWorker(Worker):
    ###############
    # dont edit this part
    ###############
    route_base = "/"
    conn:Connection
    requests: dict = {}
    def __init__(self):
        # we'll assign these in run()
        self._port: int = None

        self.requests: dict = {}
        
    def run(self, conn: Connection, config:dict):
        # assign here
        ETMWorker.conn = conn

        #### add your worker initialization code here
        
        self.dataset_path = 'src/vocabs/octis_data/'
        self.dataset =Dataset()
        self.dataset.load_custom_dataset_from_folder(self.dataset_path)
        log("ETMWorker initialized", "info")
        
        #### until this part
        # start background threads *before* blocking server
        threading.Thread(target=self.listen_task, daemon=True).start()
        threading.Thread(target=self.health_check, daemon=True).start()

        # asyncio.run(self.listen_task())
        self.health_check()


    def health_check(self):
        """Send a heartbeat every 10s."""
        while True:
            sendMessage(
                conn=ETMWorker.conn,
                messageId="heartbeat",
                status="healthy"
            )
            time.sleep(10)
    def listen_task(self):
        while True:
            try:
                if ETMWorker.conn.poll(1):  # Check for messages with 1 second timeout
                    message = self.conn.recv()
                    dest = [
                        d
                        for d in message["destination"]
                        if d.split("/", 1)[0] == "ETMWorker"
                    ]
                    destSplited = dest[0].split('/')
                    method = destSplited[1]
                    param= destSplited[2]
                    instance_method = getattr(self,method)
                    instance_method(id=param,data=message['data'],message=message)
            except EOFError:
                break
            except Exception as e:
              print(e)
              log(f"Listener error: {e}",'error' )
              break

    def sendToOtherWorker(self, destination, messageId: str, data: dict = None) -> None:
      sendMessage(
          conn=ETMWorker.conn,
          destination=destination,
          messageId=messageId,
          status="completed",
          reason="Message sent to other worker successfully.",
          data=data or {}
      )
    ##########################################
    # add your worker methods here
    ##########################################
    
    
    def create_and_train_etm(self, num_topics):
      try:
        log(f"Creating and training ETM model with {num_topics} topics", "info")
        model = ETM(
            num_topics=num_topics,
            num_epochs=300,
            batch_size=64,
            dropout=0.3,
            t_hidden_size=256,
            wdecay=1e-4,
            lr=0.002,
            optimizer='rmsprop'
        )
        model_output = model.train_model(self.dataset)
        
        return (num_topics, model, model_output)
      except Exception as e:
        traceback.print_exc()
        log(f"Error in create_and_train_etm: {e}", "error")
    
    def generateTopic(self):
        best_coh = float("-inf")
        best_topic = None

        coh_score_list = []
        topics = range(2, 9)

        # 2) Parallel execution
        results = Parallel(n_jobs=-1)(
            delayed(self.create_and_train_etm)(topic)
            for topic in topics
        )

        # print("\n=== Summary ===")
        # 3) Process results
        for num_topics, _, model_output in results:
            coh_score = self.evaluate_coherence(self.dataset, model_output)
            print(f"[{num_topics} topics] Coherence: {coh_score:.4f}")

            if coh_score > best_coh:
                best_coh = coh_score
                best_topic = num_topics

        # print(f"\nBest model has {best_topic} topics with coherence={best_coh:.4f}",end="\n")

        model = self.create_and_train_etm(best_topic)
        
        return model
        
    # def document(self, data_tweet, etm_model):
    #     train_corpus = self.dataset.get_partitioned_corpus()[0]
    #     print("Training corpus size:", len(train_corpus))
    #     documents_probability = []
    #     probs = etm_model[2]['topic-document-matrix']
    #     print("Topic-document matrix shape:", probs.shape)
    #     print("Matrix type:", type(probs))
    #     print(probs)
    #     for i, train_corpus in enumerate(train_corpus):
    #         top_topic = max(probs, key=lambda x: x[1])
    #         print("Topic for document {}: {}".format(i+1, top_topic[0]))
    #         data_tweet[i].update({
    #             "topic": str(top_topic[0]),
    #             "probability": str(top_topic[1])
    #         })
    #         documents_probability.append(data_tweet[i])
        
    #     return documents_probability
    
    def document(self, data_tweet, etm_model):
        train_corpus = self.dataset.get_partitioned_corpus()[0]
        # print("Training corpus size:", len(train_corpus))
        documents_probability = []
        
        probs = etm_model[2]['topic-document-matrix']
        # print("Topic-document matrix shape:", probs.shape)
        # print("Matrix type:", type(probs))
        # print(probs)

        num_docs = probs.shape[1]

        for i in range(num_docs):
            column = probs[:, i]
            topic_index = np.argmax(column)
            probability = column[topic_index]
            
            # print("Doc {}: topic={}, prob={}".format(i+1, topic_index+1, probability))

            data_tweet[i] ={
                "full_text": data_tweet[i],
                "topic": str(topic_index),
                "probability": str(probability)
            }
            documents_probability.append(data_tweet[i])

        return documents_probability


    def evaluate_coherence(self, dataset, model_output):
        coh = Coherence(texts=dataset.get_corpus(),topk=10,
                    measure='c_v')
        
        return coh.score(model_output)
    def run_etm(self,id,data,message):
      try:
        tweets = data['tweets']
        keyword = data['keyword']
        start_date = data['start_date']
        end_date = data['end_date']
        log(f"Running ETM with id {id}", "info")
        
        generated_topic = self.generateTopic()
        
        num_of_topic = generated_topic[0]
        print(f"Generated {num_of_topic} topics", "info")
        print(generated_topic[2])
        topics = generated_topic[2]['topics']
        # documents_prob = self.document(data_tweet=tweets, etm_model=generated_topic)
        # print("Documents with topics and probabilities:")
        # 	context = Llm.getContext(topics, keyword, num_topics)

        self.sendToOtherWorker(
            destination=[f"LLMWorker/getContext/{id}"],
            messageId=str(uuid.uuid4()),
            data={
                "topics": topics,
                "num_of_topic": num_of_topic,
                "keyword": keyword,
                "start_date": start_date,
                "end_date": end_date,
            }
        )
        # print(documents_prob)
        # print("Topics:")
        
        # print(topics)
        pass
      except Exception as e:
        traceback.print_exc()
        print(e)

def main(conn: Connection, config: dict):
    worker = ETMWorker()
    worker.run(conn, config)
