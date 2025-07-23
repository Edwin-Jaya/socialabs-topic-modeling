from multiprocessing.connection import Connection
import threading
import uuid
import time
from utils.log import log
from utils.handleMessage import sendMessage, convertMessage

from .Worker import Worker
import numpy as np

from octis.dataset.dataset import Dataset
from octis.models.ETM import ETM
from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence
from joblib import Parallel, delayed


class TemplateWorker(Worker):
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
        TemplateWorker.conn = conn

        # add your worker initialization code here

        dataset_path = "../temp/octis_data"
        self.dataset = Dataset()
        self.dataset.load_custom_dataset_from_folder(dataset_path)

        # until this part
        # start background threads *before* blocking server
        threading.Thread(target=self.listen_task, daemon=True).start()
        threading.Thread(target=self.health_check, daemon=True).start()

        # asyncio.run(self.listen_task())
        self.health_check()

    def health_check(self):
        """Send a heartbeat every 10s."""
        while True:
            sendMessage(
                conn=TemplateWorker.conn,
                messageId="heartbeat",
                status="healthy"
            )
            time.sleep(10)

    def listen_task(self):
        while True:
            try:
                # Check for messages with 1 second timeout
                if TemplateWorker.conn.poll(1):
                    message = self.conn.recv()
                    dest = [
                        d
                        for d in message["destination"]
                        if d.split("/", 1)[0] == "TemplateWorker"
                    ]
                    destSplited = dest[0].split('/')
                    method = destSplited[1]
                    param = destSplited[2]
                    instance_method = getattr(self, method)
                    instance_method(message)
            except EOFError:
                break
            except Exception as e:
                print(e)
                log(f"Listener error: {e}", 'error')
                break

    def sendToOtherWorker(self, destination, messageId: str, data: dict = None) -> None:
        sendMessage(
            conn=TemplateWorker.conn,
            destination=destination,
            messageId=messageId,
            status="completed",
            reason="Message sent to other worker successfully.",
            data=data or {}
        )
    ##########################################
    # add your worker methods here
    ##########################################

    def create_and_train_etm(self, dataset, num_topics: int):

        model = ETM(
            num_topics=num_topics,
            num_epochs=100,
            batch_size=256,
            dropout=0.2,
            embeddings_path="../utils/idwiki_word2vec_100_new_lower.txt",
            embeddings_type="word2vec",
            t_hidden_size=256,
            wdecay=1e-6,
            lr=0.002,
            optimizer='rmsprop'
        )

        model_output = model.train_model(dataset)

        return (num_topics, model, model_output)

    def generateTopic(self):
        best_coh = float("-inf")
        best_topic = None

        topics = range(2, 9)

        # 2) Parallel execution
        results = Parallel(n_jobs=-1)(
            delayed(self.create_and_train_etm)(self.dataset, topic)
            for topic in topics
        )

        print("\n=== Summary ===")
        # 3) Process results
        for num_topics, _, model_output in results:
            coh_score = self.evaluate_coherence(self.dataset, model_output)
            # print(f"[{num_topics} topics] Coherence: {coh_score:.4f}")

            if coh_score > best_coh:
                best_coh = coh_score
                best_topic = num_topics

        # print(f"\nBest model has {best_topic} topics with coherence={best_coh:.4f}",end="\n")

        model = self.create_and_train_etm(self.dataset, best_topic)

        return model

    def document(self, data_tweet, etm_model):
        train_corpus = self.dataset.get_partitioned_corpus()[0]
        print("Training corpus size:", len(train_corpus))
        documents_probability = []

        probs = etm_model[2]['topic-document-matrix']
        print("Topic-document matrix shape:", probs.shape)
        print("Matrix type:", type(probs))
        print(probs)

        num_docs = probs.shape[1]

        for i in range(num_docs):
            column = probs[:, i]
            topic_index = np.argmax(column)
            probability = column[topic_index]

            print("Doc {}: topic={}, prob={}".format(
                i+1, topic_index+1, probability))

            data_tweet[i].update({
                "topic": str(topic_index),
                "probability": str(probability)
            })
            documents_probability.append(data_tweet[i])

        return documents_probability

    def evaluate_coherence(self, dataset, model_output):
        coh = Coherence(texts=dataset.get_corpus(), topk=10,
                        measure='c_v')

        return coh.score(model_output)

    def test(self, message) -> None:
        """
        Example method to test the worker functionality.
        Replace this with your actual worker methods.
        """
        data = message.get("data", {})

        # process

        # send back to RestAPI
        self.sendToOtherWorker(
            messageId=message.get("messageId"),
            destination=["RestApiWorker/onProcessed"],
            data=data
        )
        #   sendMessage(
        #     status="completed",
        #     reason="Test method executed successfully.",
        #     destination=["supervisor"],
        #     data={"message": "This is a test response."}
        # )
        log("Test method called", "info")
        # return {"status": "success", "data": "This is a test response."}


def main(conn: Connection, config: dict):
    worker = TemplateWorker()
    worker.run(conn, config)
