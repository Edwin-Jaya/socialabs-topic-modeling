from multiprocessing.connection import Connection
import threading
import uuid
import time
from utils.log import log
from utils.handleMessage import sendMessage, convertMessage

from .Worker import Worker
import json
import os
import re
from openai import AzureOpenAI


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

    def getContext(topics, keyword, best_num_topics_str):
        client = AzureOpenAI(
            api_version=os.getenv("AZURE_OPENAI_MODEL_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
        )
        # print(topics)
        completion = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[
                {
                    "role": "system",
                    "content": "Anda adalah AI Linguistik yang dapat menentukan kalimat dari beberapa topik hasil dari proses topic modeling yang berupa kumpulan kata-kata,  dengan mempertimbangkan bobot setiap topik yang ada, dalam merangkai kata-kata kunci menjadi kalimat yang padu untuk sebuah topik yang diperbincangkan di Twitter dengan mengambil kata dari hasil topic modeling lalu menyusunnya menjadi sebuah kalimat yang padu yang mudah dipahami."
                },
                {
                    "role": "user",
                    "content": f"Topik ini membahas tentang keyword: {keyword} dengan berbagai pandangan masyarakat terhadap topik tersebut dengan hasil topic modeling dengan {best_num_topics_str} topik terdiri dari beberapa kata kunci berikut: {topics} Buatkan dengan format JSON dengan 1 topik untuk 1 kalimat utama dengan jumlah sesuai jumlah topik yang diberikan yaitu: {best_num_topics_str}. Berikut ini adalah format JSON-nya: \n            [\n                {{\n                    \"kata_kunci\": \"...\"\n                    \"kalimat\": Topik ini tentang \"...\"\n                }}\n                ...\n            ]\n            ONLY answer in JSON FORMAT without opening words. "
                }
            ],
        )
        generated_sentence = completion.choices[0].message.content or ""
        # for chunk in completion:
        #     content = chunk.choices[0].delta.content or ""
        #     print(content, end="")         # prints to stdout
        #     generated_sentence += content  # appends to string

        print(generated_sentence)
        pattern = re.compile(r'\[(?:\s*{[^{}]*}\s*,?)*\s*\]')
        match = pattern.search(generated_sentence)

        if match:
            json_text = match.group()
            print(json_text)
        else:
            print("Tidak ada JSON yang ditemukan dalam string.")
        # Print the generated sentence
        res_json = json.loads(json_text)
        print(type(res_json))
        res = {
            "context": "",
            "interpretation": []
        }
        for index, item in enumerate(res_json):
            res['context'] += str(index+1)+". "+item['kalimat']+"<br/>"
            res['interpretation'].append({
                "word_topic": item['kata_kunci'],
                "word_interpretation": item['kalimat']
            })
            
        return res 

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
