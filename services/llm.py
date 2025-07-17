from groq import Groq
import json
import os
import re
from openai import AzureOpenAI

class Llm:
    # def __init__(self):
        # client = Groq(api_key=os.getenv('GROQ_API_KEY'))

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