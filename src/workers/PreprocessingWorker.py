import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from multiprocessing.connection import Connection
import traceback
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import re
import threading
from typing import Counter
import uuid
import time
import nest_asyncio

import numpy
from openai import AsyncAzureOpenAI, AzureOpenAI
import pandas
from  utils.log import log 
from utils.handleMessage import sendMessage, convertMessage

from .Worker import Worker

class PreprocessingWorker(Worker):
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
        PreprocessingWorker.conn = conn

        #### add your worker initialization code here
        log("Initializing PreprocessingWorker", "info")
        self.client = AzureOpenAI(
            api_version=config['azure']['api_version'],
            azure_endpoint=config['azure']['endpoint'],
            api_key=config['azure']['api_key'],
        )
        print("Azure OpenAI client initialized")
        self.model_name = config['azure']['model']['completion']
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        
        
        self.async_client = AsyncAzureOpenAI(
                    api_key=config['azure']['api_key'],
                    azure_endpoint=config['azure']['endpoint'],
                    api_version=config['azure']['api_version'],
                )
        
        log(f"Initialized OpenAI client with model {self.model_name}", "info")
        
        
        
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
                conn=PreprocessingWorker.conn,
                messageId="heartbeat",
                status="healthy"
            )
            time.sleep(10)
    def listen_task(self):
        while True:
            try:
                if PreprocessingWorker.conn.poll(1):  # Check for messages with 1 second timeout
                    message = self.conn.recv()
                    dest = [
                        d
                        for d in message["destination"]
                        if d.split("/", 1)[0] == "PreprocessingWorker"
                    ]
                    destSplited = dest[0].split('/')
                    method = destSplited[1]
                    param= destSplited[2]
                    instance_method = getattr(self,method)
                    instance_method(id=param,data=message['data'], message=message)
            except EOFError:
                break
            except Exception as e:
              print(e)
              log(f"Listener error: {e}",'error' )
              break

    def sendToOtherWorker(self, destination, messageId: str, data: dict = None) -> None:
      sendMessage(
          conn=PreprocessingWorker.conn,
          destination=destination,
          messageId=messageId,
          status="completed",
          reason="Message sent to other worker successfully.",
          data=data or {}
      )
    ##########################################
    # add your worker methods here
    ##########################################
    
    def create_dataframe(self,tweets):
        df = pandas.DataFrame({
            'tweets': tweets
        })
        return df
    
    def create_explanation(self, keyword):
        # print("Initialized OpenAI")
       
        # print("Initialized Response")
        # print(self.model_name)
        response = self.client.chat.completions.create(
            messages=[{"role": "system",
                        "content": f"""You are a diligent assistant. The fate of
                                    the world depends on your answer being
                                    correct. Think carefully step by step."""},
                    {"role": "user",
                        "content": f"""
                        Berikan penjelasan singkat dalam bentuk 1 paragraf singkat dan dalam bahasa Indonesia mengenai kata kunci berikut di Indonesia: {keyword}.
                        """}], 
            max_completion_tokens=4096,
            model=self.model_name,
            # temperature=0.5,
            # top_p=1
        )

        
        content = response.choices[0].message.content
        # print("Konten:", content)
    
        return content
            
    

    async def create_augmentation_async(self,
         tweets: list,
        batch_size: int,
        temperature: float,
        tokens: int,
        top_p: float,
        keyword: str,
        explanation: str,
        max_retries: int = 3
    ) -> list:
        """
        Create augmentation for a single batch of tweets, with retry and fallback.
        """
        attempt = 0
        success_count = 0
        error_count = 0
        result =[]
        log(f"[Batch] Processing {len(tweets)} tweets with batch size {batch_size}...", "info")
        # print(f"[Batch] Processing {len(tweets)} tweets with batch size {batch_size}...")
        while attempt < max_retries:
            try:
                

                prompt = f"""
    You are given a list of {batch_size} posts from an {keyword} community on a social network.
    For each post in the list:
    - If the post is already in Indonesian, rephrase it into a more detail version formal Bahasa Indonesia with at least 70 words.
    - If the post is in a foreign language, translate and rephrase it into more detail version formal Bahasa Indonesia with at least 70 words.

    Return your answer as a Python list of strings, containing only the more detail version final formal Indonesian version of each post.
    Return your answer as a Python list of {batch_size} strings, in the same order as the input.
    Do NOT include the original text or any translation notes—only the more detail version final formal Indonesian versions.

    Topic: {keyword}
    Explanation: {explanation}
    Posts: {tweets}

    Example input:
    ["This is a test.", "Kita harus bekerja sama.", "¡Vamos a ganar!"]

    Example output:
    [
        "Ini adalah sebuah uji coba.",
        "Kita harus bekerja sama.",
        "Kita akan menang!"
    ]

    Answer ONLY with the Python list of strings, nothing else.
    """
                # print(self.model_name)
                # print(f"[Batch] Attempt {attempt + 1}")
                
                response = await self.async_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=4096,
                    model=self.model_name,
                )
                output = response.choices[0].message.content or ""
                
                # print(output)
                # print(type(output))
                # print('==============================')
                # Parse output safely
                output.strip()
                augmented = ast.literal_eval(output.strip()) 

                log(f"[Batch] Attempt {attempt + 1} succeeded with {len(tweets)} tweets", "info")
                success_count += 1
                return augmented

            except Exception as e:
                attempt += 1
                error_count += 1
                log(f"[Batch] Error on attempt {attempt}: {e}", "error")
                # print(f"[Batch] Error on attempt {attempt}: {e}")
                await asyncio.sleep(1)

        # If all retries fail, fallback
        # print("[Batch] All retries failed—returning original batch.")
        # print(f"[Batch] Success count: {success_count}/{attempt}, Error count: {error_count}/{attempt}")
        return tweets
            
        
    async def augment_all_batches(
        self,
        all_tweets: list,
        keyword: str,
    ):
        batch_size=10
        temperature=0.5
        tokens=4000
        top_p=1
        # Divide tweets into batches
        # print("Membuat eksplanasi")
        explanation = self.create_explanation(keyword)
        # print("Membuat batches")
        batches = [all_tweets[i:i+batch_size] for i in range(0, len(all_tweets), batch_size)]
        # print(f"Processing {len(batches)} batches of {batch_size}...")
        log(f"Processing {len(batches)} batches of {batch_size}/{len(all_tweets)} tweets with keyword '{keyword}'", "info")
        tasks = [
            self.create_augmentation_async(
                tweets=batch,
                batch_size=batch_size,
                temperature=temperature,
                tokens=tokens,
                top_p=top_p,
                keyword=keyword,
                explanation=explanation,
            )
            for batch in batches
        ]
        log(f"Starting augmentation for {len(batches)} batches of size {batch_size} with keyword '{keyword}'", "info")
        # Run all batches concurrently
        augmented_results = await asyncio.gather(*tasks)
        # Flatten results (list of lists → single list)
        all_augmented = [aug for batch in augmented_results for aug in batch]
        log(f"Augmentation completed for {len(all_augmented)} tweets with keyword '{keyword}'", "info")
        return all_augmented
        
        
    def remove_url(self, tweets):
        # This pattern matches more URL variations
        url_pattern = re.compile(
            r'(?:https?://|www\.)'  # http://, https://, or www.
            r'(?:[^\s./]+\.)+'       # domain parts
            r'[^\s./]+'              # last domain part
            r'(?:/\S*)?'             # optional path
        )
        return [url_pattern.sub('', s).strip() for s in tweets]

    ## Change Emoticons
    def replace_emoticons(self, tweet):
        """
        Replace common emoticons with descriptive text.
        
        Args:
            text (str or list): Input string or list of strings
            
        Returns:
            str or list: Text with emoticons replaced
        """
        # Define emoticon mappings
        emoticon_map = {
            r':\)|:-\)|=\)': 'emot-senyum',    # :) :-) =)
            r':\(|:-\(|=\(': 'emot-sedih',     # :( :-( =(
            r':D|:-D|=D': 'emot-tertawa',      # :D :-D =D
            r';\)|;-\)': 'emot-mengedip',       # ;) ;-)
            r':P|:-P|=P': 'emot-julur',        # :P :-P =P
            r':O|:-O|=O': 'emot-terkejut',     # :O :-O =O
            r':\/|:-\\': 'emot-bingung',       # :/ :-\
            r'<3': 'emot-hati',                # <3 (heart)
            r':\*|:-\*': 'emot-ciuman',        # :* :-* (kiss)
        }
        
        if isinstance(tweet, list):
            return [self.replace_emoticons(s) for s in tweet]
        else:
            for pattern, replacement in emoticon_map.items():
                tweet = re.sub(pattern, replacement, tweet)
            return tweet

    def remove_twitter_symbols(self, tweet):
        """
        Remove Twitter-specific symbols:
        - Hashtags (#example)
        - Mentions (@username)
        - Retweet (RT)
        
        Args:
            text (str or list): Input string or list of strings
            
        Returns:
            str or list: Text with Twitter symbols removed
        """
        if isinstance(tweet, list):
            return [self.remove_twitter_symbols(s) for s in tweet]
        else:
            # Remove hashtags (e.g., #Hello → " ")
            tweet = re.sub(r'#\w+', ' ', tweet)
            # Remove mentions (e.g., @user → " ")
            tweet = re.sub(r'@\w+', ' ', tweet)
            # Remove "RT " (Retweet)
            tweet = re.sub(r'\bRT\b', ' ', tweet)
            # Clean extra spaces
            tweet = re.sub(r'\s+', ' ', tweet).strip()
            
            return tweet
    
    def remove_symbols_and_punctuation(self, tweet):
        """
        Remove all ASCII symbols, numbers, and punctuation from text.
        Keeps only letters (a-z, A-Z) and basic whitespace.
        
        Args:
            text (str or list): Input string or list of strings
            
        Returns:
            str or list: Cleaned text without symbols/numbers/punctuation
        """
        if isinstance(tweet, list):
            return [self.remove_symbols_and_punctuation(s) for s in tweet]
        else:
            # Remove all non-alphabetic characters except spaces
            tweet = re.sub(r'[^a-zA-Z\s]', ' ', tweet)
            # Collapse multiple spaces into one
            tweet = re.sub(r'\s+', ' ', tweet).strip()
            
            return tweet
    
    def case_folding(self, tweets):
        return [[token.lower() for token in tweet] for tweet in tweets]

    def tokenizing(self, tweets):
        return [str(tweet).split() for tweet in tweets]

    def delete_extra_letters(self, tweets):
        sequence_pattern = r'([A-Za-z])\1{2,}'  # Matches 3 or more consecutive identical letters
        seq_replace_pattern = r'\1'

        # Iterate through each sentence and token
        return [
            [re.sub(sequence_pattern, seq_replace_pattern, token) for token in sentence]
            for sentence in tweets
        ]
    
    def normalization(self, tweets):
        res = []
        base_path = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.abspath(base_path+'/../utils/kbba.txt'), 'r', encoding='utf-8') as file:
            lines = file.readlines()

            data = [line.strip().split('\t') for line in lines]
            data_singkatan = pandas.DataFrame(data, columns=['Kata', 'Asli'])

            kontraksi_dict = dict(zip(data_singkatan['Kata'], data_singkatan['Asli']))

            for tweet in tweets:
                expanded_text = [kontraksi_dict[word] if word in kontraksi_dict else word for word in tweet]

                res.append(expanded_text)

            return res

    def stem_tokens(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]

    def stem_tokenized_list_parallel(self, tweets, max_workers=4):
        # print(tweets)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.stem_tokens, tweets))
        return results

    def curating_stopword(self, tweets):
        result = [" ".join(sublist) for sublist in tweets]
        tr_idf_model  = TfidfVectorizer()
        tf_idf_vector = tr_idf_model.fit_transform(result)
        tf_idf_array = tf_idf_vector.toarray()
        words_set = tr_idf_model.get_feature_names_out()
        df_tf_idf = pandas.DataFrame(tf_idf_array, columns = words_set)
        columns_with_one = df_tf_idf.columns[(df_tf_idf > 0.7).any()].tolist()
        word_freq = Counter(word for doc in tweets for word in doc)
        if (len(tweets))>=10000:
            rare_words = [word for word, freq in word_freq.items() if freq <= 10]
        elif (len(tweets))<10000 and (len(tweets))>=100:
            rare_words = [word for word, freq in word_freq.items() if freq < 2]
        else:
            # rare_words = [word for word, freq in word_freq.items() if freq < 2]
            rare_words = [""]
        return columns_with_one, rare_words
    
    def stopword_removal(self, tweets):
        """
        Remove Indonesian stopwords, single/two-character tokens, and custom words.
        
        Args:
            tokenized_texts (list): List of lists, where each sublist contains tokenized words.
            
        Returns:
            list: Lists of tokens with stopwords, short tokens (≤2 chars), and custom words removed.
        """
        factory = StopWordRemoverFactory()
        stopword_remover = factory.create_stop_word_remover()

        # PRON (kata ganti)
        PRON = [
            "aku","saya","gue","gw","kamu","kau","engkau",
            "dia","ia","kita","kami","mereka","anda","lo","lu", "kalian"
        ]

        columns_with_one, rare_words = self.curating_stopword(tweets)
        custom_stopwords = set(columns_with_one + rare_words + ['aduh','sangat','amp', 'the', 'link', 'yang', "iya", "ada", "tin", 'sangat', 'tidak', 'jadi', 'mungkin', 'apa', 'orang', 'wah'] + PRON)
        
        cleaned_texts = []
        
        for tokens in tweets:
            sentence = ' '.join(tokens)
            # Step 1: Remove default Indonesian stopwords using Sastrawi
            cleaned_sentence = stopword_remover.remove(sentence)
            # Step 2: Tokenize and filter short/custom tokens
            cleaned_tokens = [
                token for token in cleaned_sentence.split()
                if len(token) > 2 and token.lower() not in custom_stopwords
            ]
            cleaned_texts.append(cleaned_tokens)
        
        cleaned_texts =[tweet for tweet in cleaned_texts if tweet]
        
        return cleaned_texts
    
    def split_dataset(self, tweets):

        # Compute split indices
        train_size = int(0.85 * len(tweets))
        val_size = int(0.05 * len(tweets))

        # Label rows
        tweets['label'] = numpy.where(
            tweets.index < train_size,
            'train',
            numpy.where(
                tweets.index < train_size + val_size,
                'val',
                'test'
            )
        )

        # Ensure tweets are string and clean
        tweets['tweets'] = tweets['tweets'].astype(str)
        tweets['tweets'] = tweets['tweets'].apply(self.clean_tweet_string)
        
        return tweets


    def clean_tweet_string(self, tweet_str):
        try:
            # Convert string representation of list to actual list
            tweet_list = ast.literal_eval(tweet_str)
            # Join list elements with spaces
            return ' '.join(tweet_list)
        except (ValueError, SyntaxError):
            # Fallback if the string isn't a valid list representation
            return tweet_str.replace('[', '').replace(']', '').replace('\'', '')

    
    def saving_vocab_corpus(self, vocabulary, tweet):
        path = "./src/vocabs/octis_data/"
        with open(path + 'vocabulary.txt', 'w') as file:
            for word in sorted(vocabulary):
                file.write(word + '\n')
        # print("Vocabulary file created successfully!")
        tweet.to_csv(path +"corpus.tsv", index=False, sep="\t", header=False) 
        # print("Corpus file created successfully!") 
    
            
    def create_vocabulary(self, tweets):

        vocabulary = set(word.lower() for text in tweets['tweets'] for word in text.split())
        # Save vocabulary to .txt file
        self.saving_vocab_corpus(vocabulary, tweets[['tweets','label']])
        
        # print("Done!")
        
        return tweets
      
    def prepare_preprocessing(self, data, id, message):
      keyword = data['keyword']
      start_date = data['start_date']
      end_date = data['end_date']
      project_id = data['project_id']
      m_id = message['messageId']
      log(f"Preparing preprocessing for keyword: {keyword}, project_id: {project_id}, messageId: {m_id}", "info")
      self.sendToOtherWorker(
          destination=[f'DatabaseInteractionWorker/getTweetByKeyword/{project_id}'],
          messageId=m_id,
          data={
              'keyword': keyword,
              'start_date': start_date,
              'end_date': end_date
          }
      )
      log(f"Sent request to DatabaseInteractionWorker for keyword: {keyword}, project_id: {project_id}, messageId: {m_id}", "info")
    def run_preprocessing(self, id,data,message):
        try:
            log(f"Running preprocessing for keyword: {data['keyword']}, project_id: {id}, messageId: {message['messageId']}", "info")
            tweets=data['tweets']
            keyword=data['keyword']
            start_date=data['start_date']
            end_date=data['end_date']
            
            
            text_tweet = [tweet['full_text'] for tweet in tweets if 'full_text' in tweet]
            try:
                log(f"Starting augmentation for keyword: {keyword}, project_id: {id}, messageId: {message['messageId']}", "info")
                
                evt_loop = asyncio.get_event_loop() 

                # Run async augmentation synchronously
                if evt_loop.is_running():
                    log(f"Event loop is already running for keyword: {keyword}, project_id: {id}, messageId: {message['messageId']}", "info")
                    nest_asyncio.apply()
                    
                    data = evt_loop.run_until_complete(
                        self.augment_all_batches(
                        all_tweets=text_tweet,
                        keyword=keyword,
                    ))
                else:
                    log(f"Creating new event loop for keyword: {keyword}, project_id: {id}, messageId: {message['messageId']}", "info")
                    data = asyncio.run(
                        self.augment_all_batches(
                        all_tweets=text_tweet,
                        keyword=keyword,
                    ))
            except Exception as e:
                data = asyncio.run(
                        self.augment_all_batches(
                        all_tweets=text_tweet,
                        keyword=keyword,
                    ))
            log(f"Augmentation completed for keyword: {keyword}, project_id: {id}, messageId: {message['messageId']}", "info")
            log(f"Starting preprocessing steps for keyword: {keyword}, project_id: {id}, messageId: {message['messageId']}", "info")
            log(f"Removing URLs for keyword: {keyword}, project_id: {id}, messageId: {message['messageId']}", "info")
            data = self.remove_url(data)
            log(f"Removing Emoticons for keyword: {keyword}, project_id: {id}, messageId: {message['messageId']}", "info")
            data = self.replace_emoticons(data)
            log(f"Removing Twitter symbols for keyword: {keyword}, project_id: {id}, messageId: {message['messageId']}", "info")
            data = self.remove_twitter_symbols(data)
            log(f"Removing symbols and punctuation for keyword: {keyword}, project_id: {id}, messageId: {message['messageId']}", "info")
            data = self.remove_symbols_and_punctuation(data)
            log(f"Tokenizing for keyword: {keyword}, project_id: {id}, messageId: {message['messageId']}", "info")
            data = self.tokenizing(data)
            log(f"Case folding for keyword: {keyword}, project_id: {id}, messageId: {message['messageId']}", "info")
            data = self.case_folding(data)
            log(f"Deleting extra letters for keyword: {keyword}, project_id: {id}, messageId: {message['messageId']}", "info")
            data = self.delete_extra_letters(data)
            log(f"Normalization for keyword: {keyword}, project_id: {id}, messageId: {message['messageId']}", "info")
            data = self.normalization(data)
            log(f"Stemming for keyword: {keyword}, project_id: {id}, messageId: {message['messageId']}", "info")
            data = self.stem_tokenized_list_parallel(data)
            log(f"Curating stopwords for keyword: {keyword}, project_id: {id}, messageId: {message['messageId']}", "info")
            data = self.stopword_removal(data)
            data = self.create_dataframe(data)
            data = self.split_dataset(data)
            data = self.create_vocabulary(data)
            log(f"Preprocessing completed for keyword: {keyword}, project_id: {id}, messageId: {message['messageId']}", "info")
            self.sendToOtherWorker(
                destination=[f'ETMWorker/run_etm/{id}'],
                messageId= message['messageId'],
                data={
                    'keyword': keyword,
                    'tweets': data['tweets'].tolist(),
                    'label': data['label'].tolist(),
                    "start_date": start_date,
                    "end_date": end_date,
                }
            )
            
            return
        except Exception as e:
            traceback.print_exc()
            log(f"Error during preprocessing for keyword: {keyword}, project_id: {id}, messageId: {message['messageId']}: {e}", "error")
      # Send the processed data back to the DatabaseInteractionWorker
def main(conn: Connection, config: dict):
    worker = PreprocessingWorker()
    worker.run(conn, config)

if __name__ == "__main__":
    # Example usage
    conn = Connection(('localhost', 6000))  # Replace with actual connection details
    config = {
        'azure': {
            'api_version': '2025-01-01-preview',
            'endpoint': 'https://research-etm.openai.azure.com/',
            'api_key': '6YwpSUX7CKAhTAWRFieW7zj7Q3OoXJNtjGOCsvZsFnTN7g7MyX7SJQQJ99BGACYeBjFXJ3w3AAABACOGG9Qs',
            'model': {
                'completion': 'o4-mini'
            }
        }
    }
    main(conn, config)