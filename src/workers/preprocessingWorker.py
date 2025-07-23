from multiprocessing.connection import Connection
import threading
import uuid
import time
from utils.log import log
from utils.handleMessage import sendMessage, convertMessage

from .Worker import Worker
import numpy as np
import re
import re
import pandas
import nltk
import ast
from collections import Counter
from nltk.tokenize import word_tokenize
import numpy
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import os
from concurrent.futures import ThreadPoolExecutor
import asyncio
import time
import ast
from openai import AsyncAzureOpenAI
from openai import AzureOpenAI

nltk.download('punkt')


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

    def run(self, conn: Connection, port: int, tweet, keyword):
        # assign here
        TemplateWorker.conn = conn

        # add your worker initialization code here

        self.data = tweet
        self.keyword = keyword
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        print("Jumlah Data : {}".format(len(self.data)))

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

    def run_async(self, coro):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(coro)
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    def run_preprocessing(self, api_key=None, base_url=None):
        if not api_key:
            api_key = os.getenv("AZURE_OPENAI_KEY")
        if not base_url:
            base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
        print("Memulai Augmentasi")
        # Run async augmentation synchronously
        self.data = self.run_async(self.augment_all_batches(
            all_tweets=self.data,
            keyword=self.keyword,
            api_key=api_key,
            base_url=base_url
        ))
        print("Memulai URL")
        self.data = self.remove_url(self.data)
        print("Memulai Emoticon")
        self.data = self.replace_emoticons(self.data)
        print("Memulai Remove Symbol")
        self.data = self.remove_twitter_symbols(self.data)
        self.data = self.remove_symbols_and_punctuation(self.data)
        print("Memulai Tokenizing")
        self.data = self.tokenizing(self.data)
        print("Memulai Case Folding")
        self.data = self.case_folding(self.data)
        print("Memulai Delete Extra Letters")
        self.data = self.delete_extra_letters(self.data)
        print("Memulai Normalisasi")
        self.data = self.normalization(self.data)
        print("Memulai Stem")
        self.data = self.stem_tokenized_list_parallel(self.data)
        print("Memulai Stopword Removal")
        self.data = self.stopword_removal(self.data)
        self.data = self.create_dataframe(self.data)
        self.data = self.split_dataset(self.data)
        self.data = self.create_vocabulary(self.data)
        
        return self.data
    
    def create_dataframe(self,tweets):
        df = pandas.DataFrame({
            'tweets': tweets
        })
        return df
    
    def create_explanation(self, keyword):
        print("Initialized OpenAI")
        client = AzureOpenAI(
            api_version=os.getenv("AZURE_OPENAI_MODEL_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
        )
        print("Initialized Response")
        response = client.chat.completions.create(
            messages=[{"role": "system",
                        "content": f"""You are a diligent assistant. The fate of
                                    the world depends on your answer being
                                    correct. Think carefully step by step."""},
                    {"role": "user",
                        "content": f"""
                        Berikan penjelasan singkat dalam bentuk 1 paragraf singkat dan dalam bahasa Indonesia mengenai kata kunci berikut di Indonesia: {keyword}.
                        """}], 
            max_completion_tokens=4096,
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            # temperature=0.5,
            # top_p=1
        )

        
        content = response.choices[0].message.content
        print("Konten:", content)
    
        return content
            
    

    async def create_augmentation_async(self,
        tweets: list,
        batch_size: int,
        temperature: float,
        tokens: int,
        top_p: float,
        keyword: str,
        explanation: str,
        api_key: str,
        base_url: str = None,
        max_retries: int = 3
    ) -> list:
        """
        Create augmentation for a single batch of tweets, with retry and fallback.
        """
        attempt = 0

        while attempt < max_retries:
            try:
                client = AsyncAzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=base_url,
                    api_version=os.getenv("AZURE_OPENAI_MODEL_VERSION")
                )

                prompt = f"""
    You are given a list of {batch_size} posts from an {keyword} community on a social network.
    For each post in the list:
    - If the post is already in Indonesian, rephrase it into formal Bahasa Indonesia.
    - If the post is in a foreign language, translate and rephrase it into formal Bahasa Indonesia.

    Return your answer as a Python list of strings, containing only the final formal Indonesian version of each post.
    Return your answer as a Python list of {batch_size} strings, in the same order as the input.
    Do NOT include the original text or any translation notes—only the final formal Indonesian versions.

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

                print(f"[Batch] Attempt {attempt + 1}")
                response = await client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=4096,
                    model=os.getenv("AZURE_OPENAI_DEPLOYMENT")
                )

                output = response.choices[0].message.content or ""
                print("Completion:", output)

                # Parse output safely
                augmented = ast.literal_eval(output)

                return augmented

            except Exception as e:
                attempt += 1
                print(f"[Batch] Error on attempt {attempt}: {e}")
                await asyncio.sleep(1)

        # If all retries fail, fallback
        print("[Batch] All retries failed—returning original batch.")
        return tweets
            
        
    async def augment_all_batches(
        self,
        all_tweets: list,
        keyword: str,
        api_key: str,
        base_url: str = None
    ):
        batch_size=10
        temperature=0.5
        tokens=4000
        top_p=1
        # Divide tweets into batches
        print("Membuat eksplanasi")
        explanation = self.create_explanation(keyword)
        print("Membuat batches")
        batches = [all_tweets[i:i+batch_size] for i in range(0, len(all_tweets), batch_size)]
        print(f"Processing {len(batches)} batches of {batch_size}...")
        tasks = [
            self.create_augmentation_async(
                tweets=batch,
                batch_size=batch_size,
                temperature=temperature,
                tokens=tokens,
                top_p=top_p,
                keyword=keyword,
                explanation=explanation,
                api_key=api_key,
                base_url=base_url,
            )
            for batch in batches
        ]
        # Run all batches concurrently
        augmented_results = await asyncio.gather(*tasks)
        # Flatten results (list of lists → single list)
        all_augmented = [aug for batch in augmented_results for aug in batch]
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
        with open('../utils/kbba.txt', 'r', encoding='utf-8') as file:
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
        path = "./services/octis_data/"
        with open(path + 'vocabulary.txt', 'w') as file:
            for word in sorted(vocabulary):
                file.write(word + '\n')
        print("Vocabulary file created successfully!")
        tweet.to_csv(path +"corpus.tsv", index=False, sep="\t", header=False) 
        print("Corpus file created successfully!") 
    
            
    def create_vocabulary(self, tweets):

        vocabulary = set(word.lower() for text in tweets['tweets'] for word in text.split())
        # Save vocabulary to .txt file
        self.saving_vocab_corpus(vocabulary, tweets[['tweets','label']])
        
        print("Done!")
        
        return tweets

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
