import numpy as np
from octis.dataset.dataset import Dataset
from octis.models.ETM import ETM
from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence
from joblib import Parallel, delayed

class EmbeddedTopicModeling:
    def __init__(self):
        dataset_path = "./services/octis_data/"
        self.dataset = Dataset()
        self.dataset.load_custom_dataset_from_folder(dataset_path)

    
    def create_and_train_etm(self, dataset, num_topics):
        
        # model = ETM(
        #     num_topics=num_topics,
        #     num_epochs=100,
        #     batch_size=256,
        #     dropout=0.2,
        #     # embedding_size=100,
        #     t_hidden_size=256,
        #     wdecay=1e-6,
        #     lr=0.002,
        #     optimizer='adam'
        # )
        
        model = ETM(
            num_topics=num_topics,
            num_epochs=100,
            batch_size=256,
            dropout=0.3,
            activation="tanh",
            embeddings_path="idwiki_word2vec_100_new_lower.txt",
            embeddings_type="word2vec",
            t_hidden_size=512,
            wdecay=1e-5,
            lr=0.001,
            optimizer='SGD'
        
        )
        # model = ETM(
        #     num_topics=num_topics,        # Verify if num_topics matches dataset complexity
        #     num_epochs=200,               # Increased from 100
        #     batch_size=128,               # Reduced from 256 for finer updates
        #     dropout=0.5,                  # Increased from 0.2
        #     t_hidden_size=512,            # Doubled from 256
        #     wdecay=1e-5,                  # Increased from 1e-6
        #     lr=0.0005,                    # Reduced from 0.002
        #     optimizer='adamw',            # Changed from Adam to AdamW
        #     # scheduler='plateau',          # New: Reduce LR when validation loss stalls
        #     activation='leaky_relu'       # New: Better gradient flow
        # )
        model_output = model.train_model(dataset)
        
        return (num_topics, model, model_output)
    
    def generateTopic(self):
        best_coh = float("-inf")
        best_topic = None

        coh_score_list = []
        topics = range(1, 7)

        # 2) Parallel execution
        results = Parallel(n_jobs=-1)(
            delayed(self.create_and_train_etm)(self.dataset, topic)
            for topic in topics
        )

        print("\n=== Summary ===")
        # 3) Process results
        for num_topics, _, model_output in results:
            coh_score = self.evaluate_coherence(self.dataset, model_output)
            print(f"[{num_topics} topics] Coherence: {coh_score:.4f}")

            if coh_score > best_coh:
                best_coh = coh_score
                best_topic = num_topics

        print(f"\nBest model has {best_topic} topics with coherence={best_coh:.4f}",end="\n")

        model = self.create_and_train_etm(self.dataset,best_topic)
        
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
            
            print("Doc {}: topic={}, prob={}".format(i+1, topic_index+1, probability))

            data_tweet[i].update({
                "topic": str(topic_index),
                "probability": str(probability)
            })
            documents_probability.append(data_tweet[i])

        return documents_probability


    def evaluate_coherence(self, dataset, model_output):
        coh = Coherence(texts=dataset.get_corpus(),topk=10,
                    measure='c_v')
        
        return coh.score(model_output)
