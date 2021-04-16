from flair.models import TextClassifier
import pandas as pd
from flair.data import Sentence
from flair.models import TextClassifier
from segtok.segmenter import split_single
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentLSTMEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from pathlib import Path
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.data import Corpus
from flair.datasets import SENTEVAL_SST_GRANULAR
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
import flair, torch

class Flair_Model_Custom:

    def __init__(self):
        self.classifier = TextClassifier.load('en-sentiment')
        self.data = pd.read_csv('C:\\Users\\chuks\\PycharmProjects\\FYP\\dataset\\dataset.csv')
        self.data_copy = self.data.copy()
        self.clean_up()
        # self.scores = self.data['Top1'].apply(vader.polarity_scores).tolist()
        self.top_headline = self.data_copy.pop('Top1')
        self.dates = self.data_copy.pop('Date')
        self.labels = self.data_copy.pop('Label')
        # self.scores_df = pd.DataFrame(self.scores)
        self.dates_df = pd.DataFrame(self.dates)
        self.rename_columns()
        #self.create_dataset()
        self.create_model()
        self.labels_df = pd.DataFrame(self.labels)
        self.top_headline_df = pd.DataFrame(self.top_headline)
        # # print(top_headline_df.head(5))
        # # self.data_and_sentiment = pd.concat([self.top_headline, self.scores_df], axis=1)
        # self.get_sentiment()
        self.final_df = pd.concat([self.dates_df, self.top_headline_df['scores_sum']], axis=1)
        self.final_df_copy = self.final_df.set_index('Date')
        self.labels_with_date = pd.concat([self.dates_df,self.labels_df],axis=1)
        self.labels_with_date_copy = self.labels_with_date.set_index('Date')
        self.sentiment_and_labels = pd.concat([self.final_df_copy['scores_sum'],self.labels_with_date_copy],axis=1)
        # # self.pos_vals = self.final_df['pos'].values
        # # self.neg_vals = self.final_df['neg'].values
        

    def get_data_head(self):
        return self.data.head()

    def clean_up(self):
        self.data['Top23'].fillna(self.data['Top23'].median, inplace=True)
        self.data['Top24'].fillna(self.data['Top24'].median, inplace=True)
        self.data['Top25'].fillna(self.data['Top25'].median, inplace=True)
        self.data.reset_index()
        # rearrange columns
        self.data = self.data[['Date', 'Label', 'Top1', 'Top2', 'Top3', 'Top4', 'Top5',
                               'Top6', 'Top7', 'Top8', 'Top9', 'Top10', 'Top11',
                               'Top12', 'Top13', 'Top14', 'Top15', 'Top16', 'Top17',
                               'Top18', 'Top19', 'Top20', 'Top21', 'Top22', 'Top23', 'Top24', 'Top25']]
        #
        self.data.drop(['Top2'],axis=1,inplace=True)
        self.data.drop(['Top3'],axis=1,inplace=True)
        self.data.drop(['Top4'],axis=1,inplace=True)
        self.data.drop(['Top5'],axis=1,inplace=True)
        self.data.drop(['Top6'],axis=1,inplace=True)
        self.data.drop(['Top7'],axis=1,inplace=True)
        self.data.drop(['Top8'],axis=1,inplace=True)
        self.data.drop(['Top9'],axis=1,inplace=True)
        self.data.drop(['Top10'],axis=1,inplace=True)
        self.data.drop(['Top11'],axis=1,inplace=True)
        self.data.drop(['Top12'],axis=1,inplace=True)
        self.data.drop(['Top13'],axis=1,inplace=True)
        self.data.drop(['Top14'],axis=1,inplace=True)
        self.data.drop(['Top15'],axis=1,inplace=True)
        self.data.drop(['Top16'],axis=1,inplace=True)
        self.data.drop(['Top17'],axis=1,inplace=True)
        self.data.drop(['Top18'],axis=1,inplace=True)
        self.data.drop(['Top19'],axis=1,inplace=True)
        self.data.drop(['Top20'],axis=1,inplace=True)
        self.data.drop(['Top21'],axis=1,inplace=True)
        self.data.drop(['Top22'], axis=1,inplace=True)
        self.data.drop(['Top23'],axis=1,inplace=True)
        self.data.drop(['Top24'],axis=1,inplace=True)
        self.data.drop(['Top25'],axis=1,inplace=True)
        self.data.drop_duplicates()

    # def get_sentiment_data(self):
    #     return self.final_df
    #
    # def get_pos_data(self):
    #     return self.pos_vals
    #
    # def get_neg_data(self):
    #     return self.neg_vals

    # def get_headline(self):
    #     return self.top_headline
    #
    # def get_headline_values(self):
    #     return self.top_headline.tolist()

    # create list of sentences:
    def make_sentences(self, text):
        sentences = [sentence for sentence in split_single(text)]
        return sentences

    def get_sentiment_data(self, sentence):
        if sentence == "":
            return 0
        text = Sentence(sentence)
        # stacked_embeddings.embed(text)
        self.classifier.predict(text)
        value = text.labels[0].to_dict()['value']
        if value == 'POSITIVE':
            result = text.to_dict()['labels'][0]['confidence']
        else:
            result = -(text.to_dict()['labels'][0]['confidence'])
        return round(result, 3)

    def get_scores(self, sentences):
        """ Call predict on every sentence of a text """
        results = []

        for i in range(0, len(sentences)):
            results.append(self.get_sentiment_data(sentences[i]))
        return results

    def get_sum(self, scores):

        result = round(sum(scores), 3)
        return result

    def get_sentiment(self):
        self.top_headline_df['sentences'] = self.top_headline_df['Top1'].apply(self.make_sentences)
        self.top_headline_df['scores'] = self.top_headline_df['sentences'].apply(self.get_scores)
        self.top_headline_df['scores_sum'] = self.top_headline_df.scores.apply(self.get_sum)
        return self.top_headline_df['scores_sum']

    def calc_accuracy(self):
        accCount =0
        for i in range(0,len(self.sentiment_and_labels)):
            if(self.sentiment_and_labels['scores_sum'][i]<0 and self.sentiment_and_labels['Label'][i]==0):
                accCount = accCount + 1
            if (self.sentiment_and_labels['scores_sum'][i] >= 0 and self.sentiment_and_labels['Label'][i] == 1):
                accCount = accCount + 1
        return (accCount/len(self.sentiment_and_labels))*100
    
    def rename_columns(self):
        self.data = self.data[['Label','Top1']].rename(columns={"Label":"label","Top1":"text"})
        
    def create_dataset(self):
        #self.data['label'] = '__label__' + self.data['label'].astype(str)
        self.data.iloc[0:int(len(self.data) * 0.8)].to_csv('train.csv', sep='\t', index=False, header=False)
        self.data.iloc[int(len(self.data) * 0.8):int(len(self.data) * 0.9)].to_csv('test.csv', sep='\t', index=False, header=False)
        self.data.iloc[int(len(self.data) * 0.9):].to_csv('dev.csv', sep='\t', index=False, header=False)

    def create_model(self):
        #flair.device = torch.device('cuda:0')
        data_folder = 'C:/Users/chuks/PycharmProjects/FYP/data'

        # column format indicating which columns hold the text and label(s)
        column_name_map = {1: "text", 0: "label"}

        # load corpus containing training, test and dev data and if CSV has a header, you can skip it
        corpus: Corpus = CSVClassificationCorpus(data_folder,
                                                 column_name_map,
                                                 skip_header=True,
                                                 delimiter='\t',  # tab-separated files
                                                 )


        #corpus: Corpus = SENTEVAL_SST_GRANULAR()

        # 2. create the label dictionary
        label_dict = corpus.make_label_dictionary()

        # 3. make a list of word embeddings
        word_embeddings = [WordEmbeddings('glove')]

        # 4. initialize document embedding by passing list of word embeddings
        # Can choose between many RNN types (GRU by default, to change use rnn_type parameter)
        document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=256)

        # 5. create the text classifier
        self.classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)

        # 6. initialize the text classifier trainer
        trainer = ModelTrainer(self.classifier, corpus)

        # 7. start the training
        trainer.train('resources/taggers/senteval_sst_granular',
                      learning_rate=0.1,
                      mini_batch_size=32,
                      anneal_factor=0.5,
                      patience=5,
                      max_epochs=5)