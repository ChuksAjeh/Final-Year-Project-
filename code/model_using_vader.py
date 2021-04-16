import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class Vader_Model:

    def __init__(self):
        vader = SentimentIntensityAnalyzer()
        self.data = pd.read_csv('C:\\Users\\chuks\\PycharmProjects\\FYP\\dataset\\dataset.csv')
        self.clean_up()
        self.scores = self.data['Top1'].apply(vader.polarity_scores).tolist()
        self.top_headline = self.data.pop('Top1')
        self.dates = self.data.pop('Date')
        self.labels = self.data.pop('Label')
        self.scores_df = pd.DataFrame(self.scores)
        self.dates_df = pd.DataFrame(self.dates)
        self.labels_df = pd.DataFrame(self.labels)
        self.top_headline_df = pd.DataFrame(self.top_headline)
        # print(top_headline_df.head(5))
        self.data_and_sentiment = pd.concat([self.top_headline, self.scores_df], axis=1)
        self.final_df = pd.concat([self.dates_df, self.data_and_sentiment], axis=1)
        self.final_df_copy = self.final_df.set_index('Date')
        self.labels_with_date = pd.concat([self.dates_df, self.labels_df], axis=1)
        self.labels_with_date_copy = self.labels_with_date.set_index('Date')
        self.sentiment_and_labels = pd.concat([self.final_df_copy['compound'], self.labels_with_date_copy], axis=1)
        self.pos_vals = self.final_df['pos'].values
        self.neg_vals = self.final_df['neg'].values

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

    def get_sentiment_data(self):
        return self.final_df

    def get_pos_data(self):
        return self.pos_vals

    def get_neg_data(self):
        return self.neg_vals

    def get_headline(self):
        return self.top_headline

    def get_headline_values(self):
        return self.top_headline.tolist()

    def classfication_Report(self):
        return None

    def calc_accuracy(self):
        accCount =0
        for i in range(0,len(self.sentiment_and_labels)):
            if(self.sentiment_and_labels['compound'][i]<0 and self.sentiment_and_labels['Label'][i]==0):
                accCount = accCount + 1
            if (self.sentiment_and_labels['compound'][i] >= 0 and self.sentiment_and_labels['Label'][i] == 1):
                accCount = accCount + 1
        return (accCount/len(self.sentiment_and_labels))*100