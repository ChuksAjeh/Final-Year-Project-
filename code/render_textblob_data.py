import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import yfinance as yf

from code.model_using_textblob import Textblob_Model

class RenderTextblobData:
    data = None
    ticker = None
    textblob_model = None
    full_data = None
    def __init__(self):

        self.tickers = 'SPY'
        self.data = yf.download(self.tickers, start="2008-07-08", end="2016-07-02", group_by='ticker', interval="1mo")
        # self.data.drop(['Volume'],axis=1)
        self.textblob_model = Textblob_Model()
        #zero volume values
        self.data['Volume'] = self.data['Volume'].apply(self.zero).astype(float)
        #Drop NaN values:
        self.data.dropna(inplace=True)

        self.textblob_model.final_df_copy.index = pd.to_datetime(self.textblob_model.final_df_copy.index)
        #drop the column:
        self.data.drop(['Volume'],axis=1)
        self.data['Volume'] = self.textblob_model.final_df_copy['compound']
        self.data.dropna(inplace=True)

    # convert into a dataframe and return it.
    def get_data(self):
        return self.data

    def get_tickers(self):
        return self.tickers

    def open_data_to_plot(self):
        open_ticker_data = self.data['Open']
        return open_ticker_data

    def close_data_to_plot(self):
        close_ticker_data = self.data['Close']
        return close_ticker_data

    def plot_points_on_graph(self):
        data_df = pd.DataFrame(self.data)
        apd = mpf.make_addplot(self.data['Open'], type='line')
        mpf.plot(data_df, title="S&P 500 Data using TextBlob Model",ylabel_lower="Sentiment", type='candle', style='charles', mav=3, volume=True, addplot=apd, figratio=(18, 10))
        #mpf.savefig('TextBlob_Model_Price_Sentiment.png')

    # TODO: Throws timeout error needs fixing:
    # def plot_polarity(self):
    #     simple_model = Vader_Model()
    #     sentiment_data = simple_model.final_df['compound'].tolist()
    #
    #     dates = simple_model.dates.tolist()
    #
    #     plt.figure(figsize=(100,100))
    #     plt.xticks(fontsize=20)
    #     plt.yticks(fontsize=20)
    #     plt.bar(dates[:9],sentiment_data[:9])
    #     plt.show()
    #     #mpf.plot(pos_df,volume=True)

    # def generate_word_cloud(self):
    #     simple_model = Vader_Model()
    #     text = ' '.join(simple_model.get_headline_values())
    #
    #     wordcloud  = wc.WordCloud(stopwords=stopwords).generate(text)
    #     plt.imshow(wordcloud,interpolation='bilinear')
    #     plt.axis("off")
    #     plt.show()

    def get_price_point(self, date):
        open_data = self.open_data_to_plot()
        specific_open_price = open_data[date]
        return specific_open_price

    def get_type(self):
        print(self.data)

    def zero(self,x):
        return x*0
# x = RenderData()
# # x.render_open_graph()
# # x.render_close_graph()
# x.render_price_graph()
# x.plot_points_on_graph()
# print(x.get_price_point())
# print(x.open_data_to_plot('SPY'))
# print(data)
