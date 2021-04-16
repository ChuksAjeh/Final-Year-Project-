# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from code.model_using_vader import Vader_Model
from code.render_vader_data import RenderVaderData
from code.model_using_flair import Flair_Model
from code.model_using_textblob import Textblob_Model
from code.render_textblob_data import RenderTextblobData
from code.render_flair_data import RenderFlairData
from code.model_using_flair_custom_training import Flair_Model_Custom
from tabulate import tabulate
import matplotlib.pyplot as plt
def print_hi(name):
    #x = RenderVaderData()
    # x.render_open_graph()
    # x.render_close_graph()
    #x.render_price_graph()
    #x.plot_points_on_graph()
    #vader_model = Vader_Model()
    # print(vader_model.calc_accuracy())
    #x.plot_points_on_graph()
    #x.plot_points_on_graph()
    #print(path_one.get_pos_data())
    #x = Flair_Model()
    #print(x.calc_accuracy())
    #x = Textblob_Model()
    # print(x.get_polartiy())
    #y = RenderFlairData()
    #y.plot_points_on_graph()

    #z= Textblob_Model()
    #print(z.get_polartiy())
    #print(z.calc_accuracy())
    #xy = RenderTextblobData()
    #xy.plot_points_on_graph()

    #---------------------Render ModelAccuracy Data--------------------
    # model_names =["Vader Model","TextBlob Model", "Flair Model"]
    # accuracy = []
    # accuracy.append(vader_model.calc_accuracy())
    # accuracy.append(z.calc_accuracy())
    # accuracy.append(x.calc_accuracy())
    # plt.xlabel("Model",fontsize=14)
    # plt.ylabel("Accuracy in %",fontsize=14)
    # plt.title("Model Sentiment Prediction Accuracy")
    # plt.bar(model_names,accuracy,color=['red','blue','green'])
    # plt.legend()
    # plt.show()
    # plt.savefig("Accuracy_Levels_Of_Models.png")
    
    custom = Flair_Model_Custom()
    print(custom.data.head(5))




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
