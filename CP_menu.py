import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# * Data Wrangling Definitions ================================================================

def HeadOfData(data):
    data.head()

def DescribeData(data):
    data.describe()

def FillMissingToZero(data):
    data.fillna(0)

def FillMissingToMean(data):
    data.fillna(data.mean())

def CreateScatterPlot(data):
    data_target_name = "Y Value"
    data_target_name = data[data_target_name]
    fig = plt.figure(figsize=(15, 15))

    # The enumerate function will give us the index as well as the value
    for (i, column) in enumerate(list(data.columns)):
        if(column == data_target_name) or (column == "name"):
            continue
        plt.subplot(5,3,i)
        plt.scatter(data[column], data[data_target_name])
        plt.xlabel(column)
        plt.ylabel(data_target_name)

    plt.show()
    
def CreateHeatMap(data):
    corrmat = data.corr()
    f, ax = plt.subplots(figsize=(9, 9))
    f = sns.heatmap(corrmat, vmax=.8, square=True, annot=True, cbar=False)

# * Switch Definitions ========================================================================
# Switch Function for Processing Method
def switchProcess(argument, data):
    switcher = {
        "Show Head": HeadOfData(data),
        "Describe Data": DescribeData(data),
        "Fill Missing to 0": FillMissingToZero(data),
        "Fill Missing to Mean": FillMissingToMean(data),
        "Create ScatterPlot": CreateScatterPlot(data),
        "Create Heat Map": CreateHeatMap(data)
    }
    return switcher.get(argument, "Invalid Process")

# File for training
train_file = input("Please input training file:")

train_data = pd.read_csv(train_file, sep=",")

# Select Processing Method
ProcessingMethod = switchProcess(input("Please input Processing Method"), train_data)

# File for testing
test_file = input("Please input testing file:")

# Menu
# exit = False
# while exit != True:

# Switch function that 
def switchAlgos(argument):
    switcher = {
        "Algorithm0": "Januar:",
        "Algorithm1": "February",
        "Algorithm2": "March",
        "Algorithm3": "April",
        "Algorithm4": "May",
        "Algorithm5": "June",
        "Algorithm6": "July",
        "Algorithm7": "August",
        "Algorithm8": "September",
        "Algorithm9": "October",
        "Algorithm10": "November",
        "Algorithm11": "December"
    }
    switcher.get(argument, "Invalid Algorithm")

Algorithm = input("Please select your algorithm:")

# Run algorithm
switchAlgos(Algorithm)
