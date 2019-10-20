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
def SupportVector(train_data,test_data)
{
    print('this function is for SVC')
    data= train_data
    data.describe(include="all")
    feature =  data.iloc[:,2:]
    target = data.iloc[:,1]
    replace = imp.fit_transform(feature)
    robust = RobustScaler()
    X_scale = robust.fit_transform(replace)
    from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
    cv = KFold(n_splits = 3, shuffle = True)
    X_train, X_test, y_train, y_test = train_test_split(X_scale, target, test_size = 0.3, stratify = target)
    svc = SVC(C = 1.1, gamma = 0.01)
    c = [1.10,100]
    gam = [0.01,0.1]
    cv = KFold(n_splits = 2, shuffle = True)
    param_dist = dict(C = c, gamma =gam)
    grid = RandomizedSearchCV(svc, param_dist, cv = cv, n_iter = 3)
    svc.fit(X_train, y_train)
    X_test_thing = test_data.iloc[:,1:]
    X_t = X_test_thing.replace('na', np.nan)
    X_new_t = imp.fit_transform(X_t)
    X_scale_t = robust.transform(X_new_t)
    y_pred = svc.predict(X_scale_t)
    export = test_data[['id']]
    export['target'] = y_pred
    export.to_csv('prediction.csv')

    }
def neuralnetworks(train_data,test_data)
{
    print('this function is for the neural networks program')
    train_d2=imp.fit_transform(train_data)
    train_d2[0,:]
    feature =  train_d2[:,2:]
    target = train_d2[:,1]
    train_d3 = imp.fit_transform(feature)
    X_train, X_test, y_train, y_test = train_test_split(feature, target, stratify = target)
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)
    model= tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(170,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(85, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.sigmoid))

    model.compile(optimizer='nadam',decay=1e-6, loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, validation_data=[X_test, y_test])
    print(model.evaluate(X_test, y_test))
    model.save('conoco.model')
    new_model= tf.keras.models.load_model('conoco.model')
    test_d=test_data;
    targ_t.replace('na', np.nan, inplace = True)
    targ_im = imp.fit_transform(targ_t)
    final_xtest = tf.keras.utils.normalize(targ_im, axis=1)
    prediction = new_model.predict(final_xtest)
    export = test_d[['id']]
    y_pred = binarize(prediction[:,1].reshape(-1,1),threshold=0.49)
    export['target'] = y_pred
    export.to_csv('prediction.csv')

    }

# Run algorithm
switchAlgos(Algorithm)
