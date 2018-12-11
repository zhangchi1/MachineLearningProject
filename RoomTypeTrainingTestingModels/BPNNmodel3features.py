
# The numpy.array data type will be used to store our data
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# save the confusion matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

import keras
import tensorflow as tf

#read training samples, eko and equation floor plans
trainingCSV = '/Users/chizhang/2018Fall/cs542 Machine Learning/Project/MachineLearningProject/DataProcessing/EKO_features.csv'
trainingCSV_equ = '/Users/chizhang/2018Fall/cs542 Machine Learning/Project/MachineLearningProject/DataProcessing/EQUATION_features.csv'

trainDf_eko = pd.read_csv(trainingCSV)
trainDf_equ = pd.read_csv(trainingCSV_equ)

# concatenate eko and equation floor plans
trainDf_eko['id'] = 'eko_' + trainDf_eko['id']
trainDf_equ['id'] = 'equ_' + trainDf_equ['id']
trainDf = pd.concat([trainDf_eko, trainDf_equ], ignore_index=True)
x_trainDf = trainDf.iloc[:,1:4]
x_train = np.array(x_trainDf)
y_trainDf = trainDf.iloc[:,-1]
y_train = np.array(y_trainDf)

#read testing samples oxygen 186 samples
testingCSV = '/Users/chizhang/2018Fall/cs542 Machine Learning/Project/MachineLearningProject/DataProcessing/OXYGEN_features.csv'
testingDf = pd.read_csv(testingCSV)
x_testingDf = testingDf.iloc[:,1:4]
x_testing = np.array(x_testingDf)
y_testingDf = testingDf.iloc[:,-1]
y_testing = np.array(y_testingDf)

# normalize x_train and x_test
scaler_x_train = preprocessing.StandardScaler()
scaler_x_train.fit(x_train)
norm_x_train = scaler_x_train.transform(x_train)

scaler_x_train = preprocessing.StandardScaler()
scaler_x_train.fit(x_testing)
norm_x_testing = scaler_x_train.transform(x_testing)



# Cross validation on training samples eko + equ
cv_X_train, cv_X_test, cv_y_train, cv_y_test = train_test_split(norm_x_train, y_train, test_size=0.2)




# define a BP neural network: 
# x_train, y_train are np arrary
# if layers = 2, hyper_paras = [] (list len = 0)
# if layers = 3, hyper_paras = [5] (list len = 1)
# if layers = 4, hyper_paras = [5, 10] (list len = 2)
def BPNNmodel(x_train, y_train, activation_f, layers, hyper_paras, numFeatures):
    modelNN = keras.Sequential(name='NNmodel')
    if(layers == 2):
        print('layers = 2')
        modelNN.add(keras.layers.Dense(6, input_dim=numFeatures, activation=tf.nn.softmax))
    elif (layers == 3):
        #input 3 features, output hyper_paras[0] h_para
        modelNN.add(keras.layers.Dense(hyper_paras[0], input_dim=numFeatures, activation=activation_f))
        # output layer
        modelNN.add(keras.layers.Dense(6, activation=tf.nn.softmax))
    elif (layers == 4):
        #input 3 features, output hyper_paras[0] h_para
        modelNN.add(keras.layers.Dense(hyper_paras[0], input_dim=numFeatures, activation=activation_f))
        #input hyper_paras[0] h_para, output hyper_paras[1] h_para
        modelNN.add(keras.layers.Dense(hyper_paras[1], activation=activation_f))
        # ouput layer
        modelNN.add(keras.layers.Dense(6, activation=tf.nn.softmax))
    
    print('Compiling...')
    modelNN.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    # fit using cv training set
    print('Fitting...')
    modelNN.fit(x_train,y_train,epochs=500,verbose=2, batch_size=32)
    print('fit done')
    return modelNN
#test 3 features
modelNN = BPNNmodel(norm_x_train, y_train, 'relu', 2, [30], 3)

# tuning parameters
def tuningNNParameters(x_train, y_train, x_test, y_test, iterations, numFeatures):
    acc_table = pd.DataFrame(columns=['Model','Training_acc','Testing_acc'])
    activ_funcs = ['sigmoid', 'relu']
    layers = [2,3,4]
    h_paras = [5,10,30,50,100]
    h_paras_2layer = [[5,5],[5,10],[10,5],[10,10],[10,30],[30,10],[30,30],[30,50],[50,30],[50,50], [50,100], [100,50],[100,100]]
    #count number of models
    num_models = 0 
    modelName = ''
    for layers_index in range(3):
        # 2 layers
        if (layers_index == 0):
            modelNN = BPNNmodel(x_train, y_train, activ_funcs[1], layers[layers_index], h_paras[0:0], numFeatures)
            # test accuracy on train set
            preds = modelNN.predict(x_train)
            preds1 = np.argmax(preds, axis=1)
            acc_train = accuracy_score(y_train, preds1)
            # test accuracy on testing set
            testing_preds = modelNN.predict(x_test)
            testing_preds2 = np.argmax(testing_preds, axis=1)
            acc_test = accuracy_score(y_test,testing_preds2)
            modelName = activ_funcs[1] + 'layer: ' + str(layers[layers_index])
            acc_table.loc[num_models,:] = [modelName, acc_train, acc_test]
            num_models+=1
            print('saving model: ' + modelName)
        # 3 layers    
        elif (layers_index == 1):
            for h_para_index in range (len(h_paras)):
                for activ_index in range(len(activ_funcs)):
                    modelNN = BPNNmodel(x_train, y_train, activ_funcs[activ_index], layers[layers_index], [h_paras[h_para_index]], numFeatures)
                    # test accuracy on train set
                    preds = modelNN.predict(x_train)
                    preds1 = np.argmax(preds, axis=1)
                    acc_train = accuracy_score(y_train, preds1)
                    # test accuracy on testing set
                    testing_preds = modelNN.predict(x_test)
                    testing_preds2 = np.argmax(testing_preds, axis=1)
                    acc_test = accuracy_score(y_test,testing_preds2)
                    modelName = activ_funcs[activ_index] + 'layer: ' + str(layers[layers_index]) + 'h_para: ' + str(h_paras[h_para_index])
                    acc_table.loc[num_models,:] = [modelName, acc_train, acc_test]
                    num_models+=1
                    print('saving model: ' + modelName)

        # 4 layers
        elif (layers_index == 2):
            for h_para_index in range (len(h_paras_2layer)):
                for activ_index in range(len(activ_funcs)):
                    modelNN = BPNNmodel(x_train, y_train, activ_funcs[activ_index], layers[layers_index], h_paras_2layer[h_para_index], numFeatures)
                    # test accuracy on train set
                    preds = modelNN.predict(x_train)
                    preds1 = np.argmax(preds, axis=1)
                    acc_train = accuracy_score(y_train, preds1)
                    # test accuracy on testing set
                    testing_preds = modelNN.predict(x_test)
                    testing_preds2 = np.argmax(testing_preds, axis=1)
                    acc_test = accuracy_score(y_test,testing_preds2)
                    modelName = activ_funcs[activ_index] + 'layer: ' + str(layers[layers_index]) + 'h_para: ' + str(h_paras_2layer[h_para_index])
                    acc_table.loc[num_models,:] = [modelName, acc_train, acc_test]
                    num_models+=1
                    print('saving model: ' + modelName)
    return acc_table

acc_table1 = tuningNNParameters(norm_x_train, y_train, norm_x_testing, y_testing, 1, 3)
#acc_table1_dir = '/Users/chizhang/2018Fall/cs542 Machine Learning/Project/MachineLearningProject/BPNNModels/acc_table1.csv'
#acc_table1.to_csv(acc_table1_dir, index=False)     
acc_table2 = tuningNNParameters(norm_x_train, y_train, norm_x_testing, y_testing, 1, 3)
acc_table3 = tuningNNParameters(norm_x_train, y_train, norm_x_testing, y_testing, 1, 3)
acc_table4 = tuningNNParameters(norm_x_train, y_train, norm_x_testing, y_testing, 1, 3)

acc_table = acc_table1 + acc_table2+acc_table3 + acc_table4
acc_table.loc[:,'Training_acc'] =  acc_table.loc[:,'Training_acc'] /4
acc_table.loc[:,'Testing_acc'] =  acc_table.loc[:,'Testing_acc'] /4
acc_table.loc[:,'Model'] = acc_table1.loc[:,'Model']

# plot the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(title+'.png', dpi=300)
    
# =============================================================================
#  compute the confusion matrix, based on the best model
#  inputs: @y_testing, @testing_preds
# =============================================================================
#use relu 3layers, h_para=100
modelNN = BPNNmodel(norm_x_train, y_train, 'relu', 3, [100], 3)
modelNN.save('3features BPNN Relu layer=3, h_para=100'+".h5")
# test accuracy on train set
preds = modelNN.predict(norm_x_train)
preds1 = np.argmax(preds, axis=1)
acc_train = accuracy_score(y_train, preds1)

# test accuracy on testing set
testing_preds = modelNN.predict(norm_x_testing)
testing_preds1 = np.argmax(testing_preds, axis=1)
acc_test = accuracy_score(y_testing,testing_preds1)


cnf_matrix = confusion_matrix(y_testing, testing_preds1)
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plt.figure()
class_names_numerical = ['0','1','2','3','4','5']
class_names_str = ['living room','bedroom','bathroom','WC','intersection','kitchen']

#save the ouput confusion matrix as an image
plot_confusion_matrix(cnf_matrix, classes=class_names_str,
                      title='3features BPNN Relu layer=3, h_para=100')

