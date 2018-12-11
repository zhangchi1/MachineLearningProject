import pandas as pd


# convert y label to numerical values
# input a csv sample features, convert the y label into numerical values
def numericalLabelConverter(featureCSV):
    samples = pd.read_csv(featureCSV, header=None, names=['id','area', 'compactness', 'density', 'y_label'])
    # drop rgt type, nan value
    samples = samples[samples.y_label != 'rgt']
    samples = samples[pd.notna(samples.y_label)]
    samples.loc[samples.y_label == 'k', 'y_label'] = 5
    samples.loc[samples.y_label == 'l', 'y_label'] = 0
    samples.loc[samples.y_label == 'be', 'y_label'] = 1
    samples.loc[samples.y_label == 'ba', 'y_label'] = 2
    samples.loc[samples.y_label == 'ba/wc', 'y_label'] = 2
    samples.loc[samples.y_label == 'wc', 'y_label'] = 3
    samples.loc[samples.y_label == 'i', 'y_label'] = 4
    #samples.loc[samples.y_label == 'dgt', 'y_label'] = 4
    samples.to_csv(featureCSV, index=False)



featureCSV = '/Users/chizhang/2018Fall/cs542 Machine Learning/Project/MachineLearningProject/DataProcessing/EKO_features.csv'
numericalLabelConverter(featureCSV)


featureCSV = '/Users/chizhang/2018Fall/cs542 Machine Learning/Project/MachineLearningProject/DataProcessing/EQUATION_features.csv'
numericalLabelConverter(Equa_featureCSV)