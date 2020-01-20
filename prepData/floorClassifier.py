import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn import metrics
from sklearn.impute import SimpleImputer

def trainClassifier(trainingData):
	dataset = pd.read_csv(trainingData)

	X = dataset.iloc[:,0:3].values
	y = dataset.iloc[:,3].values.astype('int')

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)

	clf = RandomForestClassifier(n_estimators=100)
	clf.fit(X_train, y_train)

	y_pred = clf.predict(X_test)

	y_pred_list = y_pred.tolist()
	yPredDF = pd.DataFrame({'FloorPrediction':y_pred_list})

	y_test_list = y_test.tolist()
	yTestDF = pd.DataFrame({'Floors':y_test_list})

	XTestDF = pd.DataFrame(X_test, columns=['PUC', 'Height_ft', 'Area_ft'])

	testDF = pd.concat([XTestDF,yTestDF,yPredDF],axis=1)
	#testDF.to_csv('testDataOutput.csv')

	print('Floor Classifier Stats:')
	print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
	print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
	print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 

	return clf, X_test, y_test, y_pred

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    print('classes1', classes)
    classes = classes[unique_labels(y_true, y_pred)]
    print('classes2', classes) 
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def floorCountPrediction(PUC, Height, Area, Classifier):

	floors = Classifier.predict([[PUC, Height, Area]])
	floors = int(float(floors))
	return floors


if __name__ == '__main__':
	trainingData = 'floorClassifierTrainingData.csv'
	Classifier, class_names, y_test, y_pred = trainClassifier(trainingData)

	# Building properties
	PUC = 1
	Height = 20
	Area = 2000

	floors = floorCountPrediction(PUC, Height, Area, Classifier)

	print(floors, ' floors')


	classes = classes[unique_labels(y_true, y_pred)]
	print(confusion_matrix(y_test, y_pred, labels=classes))
	'''

	np.set_printoptions(precision=2)

	# Plot non-normalized confusion matrix
	plot_confusion_matrix(y_test, y_pred, classes=class_names[:,0],
	                      title='Confusion matrix, without normalization')

	# Plot normalized confusion matrix
	plot_confusion_matrix(y_test, y_pred, classes=class_names[:,0], normalize=True,
	                      title='Normalized confusion matrix')

	'''

	print('ytest {}'.format(y_test))
	print('ypred {}'.format(y_pred))
	print('class names {}'.format(class_names[:,0]))

	plt.show()

else:
	pass