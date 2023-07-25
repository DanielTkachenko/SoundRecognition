import pandas as pd
from sklearn import preprocessing
from sklearn import svm, neighbors
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def create_and_fit_model():
    data = pd.read_csv('data/dataset_1.csv', encoding='cp1251')
    # extracting feature table (Х), and labels table (у)
    X = data.iloc[:, 2:-1]
    #X = preprocessing.StandardScaler().fit(X).transform(X)
    y = data["label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
    # create and fit model
    model = neighbors.KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Train set Accuracy: ", metrics.accuracy_score(y_train, model.predict(X_train)))
    print("Test set Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model