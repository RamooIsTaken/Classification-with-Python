from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


"""df = pd.read_csv('data.csv')
df.drop('id',axis=1,inplace=True) #id kolunu kaldırıldı
df.drop('Unnamed: 32',axis=1,inplace=True) # NAN değerler düzeltildi
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0}) # İlk satırdaki karakterleri sayılara çevirdik 
X = df.drop('diagnosis',axis=1)
y = df.diagnosis"""

def prepair(dosya):
    df = pd.read_csv('{}'.format(dosya))
    df.drop('id',axis=1,inplace=True) #id kolunu kaldırıldı
    df.drop('Unnamed: 32',axis=1,inplace=True) # NAN değerler düzeltildi
    df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0}) # İlk satırdaki karakterleri sayılara çevirdik 
    X = df.drop('diagnosis',axis=1)
    y = df.diagnosis
    
    return X,y 


def crossValidate(model,X,y,cv):
    result = cross_validate(model,X,y,cv=cv)
    print("Cross_validate scores",result['test_score'])
    print("Cross_validate score ortalaması ",result['test_score'].mean())
    scores = cross_val_score(model,X,y,cv=cv,scoring="roc_auc")
    print("AUC:", scores.mean())

    return result,scores


def modelEgit(model,data) :
    X,y = prepair('{}'.format(data))

    trainX,testX,trainY,testY =  train_test_split(X,y,test_size=0.3,random_state=0)

    model.fit(trainX,trainY)
    print("Model Score : ", model.score(testX,testY))
    
    result, scores = crossValidate(model,X,y,8)
    
    yPred = model.predict(X)
    confusionMat = confusion_matrix(y,yPred)
    tn, fp, fn, tp = confusionMat.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)

    cm = confusion_matrix(y,yPred)
    print("Confusion Matrix : \n", cm)
    
    cr = classification_report(y, yPred)
    print("Classification Report : \n", cr)

    auc = roc_auc_score(y, yPred)
    print("AUC: ", auc)


"""model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
model = KNeighborsClassifier(n_neighbors=6)
model = DecisionTreeClassifier(random_state=0)"""

model = GaussianNB()  
modelEgit(model,'data.csv')

