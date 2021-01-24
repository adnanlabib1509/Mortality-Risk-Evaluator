# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, cross_validate
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.feature_selection import chi2, mutual_info_classif
import sys
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from collections import Counter
from statistics import mean
from sklearn.metrics import plot_confusion_matrix


###########################################################
# Data Preprocessing
###########################################################

data=pd.read_csv("covid.csv")

#remove NA values from ICU. Only take covid positive patients
data=data[ (data.icu!=97) & (data.icu!=98) & (data.icu!=99)]
data=data[(data.covid_res==1)]

#create a new column to store values the vital status of patients
data["vital_status"]=np.where(data["date_died"]=="9999-99-99",0,1)
data=data.drop(["date_died"],axis=1)

#adjusting all NULL values
data.loc[:,data.columns!="age"] = data.loc[:,data.columns!="age"].replace(97, np.nan)
data.loc[:,data.columns!="age"] = data.loc[:,data.columns!="age"].replace(98, np.nan)
data.loc[:,data.columns!="age"] = data.loc[:,data.columns!="age"].replace(99, np.nan)
data.loc[:,data.columns!="age"] = data.loc[:,data.columns!="age"].replace(2,0)

#Calculate time between date of first symptoms and hospital admission date
data.entry_date = pd.to_datetime(data.entry_date, dayfirst=True, errors='coerce')
data.date_symptoms = pd.to_datetime(data.date_symptoms, dayfirst=True, errors='coerce')
data["time_diff"]=(data["entry_date"]-data["date_symptoms"]).dt.days

#drop columns which is not related to the classification task
data=data.drop(["id","patient_type","entry_date","date_symptoms","pregnancy","other_disease","contact_other_covid","covid_res"],axis=1)


#create the mortality risk column which will be the target variable with 3 classes
data["mortality_risk"]=np.where((data["icu"] == 1) & (data["vital_status"]==1), 1, 
                      (np.where((data["icu"] == 1) & (data["vital_status"]==0), 2, 
                      (np.where((data["icu"] == 0) & (data["vital_status"]==1), np.nan,3)))))
data=data.drop(["icu","vital_status","intubed"],axis=1)

#dropping all the NULL values
dropped_data=data.dropna()


#Dropping all NA labels (mortality risk)
data=data.dropna(subset=["mortality_risk"])

#Imputing all the NULL values using KNN imputer
imputer=KNNImputer(n_neighbors=3)
imputed_data=pd.DataFrame(data=imputer.fit_transform(data),columns=["sex","age","pneunomia","diabetes","copd","asthma","inmsupr","hypertension","cardiovascular","obesity","renal_chronic","tobacco","time_diff","mortality_risk"])


#############################################################
# Splitting the Dataset
#############################################################

#getting the x and y values
y=imputed_data.pop("mortality_risk")
le=LabelEncoder()
y= le.fit_transform(y)
x=imputed_data


#Used for fixing the imbalance nature of the dataset
sm = ADASYN()
x,y = sm.fit_resample(x, y)

#Feature Selection
c2 = SelectKBest(mutual_info_classif, k="all").fit(x,y)
x = c2.transform(x)

# splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.25, random_state = 101)

#############################################################
# Decision Tree
#############################################################

decision_tree_model=DecisionTreeClassifier()
decision_tree_model=decision_tree_model.fit(X_train,Y_train)
Y_pred_dt=decision_tree_model.predict(X_test)
print("Decision Tree Accuracy: ",metrics.accuracy_score(Y_test, Y_pred_dt))
print("Decision Tree Precision: ",metrics.precision_score(Y_test, Y_pred_dt,average="macro"))
print("Decision Tree Recall: ",metrics.recall_score(Y_test, Y_pred_dt,average="macro"))
print("Decision Tree F_score: ",metrics.f1_score(Y_test, Y_pred_dt,average="macro"))
print(metrics.confusion_matrix(Y_test, Y_pred_dt))
plot_confusion_matrix(decision_tree_model, X_test, Y_test,cmap="Blues")

#############################################################
# KNN
#############################################################

knn_model=KNeighborsClassifier(n_neighbors=9)
knn_model=knn_model.fit(X_train,Y_train)
Y_pred_knn=knn_model.predict(X_test)

print("KNN Accuracy: ",metrics.accuracy_score(Y_test, Y_pred_knn))
print("KNN Precision: ",metrics.precision_score(Y_test, Y_pred_knn,average="macro"))
print("KNN Recall: ",metrics.recall_score(Y_test, Y_pred_knn,average="macro"))
print("KNN F_score: ",metrics.f1_score(Y_test, Y_pred_knn,average="macro"))
print(metrics.confusion_matrix(Y_test, Y_pred_knn))
plot_confusion_matrix(knn_model, X_test, Y_test,cmap="Blues")

##############################################################
# Random Forest
##############################################################

rf_model=RandomForestClassifier()
rf_model=rf_model.fit(X_train,Y_train)
Y_pred_rf=rf_model.predict(X_test)
print("Random Forest Accuracy: ",metrics.accuracy_score(Y_test, Y_pred_rf))
print("Random Forest Precision: ",metrics.precision_score(Y_test, Y_pred_rf,average="macro"))
print("Random Forest Recall: ",metrics.recall_score(Y_test, Y_pred_rf,average="macro"))
print("Random Forest F_score: ",metrics.f1_score(Y_test, Y_pred_rf,average="macro"))
print(metrics.confusion_matrix(Y_test, Y_pred_rf))
plot_confusion_matrix(rf_model, X_test, Y_test,cmap="Blues")














