##############################################################################################################################################################
##############################################################################################################################################################
#########################################################Individual Project###################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
import pandas as pd
import numpy as np

KS_df=pd.read_excel("/Users/collinwang/Desktop/Individual Project/Final/Kickstarter.xlsx")####Import dataset 

###########Data Preprocessing (a: Removing Variables Intuitively)
KS_df.drop(['project_id','name', 'currency', 'deadline', 'state_changed_at', 
            'created_at', 'launched_at', 'launch_to_state_change_days'], axis=1, inplace=True)

##############################(b:Checking collinearity & Dropping the high collinearity)
co1=pd.DataFrame(KS_df.corr())

KS_df.drop(['pledged', 'name_len', 'blurb_len', 'state_changed_at_month', 
            'state_changed_at_day',  'state_changed_at_hr', 'state_changed_at_yr', 
            'created_at_yr', 'launched_at_yr'], axis=1, inplace=True)

##############################(c:Dropping all the N/A values & Double checking if there are any left)
KS_df.dropna(inplace=True)
KS_df.isnull().sum()

###########Develop a regression model (task 1: MSE)
###########Step1: Changing to dummy variables 
KS_df['country'] = np.where(KS_df.country == 'US', 1, 0)
KS_df['deadline_weekday'] = np.where(((KS_df['deadline_weekday'] == 'Sunday')|(KS_df['deadline_weekday'] == 'Saturday')),0,1)
KS_df['state_changed_at_weekday'] = np.where(((KS_df['state_changed_at_weekday'] == 'Sunday')|(KS_df['state_changed_at_weekday'] == 'Saturday')),0,1)
KS_df['created_at_weekday'] = np.where(((KS_df['created_at_weekday'] == 'Sunday')|(KS_df['created_at_weekday'] == 'Saturday')),0,1)
KS_df['launched_at_weekday'] = np.where(((KS_df['launched_at_weekday'] == 'Sunday')|(KS_df['launched_at_weekday'] == 'Saturday')),0,1)
state_dummies = pd.get_dummies(KS_df.state)
state_dummies = pd.get_dummies(KS_df.state, prefix='state').iloc[:, 1:]
disable_dummies = pd.get_dummies(KS_df.disable_communication)
disable_dummies = pd.get_dummies(KS_df.disable_communication, prefix='disable_communication').iloc[:, 1:]
staffpick_dummies = pd.get_dummies(KS_df.staff_pick)
staffpick_dummies = pd.get_dummies(KS_df.staff_pick, prefix='staff_pick').iloc[:, 1:]
category_dummies = pd.get_dummies(KS_df.category)
category_dummies = pd.get_dummies(KS_df.category, prefix='category').iloc[:, 1:]
spotlight_dummies = pd.get_dummies(KS_df.spotlight)
spotlight_dummies = pd.get_dummies(KS_df.spotlight, prefix='spotlight').iloc[:, 1:]
merged = pd.concat([KS_df,state_dummies, disable_dummies, staffpick_dummies, category_dummies, spotlight_dummies], axis=1)
final_KS_df = merged.drop(['state', 'disable_communication', 'staff_pick', 'category', 'spotlight'], axis=1)

###########Step2: Checking collinearity again and Dropping the high collinearity
co2=pd.DataFrame(final_KS_df.corr())
final_KS_df.drop(['spotlight_True'], axis=1, inplace=True)

###########Step3: Applying Feature Selection (LASSO)
X = final_KS_df.drop(['usd_pledged'], axis=1)
y = final_KS_df["usd_pledged"]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

from sklearn.linear_model import Lasso
ls = Lasso(alpha=0.01, positive=True)
model = ls.fit(X_std,y)

coeff = pd.DataFrame(list(zip(X.columns,model.coef_)), columns = ['predictor','coefficient'])
coeff = coeff.sort_values('coefficient', ascending=False)

###########Top 10 predictors are selected (The non-zero coefficients then selected to be a part of the model):
###########backers_count, category_Hardware, staff_pick_True, state_successful, category_Wearables
###########category_Web, category_Sound, category_Gadgets, category_Software, category_Flight 

###########Step4: Printing Summary Statistics of Top 10 predictors 
X_regression = pd.DataFrame(final_KS_df.iloc[:,[2, 35, 27, 24, 48, 49, 45, 34, 44 ,33]])
SummaryStatistics1 = X_regression.describe()

co3=pd.DataFrame(X_regression.corr())###Checking collinearity again

###########Step5: Using Linear Regression Model for Estimation Tasks 
from sklearn.linear_model import LinearRegression

X = final_KS_df.iloc[:,[2, 35, 27, 24, 48, 49, 45, 34, 44 ,33]].values                         
y = final_KS_df['usd_pledged']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 5)

lm = LinearRegression()
model1= lm.fit(X_train,y_train)

y_test_pred = lm.predict(X_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_test_pred)
print(mse)
###########Results: The MSE of model1 (Linear Regression) is 5,686,083,271.703872

###########Step6: Using Random Forest for Estimation Tasks
from sklearn.ensemble import RandomForestRegressor

X = final_KS_df.iloc[:,[2, 35, 27, 24, 48, 49, 45, 34, 44,33]].values                       
y = final_KS_df['usd_pledged']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 5)

rf = RandomForestRegressor(random_state=0, n_estimators=100)
model2 = rf.fit(X_train,y_train)

y_test_pred = rf.predict(X_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_test_pred)
print(mse)
###########Results: The MSE of model2 (Random Forest) is 9,067,032,717.549711

###########Step7: Using SVM for Estimation Tasks
from sklearn.svm import SVR

X = final_KS_df.iloc[:,[2, 35, 27, 24, 48, 49, 45, 34,44,33]].values                       
y = final_KS_df['usd_pledged']

from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
X_std = standardizer.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.33, random_state = 5)

svm = SVR(kernel='linear', epsilon=0.1)
model3 = svm.fit(X_train,y_train)

y_test_pred = svm.predict(X_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_test_pred)
print(mse)
###########Results: The MSE of model3 (SVM) is 14,146,867,779.264277

###########Step8: Using ANN for Estimation Tasks
from sklearn.neural_network import MLPRegressor

X = final_KS_df.iloc[:,[2, 35, 27, 24, 48, 49, 45, 34,44,33]].values                       
y = final_KS_df['usd_pledged']

from sklearn.preprocessing import MinMaxScaler
standardizer = MinMaxScaler()
X_std = standardizer.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.33, random_state = 5)

ann = MLPRegressor(hidden_layer_sizes=(3),max_iter=1000)
model4 = ann.fit(X_train,y_train)

y_test_pred = ann.predict(X_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_test_pred)
print(mse)
###########Results: The MSE of model4 (ANN) is 14,283,273,011.787289

###########Step9: Using KNN for Estimation Tasks
from sklearn.neighbors import KNeighborsRegressor

X = final_KS_df.iloc[:,[2, 35, 27, 24, 48, 49, 45, 34,44,33]].values                       
y = final_KS_df['usd_pledged']

from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
X_std = standardizer.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.33, random_state = 5)

knn = KNeighborsRegressor(n_neighbors=2)
model5 = knn.fit(X_train,y_train)

y_test_pred = knn.predict(X_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_test_pred)
print(mse)
###########Results: The MSE of model5 (KNN) is 8,726,420,605.686438

###################################Final Result (Task 1)#######################################
#########The MSE of linear Regression Model is the smallest compare to RF, SVM, ANN, & KNN)#########
#########The MSE of Linear Regression Model is 5,686,083,271.703872############################





###############################################################################################
##################################Grader Validation(1)#########################################
###############################################################################################
KS_df_validation=pd.read_excel("Kickstarter-Validation.xlsx")####Import dataset 

###########Data Preprocessing (a: Removing Columns Intuitively)
KS_df_validation.drop(['project_id','name', 'currency', 'deadline', 'state_changed_at', 
                       'created_at', 'launched_at', 'launch_to_state_change_days'], axis=1, inplace=True)

##############################(b:Dropping the high collinearity)
KS_df_validation.drop(['pledged', 'name_len', 'blurb_len', 'state_changed_at_month', 
            'state_changed_at_day',  'state_changed_at_hr', 'state_changed_at_yr', 
            'created_at_yr', 'launched_at_yr'], axis=1, inplace=True)

##############################(c:Dropping all the N/A values & Double checking if there are any left)
KS_df_validation.dropna(inplace=True)
KS_df_validation.isnull().sum()

###########Develop a regression model (task 1: MSE)
###########Step1: Changing to dummy variables 
KS_df_validation['country'] = np.where(KS_df_validation.country == 'US', 1, 0)
KS_df_validation['deadline_weekday'] = np.where(((KS_df_validation['deadline_weekday'] == 'Sunday')|(KS_df_validation['deadline_weekday'] == 'Saturday')),0,1)
KS_df_validation['state_changed_at_weekday'] = np.where(((KS_df_validation['state_changed_at_weekday'] == 'Sunday')|(KS_df_validation['state_changed_at_weekday'] == 'Saturday')),0,1)
KS_df_validation['created_at_weekday'] = np.where(((KS_df_validation['created_at_weekday'] == 'Sunday')|(KS_df_validation['created_at_weekday'] == 'Saturday')),0,1)
KS_df_validation['launched_at_weekday'] = np.where(((KS_df_validation['launched_at_weekday'] == 'Sunday')|(KS_df_validation['launched_at_weekday'] == 'Saturday')),0,1)
state_dummies = pd.get_dummies(KS_df_validation.state)
state_dummies = pd.get_dummies(KS_df_validation.state, prefix='state').iloc[:, 1:]
disable_dummies = pd.get_dummies(KS_df_validation.disable_communication)
disable_dummies = pd.get_dummies(KS_df_validation.disable_communication, prefix='disable_communication').iloc[:, 1:]
staffpick_dummies = pd.get_dummies(KS_df_validation.staff_pick)
staffpick_dummies = pd.get_dummies(KS_df_validation.staff_pick, prefix='staff_pick').iloc[:, 1:]
category_dummies = pd.get_dummies(KS_df_validation.category)
category_dummies = pd.get_dummies(KS_df_validation.category, prefix='category').iloc[:, 1:]
spotlight_dummies = pd.get_dummies(KS_df_validation.spotlight)
spotlight_dummies = pd.get_dummies(KS_df_validation.spotlight, prefix='spotlight').iloc[:, 1:]
merged_validation = pd.concat([KS_df_validation,state_dummies, disable_dummies, staffpick_dummies, category_dummies, spotlight_dummies], axis=1)
final_KS_df_validation = merged_validation.drop(['state', 'disable_communication', 'staff_pick', 'category', 'spotlight'], axis=1)

###########Step2: Using Linear Regression Model for Estimation Tasks 
X_validation = final_KS_df_validation.iloc[:,[2, 35, 27, 24, 48, 49, 45, 34, 44 ,33]].values                         
y_validation = final_KS_df_validation['usd_pledged']

y_test_pred = lm.predict(X_validation)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_validation, y_test_pred)
print(mse)
###############################################################################################
##################################Grader Validation(1)#########################################
###############################################################################################





###########Develop a classification model (task 2: Accuracy Score) 
###########Step1: Filtering "state" ("Successful" & "Failed") and Changing to dummy variables 
KS_df.state.unique()
KS_df_datalist = ['failed', 'successful']
KS_df_FS = KS_df[KS_df.state.isin(KS_df_datalist)]
KS_df_FS.state.unique()
state_dummies_2 = pd.get_dummies(KS_df_FS.state)
state_dummies_2 = pd.get_dummies(KS_df_FS.state, prefix='state').iloc[:, 1:]
merged_2 = pd.concat([KS_df,state_dummies_2, disable_dummies, staffpick_dummies, category_dummies, spotlight_dummies], axis=1)
final_2_KS_df = merged_2.drop(['state', 'disable_communication', 'staff_pick', 'category', 'spotlight'], axis=1)
final_3_KS_df = final_2_KS_df.dropna()

###########Step2: Checking collinearity again 
co4=pd.DataFrame(final_3_KS_df.corr())
final_3_KS_df.drop(['spotlight_True'], axis=1, inplace=True)

###########Step3: Applying Feature Selection (Recursive Feature Elimination)
X = final_3_KS_df.drop(['state_successful'], axis=1)
y = final_3_KS_df["state_successful"]

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
rfe = RFE(lr, 5)
model = rfe.fit(X, y)

coeff2 = pd.DataFrame(list(zip(X.columns,model.ranking_)), columns = ['predictor','coefficient'])
coeff2 = coeff2.sort_values('coefficient')

###########Top 10 predictors are selected (Based on the ranking):
###########"launched_at_hr", "launch_to_deadline_days", "backers_count", "blurb_len_clean",
###########"deadline_hr", "created_at_hr", "deadline_day","created_at_day", "create_to_launch_days"

###########Step4: Printing Summary Statistics of Top 10 predictors 
X_classification = pd.DataFrame(final_3_KS_df.iloc[:,[22, 2, 20, 6, 14, 17, 12, 16, 21, 4]])
SummaryStatistics2 = X_classification.describe()

co5=pd.DataFrame(X_classification.corr())###Checking collinearity again
############################################Results: dropping 

###########Step5: Using Logistic Regression Model for Estimation Tasks 
from sklearn.linear_model import LogisticRegression

X = final_3_KS_df.iloc[:,[22, 2, 20, 6, 14, 17, 12, 16, 21, 4]].values
y = final_3_KS_df["state_successful"]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 5)

lr = LogisticRegression()
model1 = lr.fit(X_train,y_train)

from sklearn import metrics

y_test_pred = lr.predict(X_test)
print(metrics.accuracy_score(y_test,y_test_pred))
###########Results: The Accuracy Score of model1 (Logistic Regression) is 0.8112543962485346

###########Step6: Using Random Forest for Estimation Tasks
X = final_3_KS_df.iloc[:,[22, 2, 20, 6, 14, 17, 12, 16, 21, 4]].values
y = final_3_KS_df["state_successful"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=5)

from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier()

model2 = randomforest.fit(X_train, y_train)
y_test_pred = model2.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_test_pred)
###########Results: The Accuracy Score of model2 (Random Forest) is 0.8456619057770198

###########Step7: Using ANN for Estimation Task 
X = final_3_KS_df.iloc[:,[22, 2, 20, 6, 14, 17, 12, 16, 21, 4]].values
y = final_3_KS_df["state_successful"]

from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
X_std = standardizer.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_std,y,test_size=0.33, random_state=5)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(3), max_iter=1000)
model3=mlp.fit(X_train, y_train)
y_test_pred = model3.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_test_pred)
###########Results: The Accuracy Score of model3 (ANN) is 0.8507780856960137

###########Step8: Using KNN for Estimation Task 
X = final_3_KS_df.iloc[:,[22, 2, 20, 6, 14, 17, 12, 16, 21, 4]].values
y = final_3_KS_df["state_successful"]

from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
X_std = standardizer.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.33, random_state = 5)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
for i in range (1,11):
    knn = KNeighborsClassifier(n_neighbors=i)
    model4 = knn.fit(X_train,y_train)
    y_test_pred = knn.predict(X_test)
    print(accuracy_score(y_test, y_test_pred))
###########Results: The Accuracy Score of model4 (KNN) is 0.7173310594755915 

###########Step9: Using SVM for Estimation Task 
from sklearn.svm import SVC
X = final_3_KS_df.iloc[:,[22, 2, 20, 6, 14, 17, 12, 16, 21, 4]].values
y = final_3_KS_df["state_successful"]  

svc_scaler = StandardScaler()
X_std = svc_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.33, random_state = 5)

model5 = SVC(kernel='rbf', random_state=5,C=1, gamma=1).fit(X_train,y_train)
y_test_pred = model5.predict(X_test)

accuracy_score(y_test, y_test_pred)
###########Results: The Accuracy Score of SVM is 0.7252185035173737

###################################Final Result (Task 2)#######################################
#########The Accuracy Score of ANN is the highest compare to RF & Logistic Regression)#########
#########The Accuracy Score of ANN is 0.850991259859305#######################################





###############################################################################################
##################################Grader Validation(2)#########################################
###############################################################################################
###########Develop a classification model (task 2: Accuracy Score) 
###########Step1: Filtering "state" ("Successful" & "Failed") and Changing to dummy variables 
KS_df_validation.state.unique()
KS_df_validation_datalist = ['failed', 'successful']
KS_df_validation_FS = KS_df_validation[KS_df_validation.state.isin(KS_df_validation_datalist)]
KS_df_validation_FS.state.unique()
state_dummies = pd.get_dummies(KS_df_validation_FS.state)
state_dummies = pd.get_dummies(KS_df_validation_FS.state, prefix='state').iloc[:, 1:]
merged_2_validation = pd.concat([KS_df_validation,state_dummies, disable_dummies, staffpick_dummies, category_dummies, spotlight_dummies], axis=1)
final_2_KS_df_validation = merged_2_validation.drop(['state', 'disable_communication', 'staff_pick', 'category', 'spotlight'], axis=1)
final_3_KS_df_validation = final_2_KS_df_validation.dropna()

###########Step2: Using ANN for Estimation Tasks
X_validation = final_3_KS_df_validation.iloc[:,[22, 2, 20, 6, 14, 17, 12, 16, 21,4]].values
y_validation = final_3_KS_df_validation["state_successful"]

from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
X_std_validation = standardizer.fit_transform(X_validation)

y_test_pred = model3.predict(X_std_validation)
from sklearn.metrics import accuracy_score
accuracy_score(y_validation,y_test_pred)
###############################################################################################
##################################Grader Validation(2)#########################################
###############################################################################################





###########Develop a clustering model (task 3: Business Insights) 
###########Step1: Choosing the variables based on (Recursive Feature Elimination)
###########'goal','launch_to_deadline_days','backers_count','state'

###########Step2: Printing Summary Statistics of 4 Predictors
KS_df['state']=np.where(KS_df.state=='successful',1,0)
merged_3 = pd.concat([KS_df,disable_dummies, staffpick_dummies, category_dummies, spotlight_dummies], axis=1)
final_4_KS_df = merged_3.drop(['disable_communication', 'staff_pick', 'category', 'spotlight'], axis=1)
final_5_KS_df = final_4_KS_df.dropna()
X_clustering = pd.DataFrame(final_5_KS_df[['goal','launch_to_deadline_days','backers_count','state']])
SummaryStatistics3= X_clustering.describe()

X = final_5_KS_df[['goal','launch_to_deadline_days','backers_count','state']] 

###########Step3: Trying to standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

###########Step4: Looking for the best K
from sklearn.cluster import KMeans
withinss = []
for i in range (2,8):    
    kmeans = KMeans(n_clusters=i)
    model = kmeans.fit(X_std)
    withinss.append(model.inertia_)
    
from matplotlib import pyplot
pyplot.plot([2,3,4,5,6,7],withinss)
###########Results: 4 is the best K

###########Step5: Building a clusteing model 
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
model = kmeans.fit(X_std)
labels = model.predict(X_std)

###########Step6: Finding the silhouette score
from sklearn.metrics import silhouette_samples
silhouette = silhouette_samples(X_std,labels)

df = pd.DataFrame({'label':labels,'silhouette':silhouette})
print('Average Silhouette Score for Cluster 0: ',np.average(df[df['label'] == 0].silhouette))
print('Average Silhouette Score for Cluster 1: ',np.average(df[df['label'] == 1].silhouette))
print('Average Silhouette Score for Cluster 2: ',np.average(df[df['label'] == 2].silhouette))
print('Average Silhouette Score for Cluster 3: ',np.average(df[df['label'] == 3].silhouette))

from sklearn.metrics import silhouette_score
silhouette_score(X_std,labels)

###########Step7: Looking for the insights 
clustering = pd.DataFrame(model.cluster_centers_, columns=['goal','launch_to_deadline_days','backers_count','state'])
clustering