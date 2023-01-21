import pandas as pd
import json
import numpy as np
import streamlit as st
from PIL import Image
import joblib
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
from sklearn.metrics import precision_score, recall_score, f1_score
# Custom classes 
from pandas.api.types import is_numeric_dtype
from utils import isNumerical
import os
import warnings
import json

warnings.filterwarnings('ignore')


def main():

	#data= load_data()
	data = pd.read_csv('Performance Data.csv')
	page= st.sidebar.selectbox("Choose a page", ['Homepage', 'Data Exploration', 'Data Analysis', 'Modelling', 'Prediction'])

	if page == 'Homepage':
		st.title('Employee Performance Prediction:computer:')
		st.markdown('👈Select a page in the sidebar')
		st.markdown('This application performs machine learning predictions on Employee Performance and outputs the predictions.')

		st.markdown('This application provides:')
		st.markdown('●    A machine learning prediction on Employee Performance.:computer:')
		st.markdown('●    Data Exploration of the dataset used in training and prediction.:bar_chart:')
		st.markdown('●    Custom data Visualization and Plots.:chart_with_upwards_trend:')

		if st.checkbox('Show raw Data'):
			st.dataframe(data)

	elif page =='Data Exploration':
		st.title('Explore the dataset')
		if st.checkbox('Show raw Data'):
			#Encoding all the ordinal columns and creating a dummy variable for them to see if there are any effects on Performance Rating
			enc = LabelEncoder()
			data["Education"]= enc.fit_transform(data.Education)
			data["Attried"]= enc.fit_transform(data.Attried)
			st.dataframe(data)

		st.markdown('### Analysing Column Distribution')
		all_column_names = data.columns.tolist()
		selected_column_names =st.multiselect("Select Columns To Plot", all_column_names)
		if st.button("Generate Plot"):
			st.success("Generating Customizable Bar Plot for {}".format(selected_column_names))
			cus_data= data[selected_column_names]
			st.bar_chart(cus_data)

		if st.checkbox('Show Shape'):
			st.write(data.shape)

		if st.checkbox('Show Columns'):
			all_column = data.columns.tolist()
			st.write(all_column)

		if st.checkbox("Summary"):
			st.write(data.describe())

		if st.checkbox("Selected Columns"):
			all_column = data.columns.tolist()
			selected_column = st.multiselect("Select Columns", all_column)
			new_data = data[selected_column]
			st.dataframe(new_data)

		#Looking for Missing data	
		if st.checkbox("Check Missing Data"):
			st.write(data.info())

		#if st.checkbox("Show Value Counts"):
			#st.write(data.iloc[:,[1,25]].value_counts())

		#if st.checkbox("Correlation Plot(Matplotlib)"):
		#		plt.matshow(data.corr())
		#		st.pyplot()

		#if st.checkbox("Correlation Plot(Seaborn)"):
		#		st.write(sns.heatmap(data.corr(),annot=True))
		#		st.pyplot()

		#if st.checkbox("Correlation Plot"):
		#	corr = data.corr(method='pearson')
		#	fig2, ax2 = plt.subplots()
		#	mask = np.zeros_like(corr, dtype=np.bool)
		#	mask[np.triu_indices_from(mask)] = True
		#	# Colors
		#	cmap = sns.diverging_palette(240, 10, as_cmap=True)
		#	sns.heatmap(corr, mask=mask, linewidths=.5, cmap=cmap, center=0,ax=ax2)
		#	ax2.set_title("Correlation Matrix")
		#	st.pyplot(fig2)

		if st.checkbox("Correlation Plot Analysis"):
			corrmat = data.corr()
			f, ax = plt.subplots(figsize=(24, 10))
			sns.heatmap(corrmat, vmax=.8, linewidths=.5, annot=True, center=0, square=True);
			ax.set_title("Correlation Matrix")
			st.pyplot(f)

	elif page=='Data Analysis':	
		st.markdown('### Analysis of the Highest Correlation Matrix Column with Performance Rating')
		
		if st.checkbox('Education Column Performance'):
			# A new pandas Dataframe is created to analyze Education wise performance as asked.
			edu = data.iloc[:,[3,25]].copy()
			edu_per = edu.copy()
			# Finding out the mean performance of all the departments and plotting its bar graph using seaborn.
			ep = edu_per.groupby(by='Education')['PerformanceRating'].agg(['mean', 'sum', 'count'])
			st.write(ep)
		if st.checkbox('EP Bar Chart'):
			edu = data.iloc[:,[3,25]].copy()
			edu_per = edu.copy()
			f, ax =plt.subplots(figsize=(10,4.5))
			sns.barplot(edu_per['Education'],edu_per['PerformanceRating'])

			st.pyplot(f)

		#if st.checkbox('Education analyzed seperately'):
		#	edu = data.iloc[:,[3,25]].copy()
		#	edu_per = edu.copy()
		#	# Analyze each Education separately
		#	ed= edu_per.groupby(by='Education')['PerformanceRating'].value_counts()
		#	st.write(ed)

# A new pandas Dataframe is created to analyze TIC wise performance as asked.
		if st.checkbox('Tenure in the company Column Performance'):
			tic = data.iloc[:,[4,25]].copy()
			tic_per = tic.copy()
			# Finding out the mean performance of all the Tenure used in the company and plotting its bar graph using seaborn.
			tic1 = tic_per.groupby(by='Tenureinthecompany')['PerformanceRating'].agg(['mean', 'sum', 'count'])
			st.write(tic1)
		if st.checkbox('Tenure in the company Bar Chart'):
			tic = data.iloc[:,[4,25]].copy()
			tic_per = tic.copy()
			f, ax =plt.subplots(figsize=(10,4.5))
			sns.barplot(tic_per['Tenureinthecompany'],tic_per['PerformanceRating'])
			st.pyplot(f)


		# A new pandas Dataframe is created to analyze Experience in Months wise performance as asked.
		import random
		if st.checkbox('Prevous Exp in Months Column Performance'):	
			pem = data.iloc[:,[5,25]].copy()
			pem_per = pem.copy()
			# Finding out the mean performance of all the previous experience and plotting its bar graph using seaborn.
			pem1 = pem_per.groupby(by='PrevousExpinMonths')['PerformanceRating'].agg(['mean', 'sum', 'count'])
			st.write(pem1)
		if st.checkbox('Prevous Exp in Months Scatter Plot'):
			pem = data.iloc[:,[5,25]].copy()
			pem_per = pem.copy()
			f, ax =plt.subplots(figsize=(10,4.5))
			plt.scatter(pem_per['PrevousExpinMonths'],pem_per['PerformanceRating'])
			plt.title("PrevousExpinMonths Scatterplot")
			plt.xlabel("PrevousExpinMonths")
			plt.ylabel("erformanceRating")
			st.pyplot(f)

		# A new pandas Dataframe is created to analyze Team Commitment wise performance as asked.	
		if st.checkbox('Team Commitment Column Performance'):	
			tc = data.iloc[:,[6,25]].copy()
			tc_per = tc.copy()
			# Finding out the mean performance of all the previous experience and plotting its bar graph using seaborn.
			tc1 = tc_per.groupby(by='BuildingTeamCommitment')['PerformanceRating'].agg(['mean', 'sum', 'count'])
			st.write(pem1)
		if st.checkbox('Team Commitment Bar Chart'):
			tc = data.iloc[:,[6,25]].copy()
			tc_per = tc.copy()
			f, ax =plt.subplots(figsize=(10,4.5))
			sns.barplot(tc_per['BuildingTeamCommitment'],tc_per['PerformanceRating'])
			st.pyplot(f)


		"""if st.checkbox('BarPlot for Education analyzed seperately'):
		# Creating a new dataframe to analyze each Education separately	
			edu = data.iloc[:,[3,25]].copy()
			edu_per = edu.copy()
			education = pd.get_dummies(edu_per['Education'])
			performance = pd.DataFrame(edu_per['PerformanceRating'])
			edu_rating = pd.concat([education,performance],axis=1)
		# Plotting a separate bar graph for performance of each Education using seaborn	
			f, ax= plt.subplots(figsize=(15,10))
			#f0, ax0 = plt.subplots(figsize=(2,3))
			sns.barplot(edu_rating['PerformanceRating'],edu_rating['Post Graduate'])
			f0, ax0= plt.subplots(figsize=(15,10))
			sns.barplot(edu_rating['PerformanceRating'],edu_rating['Graduate'])
			f1, ax1= plt.subplots(figsize=(15,10))
			sns.barplot(edu_rating['PerformanceRating'],edu_rating['High School'])
			f2, ax2= plt.subplots(figsize=(15,10))
			sns.barplot(edu_rating['PerformanceRating'],edu_rating['Below High School'])
			st.pyplot(f)
			st.pyplot(f0)
			st.pyplot(f1)
			st.pyplot(f2)"""

	elif page == 'Modelling': 
		st.title('Model Application')
		st.markdown('This is the application of machine learning to derive our model')
		st.markdown("#### Train Test Splitting")# Create the model parameters dictionary 
		params = {}
		# Use two column technique 
		col1, col2 = st.columns(2)
		# Design column 1 
		y_var = col1.radio("Select the variable to be predicted (y)", options=data.columns)
		# Design column 2 
		X_var = col2.multiselect("Select the variables to be used for prediction (X)", options=data.columns)
		# Check if len of x is not zero 
		if len(X_var) == 0:
			st.error("You have to put in some X variable and it cannot be left empty.")
			# Check if y not in X 
			if y_var in X_var:
				st.error("Warning! Y variable cannot be present in your X-variable.")
			
		# Option to select predition type 
		pred_type = st.radio("Select the type of process you want to run.",  options=["LR","SVM", "DT", "RF", "NB", "KNN", "XGBoost", "ANN"], help="Write about the models") #, "LSTM_CNN"],
		# Add to model parameters
		params = {'X': X_var, 'y': y_var, 'pred_type': pred_type,}
		# Divide the data into test and train set 
		st.write(f"**Variable to be predicted:** {y_var}")
		st.write(f"**Variable to be used for prediction:** {X_var}")
		X = data[X_var]
		y = data[y_var]
		# Perform data imputation 
        # st.write("THIS IS WHERE DATA IMPUTATION WILL HAPPEN")
        # Perform encoding
		X = pd.get_dummies(X)
		if not isNumerical(y):
			le = LabelEncoder()
			y = le.fit_transform(y)
			#Print all the classes 
			st.write("The classes and the class allotted to them is the following:-")
			classes = list(le.classes_)
			for i in range(len(classes)):
				st.write(f"{classes[i]} --> {i}")
				#Perform train test splits 
		st.markdown("### Train Test Splitting")
		size = st.slider("Percentage of value division", min_value=0.1, max_value=0.9, step = 0.1, value=0.8, help="This is the value which will be used to divide the data for training and testing. Default = 80%")
		X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state=42)
		st.write("Number of training samples:", X_train.shape[0])
		st.write("Number of testing samples:", X_test.shape[0])
		#Save the model params as a json file
		with open('model/model_params.json', 'w') as json_file:
			json.dump(params, json_file)
		st.markdown("### RUNNING THE MACHINE LEARNING MODELS")
		from PIL import Image
		#opening the image
		pr = Image.open('./Result/Performance Metrics.png')
		st.image(pr, caption='Performance Metrics used in this project')
		if pred_type == "LR":
			st.write("### 1. Running Logistics Regression Algorithm on Sample")		
			#Logistics regression model 
			from sklearn.linear_model import LogisticRegression
			model_logr = LogisticRegression()
			model_logr.fit(X_train,y_train)
			#Predicting the model
			y_predict_log = model_logr.predict(X_test)
			# Finding accuracy, precision, recall and confusion matrix
			st.markdown('### Confusion Matrix for Logistic Regression')
			st.write('The Confusion Matrix is used to know the performance of a Machine learning classification. It is represented in a matrix form. Confusion Matrix gives a comparison between Actual and predicted values.')
			lrcf=confusion_matrix(y_test,y_predict_log)
			lrcf_data = pd.DataFrame(lrcf,
										 index = ['Worse', 'Low', 'Good', 'Excellent', 'Outstanding'], 
										 columns = ['Worse', 'Low', 'Good', 'Excellent', 'Outstanding'])
			#Plotting the Confusion Matrix
			f4, ax=plt.subplots(figsize=(15,10))
			sns.heatmap(lrcf_data, annot=True, fmt="d")
			ax.set_title('Logistic Regression Confusion Matrix')
			ax.set_ylabel('Actual Values')
			ax.set_xlabel('Predicted Values')
			st.pyplot(f4)
			#st.text(confusion_matrix(y_test,y_predict_log))
			st.markdown('### Classification Report for Logistic Regression')			
			index = ['Worse', 'Low', 'Good', 'Excellent', 'Outstanding']
			#Plotting the Classification Report
			st.write('A Classification report is used to measure the quality of predictions from a classification algorithm. How many predictions are True and how many are False. More specifically, True Positives, False Positives, True negatives and False Negatives are used to predict the metrics of a classification report as shown below')
			st.markdown(' #### Classification Result')
			st.text(classification_report(y_test,y_predict_log,target_names=index))
			st.write('The accuracy score of the application of Logistic Regression algorithm is ', (accuracy_score(y_test,y_predict_log)))
			
			st.write('So from the above results from the Precision, Recall, F1-Score, were displayed.  The support is the number of data samples used in the report')
			#Save Model
			st.markdown('### Save Model')
			if st.button('SAVE MODEL'):
				#Exporting the trained model
				joblib.dump(model_logr,'model/LRModel.ml')

		elif pred_type == "SVM":
			st.write("### 2. Running Support Vector Machine Algorithm on Sample")
			#Support Vector Machine Model
			from sklearn.svm import SVC
			model_svc = SVC(kernel='rbf', C=100, random_state=10).fit(X_train,y_train)
			#Predicting the model
			y_predict_svm = model_svc.predict(X_test)
			#Finding accuracy, precision, recall and confusion matrix
			st.markdown('### Confusion Matrix for Support Vector Machine')
			st.write('The Confusion Matrix is used to know the performance of a Machine learning classification. It is represented in a matrix form. Confusion Matrix gives a comparison between Actual and predicted values.')
			svmcf=confusion_matrix(y_test,y_predict_svm)
			svmcf_data = pd.DataFrame(svmcf,
										 index = ['Worse', 'Low', 'Good', 'Excellent', 'Outstanding'], 
										 columns = ['Worse', 'Low', 'Good', 'Excellent', 'Outstanding'])
			#Plotting the Confusion Matrix
			f4, ax=plt.subplots(figsize=(15,10))
			sns.heatmap(svmcf_data, annot=True, fmt="d")
			ax.set_title('Support Vector Machine Confusion Matrix')
			ax.set_ylabel('Actual Values')
			ax.set_xlabel('Predicted Values')
			st.pyplot(f4)
			#st.text(confusion_matrix(y_test,y_predict_svm))
			st.markdown('### Classification Report')
			st.write('A Classification report is used to measure the quality of predictions from a classification algorithm. How many predictions are True and how many are False. More specifically, True Positives, False Positives, True negatives and False Negatives are used to predict the metrics of a classification report as shown below')
			st.markdown(' #### Classification Result')
			index = ['Worse', 'Low', 'Good', 'Excellent', 'Outstanding']
			st.text(classification_report(y_test,y_predict_svm, target_names=index))
			st.write('The accuracy score for the application of Support Vector Machine algorithm is ', (accuracy_score(y_test,y_predict_svm)))
			st.write('So from the above results the Precision, Recall, F1-Score, were displayed.  The support is the number of data samples used in the report')
			st.markdown('### SAVE MODEL')
			if st.button('SAVE MODEL'):
				#Exporting the trained model
				joblib.dump(model_svc,'model/SVCModel.ml')

		elif pred_type=="DT":
			st.write("### 3. Running Decision Tree with GridSearchCV Algorithm on Sample")
			#Decisin Tree with GridSearchCV Model
			from sklearn.tree import DecisionTreeClassifier
			classifier_dtg=DecisionTreeClassifier(random_state=42,splitter='best')
			parameters=[{'min_samples_split':[2,3,4,5],'criterion':['gini']},{'min_samples_split':[2,3,4,5],'criterion':['entropy']}]
			model_griddtree=GridSearchCV(estimator=classifier_dtg, param_grid=parameters, scoring='accuracy',cv=10)
			model_griddtree.fit(X_train,y_train)
			model_griddtree.best_params_
			#Predicting the model
			y_predict_dtree = model_griddtree.predict(X_test)
			#Finding accuracy, precision, recall and confusion matrix
			st.markdown('### Confusion Matrix for DTwithGridSearchCV')
			st.write('The Confusion Matrix is used to know the performance of a Machine learning classification. It is represented in a matrix form. Confusion Matrix gives a comparison between Actual and predicted values.')
			dtcf=confusion_matrix(y_test,y_predict_dtree)
			dtcf_data = pd.DataFrame(dtcf,
										 index = ['Worse', 'Low', 'Good', 'Excellent', 'Outstanding'], 
										 columns = ['Worse', 'Low', 'Good', 'Excellent', 'Outstanding'])
			#Plotting the Confusion Matrix
			f4, ax=plt.subplots(figsize=(15,10))
			sns.heatmap(dtcf_data, annot=True, fmt="d")
			ax.set_title('DTwithGridSearchCV Confusion Matrix')
			ax.set_ylabel('Actual Values')
			ax.set_xlabel('Predicted Values')
			st.pyplot(f4)

			#st.text(confusion_matrix(y_test,y_predict_dtree))
			st.markdown('### Classification report of DTwithGridSearchCV')
			st.write('A Classification report is used to measure the quality of predictions from a classification algorithm. How many predictions are True and how many are False. More specifically, True Positives, False Positives, True negatives and False Negatives are used to predict the metrics of a classification report as shown below')
			st.markdown('#### Classification Result')
			index = ['Worse', 'Low', 'Good', 'Excellent', 'Outstanding']
			st.text(classification_report(y_test,y_predict_dtree, target_names=index))
			st.write('The accuracy score for the application of DTwithGridSearchCV Algorithm is ', (accuracy_score(y_test,y_predict_dtree)))
			st.write('So from the above results from the Precision, Recall, F1-Score, were displayed.  The support is the number of data samples used in the report')
			st.markdown('### Save Model')
			if st.button('SAVE MODEL'):
				#Exporting the trained model
				joblib.dump(model_griddtree,'model/DTgridTreeModel.ml')

		elif pred_type=="RF":
			st.write("### 4. Running Random Forest with GridSearchCV Algorithm on Sample")
			#Random Forest with GridSearchCV Model
			from sklearn.ensemble import RandomForestClassifier
			classifier_rfg=RandomForestClassifier(random_state=33,n_estimators=23)
			parameters=[{'min_samples_split':[2,3,4,5],'criterion':['gini','entropy'],'min_samples_leaf':[1,2,3]}]
			model_gridrf=GridSearchCV(estimator=classifier_rfg, param_grid=parameters, scoring='accuracy',cv=10)
			model_gridrf.fit(X_train,y_train)
			model_gridrf.best_params_
			#Predicting the model
			y_predict_rf = model_gridrf.predict(X_test)
			#Finding accuracy, precision, recall and confusion matrix
			st.markdown('### Confusion Matrix for RFwithGridSearchCV')
			st.write('The Confusion Matrix is used to know the performance of a Machine learning classification. It is represented in a matrix form. Confusion Matrix gives a comparison between Actual and predicted values.')
			rfcf=confusion_matrix(y_test,y_predict_rf)
			rfcf_data = pd.DataFrame(rfcf,
										 index = ['Worse', 'Low', 'Good', 'Excellent', 'Outstanding'], 
										 columns = ['Worse', 'Low', 'Good', 'Excellent', 'Outstanding'])
			#Plotting the Confusion Matrix
			f4, ax=plt.subplots(figsize=(15,10))
			sns.heatmap(rfcf_data, annot=True, fmt="d")
			ax.set_title('RFwithGridSearchCV Confusion Matrix')
			ax.set_ylabel('Actual Values')
			ax.set_xlabel('Predicted Values')
			st.pyplot(f4)
			#st.text(confusion_matrix(y_test,y_predict_rf))
			st.markdown('### Classification report')
			st.write('A Classification report is used to measure the quality of predictions from a classification algorithm. How many predictions are True and how many are False. More specifically, True Positives, False Positives, True negatives and False Negatives are used to predict the metrics of a classification report as shown below')
			st.markdown('#### Classification Result')
			index = ['Worse', 'Low', 'Good', 'Excellent', 'Outstanding']
			st.text(classification_report(y_test,y_predict_rf, target_names= index))
			st.write('The accuracy score for the application of RFwithGridSearchCV algorithm is ', (accuracy_score(y_test,y_predict_rf)))
			st.write('So from the above results from the Precision, Recall, F1-Score, were displayed.  The support is the number of data samples used in the report')
			st.markdown('### Save Model')
			if st.button('SAVE MODEL'):
				#Exporting the trained model
				joblib.dump(model_gridrf,'model/RFgridTreeModel.ml')


		elif pred_type=="NB":
			st.write("### 5. Running Naive Bayes Algorithm on Sample")
			#Naive bayes Model
			from sklearn.naive_bayes import BernoulliNB
			model_nb = BernoulliNB()
			model_nb.fit(X_train,y_train)
			#Predicting the model
			y_predict_nb = model_nb.predict(X_test)
			#Finding accuracy, precision, recall and confusion matrix
			st.markdown('### Confusion Matrix for Naive Bayes')
			st.write('The Confusion Matrix is used to know the performance of a Machine learning classification. It is represented in a matrix form. Confusion Matrix gives a comparison between Actual and predicted values.')
			nbcf=confusion_matrix(y_test,y_predict_nb)
			nbcf_data = pd.DataFrame(nbcf,
										 index = ['Worse', 'Low', 'Good', 'Excellent', 'Outstanding'], 
										 columns = ['Worse', 'Low', 'Good', 'Excellent', 'Outstanding'])
			#Plotting the Confusion Matrix
			f4, ax=plt.subplots(figsize=(15,10))
			sns.heatmap(nbcf_data, annot=True, fmt="d")
			ax.set_title('Naive Bayes Confusion Matrix')
			ax.set_ylabel('Actual Values')
			ax.set_xlabel('Predicted Values')
			st.pyplot(f4)
			#st.text(confusion_matrix(y_test,y_predict_nb))
			st.markdown('### Classification Report of Naive Bayes Algorithm')
			st.write('A Classification report is used to measure the quality of predictions from a classification algorithm. How many predictions are True and how many are False. More specifically, True Positives, False Positives, True negatives and False Negatives are used to predict the metrics of a classification report as shown below')
			st.markdown('#### Classification Result')
			index = ['Worse', 'Low', 'Good', 'Excellent', 'Outstanding']
			st.text(classification_report(y_test,y_predict_nb, target_names=index))
			st.write('The accuracy score for the application of Naive Bayes Algorithm is ', (accuracy_score(y_test,y_predict_nb)))
			st.write('So from the above results from the Precision, Recall, F1-Score, were displayed.  The support is the number of data samples used in the report')
			st.markdown('### Save Model')
			if st.button('SAVE MODEL'):
				#Exporting the trained model
				joblib.dump(model_nb,'model/NBModel.ml')

		elif pred_type=="KNN":
			st.write("### 6. Running K-Nearest Neighbour Model on Sample")
			#K-Nearest Neighbour Model
			from sklearn.neighbors import KNeighborsClassifier
			model_knn = KNeighborsClassifier(n_neighbors=10,metric='euclidean') # Maximum accuracy for n=10
			model_knn.fit(X_train,y_train)
			#Predicting the model
			y_predict_knn = model_knn.predict(X_test)
			#Finding accuracy, precision, recall and confusion matrix
			st.markdown('### Confusion Matrix for K-Nearest Neighbour')
			st.write('The Confusion Matrix is used to know the performance of a Machine learning classification. It is represented in a matrix form. Confusion Matrix gives a comparison between Actual and predicted values.')
			knncf=confusion_matrix(y_test,y_predict_knn)
			knncf_data = pd.DataFrame(knncf,
										 index = ['Worse', 'Low', 'Good', 'Excellent', 'Outstanding'], 
										 columns = ['Worse', 'Low', 'Good', 'Excellent', 'Outstanding'])
			#Plotting the Confusion Matrix
			f4, ax=plt.subplots(figsize=(15,10))
			sns.heatmap(knncf_data, annot=True, fmt="d")
			ax.set_title('K-Nearest Neighbour Confusion Matrix')
			ax.set_ylabel('Actual Values')
			ax.set_xlabel('Predicted Values')
			st.pyplot(f4)

			#st.text(confusion_matrix(y_test,y_predict_knn))
			st.markdown('### Classification Report of K-Nearest Neighbour Algorithm')
			st.write('A Classification report is used to measure the quality of predictions from a classification algorithm. How many predictions are True and how many are False. More specifically, True Positives, False Positives, True negatives and False Negatives are used to predict the metrics of a classification report as shown below')
			st.markdown('### Classification Result')
			index = ['Worse', 'Low', 'Good', 'Excellent', 'Outstanding']
			st.text(classification_report(y_test,y_predict_knn, target_names=index))
			st.write('The accuracy score for the application of K-Nearest Neighbour Algorithm is ', (accuracy_score(y_test,y_predict_knn)))
			st.write('So from the above results from the Precision, Recall, F1-Score, were displayed.  The support is the number of data samples used in the report')
			
			st.markdown('### Save Model')
			if st.button('SAVE MODEL'):
				#Exporting the trained model
				joblib.dump(model_knn,'model/KNNModel.ml')

		elif pred_type=="XGBoost":
			st.write("### 7. Running XGBoost Classifer Algorithm on Sample")
			#XGBOOST Model
			from xgboost import XGBClassifier
			model_xgb = XGBClassifier()
			model_xgb.fit(X_train,y_train)
			#Predicting the model
			y_predict_xgb = model_xgb.predict(X_test)
			#Finding accuracy, precision, recall and confusion matrix
			st.write('The Confusion Matrix is used to know the performance of a Machine learning classification. It is represented in a matrix form. Confusion Matrix gives a comparison between Actual and predicted values.')
			st.markdown('### Confusion Matrix for XGBoost')
			xgbcf=confusion_matrix(y_test,y_predict_xgb)
			xgbcf_data = pd.DataFrame(xgbcf,
										 index = ['Worse', 'Low', 'Good', 'Excellent', 'Outstanding'], 
										 columns = ['Worse', 'Low', 'Good', 'Excellent', 'Outstanding'])
			#Plotting the Confusion Matrix
			f4, ax=plt.subplots(figsize=(15,10))
			sns.heatmap(xgbcf_data, annot=True, fmt="d")
			ax.set_title('XGBoost Confusion Matrix')
			ax.set_ylabel('Actual Values')
			ax.set_xlabel('Predicted Values')
			st.pyplot(f4)

			#st.text(confusion_matrix(y_test,y_predict_xgb))
			st.markdown('### Classification Report of the application of XGBoost')
			st.write('A Classification report is used to measure the quality of predictions from a classification algorithm. How many predictions are True and how many are False. More specifically, True Positives, False Positives, True negatives and False Negatives are used to predict the metrics of a classification report as shown below')			
			st.markdown('#### Classification Result')
			index = ['Worse', 'Low', 'Good', 'Excellent', 'Outstanding']
			st.text(classification_report(y_test,y_predict_xgb, target_names=index))
			st.write('The accuracy score for the application of XGBoost Algorthm is ', (accuracy_score(y_test,y_predict_xgb)))
			st.write('So from the above results from the Precision, Recall, F1-Score, were displayed.  The support is the number of data samples used in the report')
			st.markdown('### Save Model')
			if st.button('SAVE MODEL'):
				#Exporting the trained model

				#model_xgb.save_model('model/XGBModel.json')
				joblib.dump(model_xgb,'model/XGBModel.ml')

				#model_xgb.save_model('model/XGBModel.json')
				#joblib.dump(model_xgb,'model/XGBModel.ml')


		elif pred_type=="ANN":
			st.write("### 7. Running Artificial Neural Network Algorithm on Sample")
			#ANN Model
			from sklearn.neural_network import MLPClassifier
			model_mlp = MLPClassifier(hidden_layer_sizes=(100,100,100),batch_size=10,learning_rate_init=0.01,max_iter=2000,random_state=10)
			model_mlp.fit(X_train,y_train)
			#Predicting the model
			y_predict_mlp = model_mlp.predict(X_test)
			#Finding accuracy, precision, recall and confusion matrix
			st.markdown('### Confusion Matrix for Artificial Neural Network')
			st.write('The Confusion Matrix is used to know the performance of a Machine learning classification. It is represented in a matrix form. Confusion Matrix gives a comparison between Actual and predicted values.')
			mlpcf=confusion_matrix(y_test,y_predict_mlp)
			mlpcf_data = pd.DataFrame(mlpcf,
										 index = ['Worse', 'Low', 'Good', 'Excellent', 'Outstanding'], 
										 columns = ['Worse', 'Low', 'Good', 'Excellent', 'Outstanding'])
			#Plotting the Confusion Matrix
			f4, ax=plt.subplots(figsize=(15,10))
			sns.heatmap(mlpcf_data, annot=True, fmt="d")
			ax.set_title('Artificial Neural Network Confusion Matrix')
			ax.set_ylabel('Actual Values')
			ax.set_xlabel('Predicted Values')
			st.pyplot(f4)
			#st.text(confusion_matrix(y_test,y_predict_mlp))
			st.markdown('### Classification Report for the application of Artificial Neural Network')
			st.write('A Classification report is used to measure the quality of predictions from a classification algorithm. How many predictions are True and how many are False. More specifically, True Positives, False Positives, True negatives and False Negatives are used to predict the metrics of a classification report as shown below')
			st.markdown('### Classification Result')
			index = ['Worse', 'Low', 'Good', 'Excellent', 'Outstanding']
			st.text(classification_report(y_test,y_predict_mlp, target_names=index))
			st.write('The accuracy score for the application of Artificial Neural Network Algorithm is ', (accuracy_score(y_test,y_predict_mlp)))
			st.write('So from the above results from the Precision, Recall, F1-Score, were displayed.  The support is the number of data samples used in the report')
			st.markdown('### Save Model')
			if st.button('SAVE MODEL'):
				#Exporting the trained model
				joblib.dump(model_mlp,'model/ANNModel.ml')

		elif pred_type=="LSTM_CNN":
			st.write("### 8. Running LSTM_CNN Model on Sample")
			#LSTM_CNN Model
			from keras.models import Sequential
			import matplotlib.patches as mpatches
			from keras.layers import Dense
			from keras.layers import Dropout
			from keras.layers import LSTM
			from sklearn.preprocessing import MinMaxScaler
			from sklearn.metrics import mean_squared_error
			from keras.layers import Dense,RepeatVector
			from keras.layers import Flatten
			from keras.layers import TimeDistributed
			from keras.layers.convolutional import Conv1D 
			from keras.layers.convolutional import MaxPooling1D

			X_train = np.expand_dims(np.random.normal(size=(9146, 8)),axis=-1)
			y_train = np.random.choice([0,1], size=(9146,8))
			n_timesteps, n_features, n_outputs =X_train.shape[0], X_train.shape[1], y_train.shape[1]
			model_lstm_cnn = Sequential()
			model_lstm_cnn.add(Conv1D(filters=64, kernel_size=1, activation='relu',input_shape=(n_features,1)))
			model_lstm_cnn.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
			model_lstm_cnn.add(Dropout(0.5))
			model_lstm_cnn.add(MaxPooling1D(pool_size=2))
			model_lstm_cnn.add(Flatten())
			model_lstm_cnn.add(Dense(100, activation='relu'))
			model_lstm_cnn.add(Dense(n_outputs, activation='softmax'))
			model_lstm_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
			# fit network
			a = model_lstm_cnn.fit(X_train, y_train, epochs=15, batch_size=100, verbose=1)
			st.write(a)


			# Predicting the model
			X_test = np.expand_dims(np.random.normal(size=(3920, 8)),axis=-1)
			Predictions = model_lstm_cnn.predict(X_test,batch_size =100)
			# Finding accuracy, precision, recall and confusion matrix
			from sklearn.metrics import label_ranking_average_precision_score
			y_test = np.random.choice([0,1], size=(3920,8))
			st.write('The accuracy of LSTM_CNN is ', (label_ranking_average_precision_score(y_test,Predictions)))

			if st.button('SAVE MODEL'):
				#Exporting the trained model
				joblib.dump(model_,'LSTM_CNN_Model.ml')
			
	else:
		st.title('Prediction')
		st.markdown('Input values in the form below for prediction, Dont mind the long input, just to ensure the right prediction')
		#loading in the model to predict on the data

		#with open("model/XGBModel.json", 'rb') as model:
			#Load its content and make a new dictionary
			#classifier = json.dumps(model)
		#model_open = open('model/XGBModel.json', 'rb')
		#from xgboost import XGBClassifier
		#model_xgb = XGBClassifier()
		#classifier= json.loads(model_open)

		from sklearn.externals import joblib
		model_open = open('XGB.ml', 'rb')
		classifier = joblib.load(model_open)

		if st.button('Show data'):
			st.dataframe(data)

			# defining the function which will make the prediction using 
			# the data which the user inputs

		def prediction(Gender,Age,Education,Tenureinthecompany,PrevousExpinMonths,BuildingTeamCommitment,StrategicThinking,LeadsDecisionMakingandDeliversResults,AnalyticalThinking,CustomerRelations,ServiceQualityandPlanning,SolutionSelling,InMarketExecution,SalesPlanningandForecasting,Negotiation ,ActionableInsights,SolvingProblems,Engage,AppliedThinking,Change,Drive,AverageCompScore,Competen2y,Attried):
			prediction = classifier.predict([[Gender,Age,Education,Tenureinthecompany,PrevousExpinMonths,BuildingTeamCommitment,StrategicThinking,LeadsDecisionMakingandDeliversResults,AnalyticalThinking,CustomerRelations,ServiceQualityandPlanning,SolutionSelling,InMarketExecution,SalesPlanningandForecasting,Negotiation ,ActionableInsights,SolvingProblems,Engage,AppliedThinking,Change,Drive,AverageCompScore,Competen2y,Attried]])
			print(prediction)
			return prediction

		employeeid = st.text_input("Employee ID", "")
		Gender = st.selectbox('Gender', [1, 2])
		Age = st.number_input("Age", min_value=18, max_value=70, value=18, step=1)
		Education = st.selectbox('Level of Education', [1, 2, 3, 4])#'Below High Shool', 'High School','Graduate', 'Post Graduate'
		Tenureinthecompany = st.slider("Tenure in the Company", min_value=1, max_value=40, value=1, step=1, format='%f')
		PrevousExpinMonths = st.number_input("Previous Experience in Months")
		BuildingTeamCommitment = st.slider("Team Commitment", min_value=1.0, max_value=5.0, value=1.0, step=0.1, format='%f')
		StrategicThinking = st.slider("Strategic Thinking", min_value=1.0, max_value=5.0, value=1.0, step=0.1, format='%f')
		LeadsDecisionMakingandDeliversResults = st.slider("Decision Making and Result Delivery", min_value=1.0, max_value=5.0, value=1.0, step=0.1, format='%f')
		AnalyticalThinking = st.slider("Analytical Thinking", min_value=1.0, max_value=5.0, value=1.0, step=0.1, format='%f')
		CustomerRelations = st.slider("Customer Relations", min_value=1.0, max_value=5.0, value=1.0, step=0.1, format='%f')
		ServiceQualityandPlanning = st.slider("Quality of Service and Plannig", min_value=1.0, max_value=5.0, value=1.0, step=0.1, format='%f')
		SolutionSelling = st.slider("Solution Driving", min_value=1.0, max_value=5.0, value=1.0, step=0.1, format='%f')
		InMarketExecution = st.slider("InMarket Execution", min_value=1.0, max_value=5.0, value=1.0, step=0.1, format='%f')
		SalesPlanningandForecasting = st.slider("Sales Planning and Forecasting", min_value=1.0, max_value=5.0, value=1.0, step=0.1, format='%f')
		Negotiation = st.slider("Negotiation", min_value=1.0, max_value=5.0, value=1.0, step=0.1, format='%f')
		ActionableInsights = st.slider("Actionable Insights", min_value=1.0, max_value=5.0, value=1.0, step=0.1, format='%f')
		SolvingProblems = st.slider("Problem Solving", min_value=1.0, max_value=5.0, value=1.0, step=0.1, format='%f')
		Engage =st.slider("Engagement", min_value=1.0, max_value=5.0, value=1.0, step=0.1, format='%f') 
		AppliedThinking = st.slider("Applied Thinking", min_value=1.0, max_value=5.0, value=1.0, step=0.1, format='%f')
		Change= st.slider("Dynamism", min_value=1.0, max_value=5.0, value=1.0, step=0.1, format='%f')
		Drive =st.slider("Goal Driven", min_value=1.0, max_value=5.0, value=1.0, step=0.1, format='%f')
		AverageCompScore = st.slider("Average Comp Score", min_value=1.0, max_value=5.0, value=1.0, step=0.1, format='%f')
		Competen2y = st.slider("Competency", min_value=1.0, max_value=5.0, value=1.0, step=0.1, format='%f')
		Attried = st.selectbox('Attried', [1, 2])#'Yes', 'No'   
		result =""
      
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
		if st.button("Predict"):
			result = prediction(Gender,Age,Education,Tenureinthecompany,PrevousExpinMonths,BuildingTeamCommitment,StrategicThinking,LeadsDecisionMakingandDeliversResults,AnalyticalThinking,CustomerRelations,ServiceQualityandPlanning,SolutionSelling,InMarketExecution,SalesPlanningandForecasting,Negotiation ,ActionableInsights,SolvingProblems,Engage,AppliedThinking,Change,Drive,AverageCompScore,Competen2y,Attried)
			if result[0]==1:
				result = 'The performance is Worse'
			elif result[0]==2:
				result = 'The performance is Low'
			elif result[0]==3:
				result = 'The performance is Good'
			elif result[0]==4:
				result = 'The performance is Excellent'
			else:
				result = 'The performance is Outstanding'
		st.success("{}".format(result))

if __name__ == '__main__':
    main()
