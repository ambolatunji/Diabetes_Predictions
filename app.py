import pandas as pd
import json
import numpy as np
from sklearn.ensemble import VotingClassifier
import streamlit as st
from PIL import Image
import joblib
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
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
from imblearn.over_sampling import SMOTE
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')


# Configure the app
st.set_page_config(
    page_title = 'Diabetes Prediction',
    page_icon = 'random',
    layout = 'wide',
    initial_sidebar_state = 'auto'
)


def main():
		# Add a file uploader to the home page
	uploaded_file = st.file_uploader('Upload your CSV file', type=['csv', 'xlxs'])
	if uploaded_file is not None:
		# Read the uploaded file as a DataFrame
		import pandas as pd
		data = pd.read_csv(uploaded_file)	
	#data = pd.read_csv('Performance Data.csv')
	page= st.sidebar.selectbox("Choose a page", ['Homepage', 'Modelling', 'Prediction']) #'Data Exploration', 'Data Analysis' ])

	if page == 'Homepage':
		st.title('Diabetes Prediction system:computer:')
		st.markdown('👈Select a page in the sidebar')
		st.markdown('This application performs machine learning predictions on Diabetes and outputs the predictions.')
		if st.checkbox('**Show raw Data**'):
			st.dataframe(data)
			# Add brief description of your web app
		st.markdown(
    		"""<p style="font-size:20px;">
            <p>Diabetes is a condition that occurs when your blood sugar is too high. It happens when your body doesn't produce enough insulin or can't use it properly. This causes too much blood sugar to stay in your bloodstream, which can lead to serious health issues over time.</p>
            <p>Some symptoms of diabetes include:</p>
			<ul>
    		<li>Frequent urination, especially at night</li>
    <li>Excessive thirst</li>
    <li>Feeling tired</li>
    <li>Unintentional weight loss</li>
    <li>Genital itching or thrush</li>
    <li>Slow-healing cuts and wounds</li>
    <li>Blurry vision</li>
    <li>Increased hunger</li>
    <li>Numbness or tingling in the hands or feet</li>
    <li>Dry skin</li>
</ul>

<p>Factors that can contribute to developing type 2 diabetes include:</p>
<ol>
    <li>Being overweight</li>
    <li>Not getting enough exercise</li>
    <li>Genetics</li>
</ol>

<p>Early diagnosis is important to prevent the worst effects of type 2 diabetes. A healthcare provider can detect diabetes early through regular check-ups and blood tests. Depending on the type of diabetes, people may need to take medications and insulin to manage their condition and improve glucose absorption.</p>

<p>Some tips for healthy eating with diabetes include:</p>
<ul>
    <li>Choosing healthier carbohydrates</li>
    <li>Eating less salt</li>
    <li>Eating less red and processed meat</li>
    <li>Eating more fruit and vegetables</li>
    <li>Choosing healthier fats</li>
    <li>Cutting down on added sugar</li>
    <li>Being smart with snacks</li>
    <li>Drinking alcohol sensibly</li>
</ul>
    """, unsafe_allow_html=True)
			

	elif page == 'Modelling': 
		import pickle 
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
			st.error("You have to put in some X variable and it cannot be left em'pty.")
			# Check if y not in X 
			if y_var in X_var:
				st.error("Warning! Y variable cannot be present in your X-variable.")
		# Option to select data preprocessing steps
		from sklearn.impute import SimpleImputer
		st.subheader("Data Preprocessing")
		missing_checkbox = st.checkbox("Handle Missing Values")
		duplicates_checkbox = st.checkbox("Handle Duplicates")
		class_distri_checkbox = st.checkbox("Check the class of the outcome distribution")
		smote_checkbox = st.checkbox("Apply SMOTE")
		corr_checkbox = st.checkbox("Check the relationship within the data")
		    # Initialize variables to store preprocessing information
		
		duplicates_removed = 0
		missing_values_replaced = 0
		class_distri = 0
		highest_smote_applied = 0
		
		# Handle missing values
		from sklearn.impute import SimpleImputer
		if missing_checkbox:
			for col in data.columns:
				if data[col].isnull( ).any():
					if data[col].dtype == 'object':
						imputer = SimpleImputer(strategy='most_frequent')
					else:
						imputer = SimpleImputer(strategy='mean')
					col_data = data[[col]]  # Select column as DataFrame
					col_data[col] = imputer.fit_transform(col_data)  # Apply imputation
					missing_values_replaced += col_data.isnull().sum()  # Count missing values
					data[col] = col_data[col].values  # Update original data with imputed values

			
		# Check for duplicates
		if duplicates_checkbox:
			duplicates = data.duplicated()
			if duplicates.any():
				duplicates_removed = sum(duplicates)

				data = data[~duplicates]
				#y_var = y_var[~duplicates]
		#Check class distribution
		if class_distri_checkbox:
			st.subheader("Class Distribution")
			# Plot class distribution using the last column of the dataset
			sns.countplot(x=data.iloc[:, -1])  # Assuming the last column is the target variable
			st.pyplot()
			# Optionally, show the value counts in a more direct textual form
			st.write("Value counts:")
			st.write(data.iloc[:, -1].value_counts())

	
		
		
		# Apply SMOTE
				# Apply SMOTE
		if smote_checkbox:
			from imblearn.over_sampling import SMOTE
			smote = SMOTE()
			X_smote, y_smote = smote.fit_resample(data[X_var], data[y_var])
			highest_smote_applied = max(y_smote.value_counts())
			data[X_var] = X_smote
			data[y_var] = y_smote

		# Calculate and display correlation matrix
		if corr_checkbox:
			# Calculate correlation with the last column of the data
			correlation_with_target = data.corr()[data.columns[-1]].sort_values(ascending=False)
			# Display correlation heatmap
			st.subheader("Correlation Heatmap")
			plt.figure(figsize=(10, 8))
			sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
			st.pyplot()
			# Get top five correlations with the last column
			sns.set(font_scale=0.8)
			sns.set_style("dark")
			sns.set_palette("PuBuGn_d")   
			top_five_corr_with_target = correlation_with_target.drop(data.columns[-1]).head(5)
			sns.heatmap(top_five_corr_with_target.to_frame(), cmap="coolwarm", annot=True, fmt='.2f')
			st.subheader("Top Five Correlations with Target Column")
			#st.write(top_five_corr_with_target)
			st.pyplot()

		# Option to select predition type 
		pred_type = st.radio("Select the type of process you want to run.",  options=["LR","SVM", "DT", "RF", "NB", "KNN", "XGBoost", "ANN", "Ensemble"], help="Write about the models") #, "LSTM_CNN"],
		# Add to model parameters
		params = {'X': X_var, 'y': y_var, 'pred_type': pred_type,}
		# Divide the data into test and train set 
		st.write(f"**Variable to be predicted:** {y_var}")
		st.write(f"**Variable to be used for prediction:** {X_var}")
		st.write("Data preprocessing steps applied:")
		if missing_checkbox:
			st.write(f"- Missing values replaced: {missing_values_replaced}")
		if duplicates_checkbox:
			st.write(f"- Duplicates removed: {duplicates_removed}")
		if smote_checkbox:
			st.write(f"- SMOTE applied (highest amount): {highest_smote_applied}")
		X = data[X_var]
		y = data[y_var]

		# Perform data imputation 
        # st.write("THIS IS WHERE DATA IMPUTATION WILL HAPPEN")
        # Perform encoding
		from sklearn.model_selection import train_test_split
		from  sklearn.preprocessing import LabelEncoder
		X = pd.get_dummies(X)
		if not isinstance(y[0], (int, float)):
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
		from sklearn.ensemble import RandomForestClassifier
		# Implementing cross-validation using cross_val_score
  
		model_cv = RandomForestClassifier(random_state=42)
		cv_scores = cross_val_score(model_cv, X_train, y_train, cv=5)
		st.write(f"Cross-Validation Scores: {cv_scores}")
		st.write(f"Average CV Score: {cv_scores.mean()}")

		from imblearn.over_sampling import SMOTE
		# Handling potentially imbalanced data with SMOTE
  
		smote = SMOTE()
		X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
		import pandas as pd  # Ensure pandas is imported

		st.write("Resampled dataset shape:")
		st.write(pd.Series(y_train_smote).value_counts())

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
										 index = ['Actual Low Diabetes', 'Actual High Diabetes'], 
										 columns = ['Predicted Low Diabetes', 'Predicted High Diabetes'])
			#Plotting the Confusion Matrix
			f4, ax=plt.subplots(figsize=(15,10))
			sns.heatmap(lrcf_data, annot=True, fmt="d", ax=ax)
			ax.set_title('Logistic Regression Confusion Matrix')
			ax.set_ylabel('Actual Values')
			ax.set_xlabel('Predicted Values')
			st.pyplot(f4)
			#st.text(confusion_matrix(y_test,y_predict_log))
			st.markdown('### Classification Report for Logistic Regression')			
			index = ['Actual Low Diabetes', 'Actual High Diabetes']
			#Plotting the Classification Report
			st.write('A Classification report is used to measure the quality of predictions from a classification algorithm. How many predictions are True and how many are False. More specifically, True Positives, False Positives, True negatives and False Negatives are used to predict the metrics of a classification report as shown below')
			st.markdown(' #### Classification Result')
			st.text(classification_report(y_test,y_predict_log,target_names=index))
   
			st.write('The accuracy score of the application of Logistic Regression algorithm is ', (accuracy_score(y_test,y_predict_log)))
			#def class_report(y_test,y_predict_log):
			#	report= classification_report(y_test,y_predict_log,target_names=index, output_dict=True)
			#	data = pd.Dataframe(report).transpose()
			#	return data
			
			st.write('So from the above results from the Precision, Recall, F1-Score, were displayed.  The support is the number of data samples used in the report')
			#Save Model
			st.markdown('### Save Model')
			if st.button('SAVE MODEL'):
				#Exporting the trained model
				#joblib.dump(model_logr,'model/LRModel.ml')
				  with open('model/model_logr.pkl', 'wb') as file: 
					  pickle.dump(model_logr, file)

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
										 index = ['Actual Low Diabetes', 'Actual High Diabetes'], 
										 columns = ['Predicted Low Diabetes', 'Predicted High Diabetes'])
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
			index = ['Actual Low Diabetes', 'Actual High Diabetes']
			st.text(classification_report(y_test,y_predict_svm, target_names=index))
			st.write('The accuracy score for the application of Support Vector Machine algorithm is ', (accuracy_score(y_test,y_predict_svm)))
			st.write('So from the above results the Precision, Recall, F1-Score, were displayed.  The support is the number of data samples used in the report')
			st.markdown('### SAVE MODEL')
			if st.button('SAVE MODEL'):
				#Exporting the trained model
				#joblib.dump(model_svc,'model/SVCModel.ml')
				with open('model/model_svc.pkl','wb') as f:
					pickle.dump(model_svc, f)

		elif pred_type=="DT":
			st.write("### 3. Running Decision Tree with GridSearchCV Algorithm on Sample")
			#Decisin Tree with GridSearchCV Model
			from sklearn.tree import DecisionTreeClassifier
			classifier_dtg=DecisionTreeClassifier(random_state=42,splitter='best')
			parameters=[{'min_samples_split':[2,3],'criterion':['gini']},{'min_samples_split':[2,3],'criterion':['entropy']}]
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
										 index = ['Actual Low Diabetes', 'Actual High Diabetes'], 
										 columns = ['Predicted Low Diabetes', 'Predicted High Diabetes'])
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
			index = ['Actual Low Diabetes', 'Actual High Diabetes']
			st.text(classification_report(y_test,y_predict_dtree, target_names=index))
			st.write('The accuracy score for the application of DTwithGridSearchCV Algorithm is ', (accuracy_score(y_test,y_predict_dtree)))
			st.write('So from the above results from the Precision, Recall, F1-Score, were displayed.  The support is the number of data samples used in the report')
			st.markdown('### Save Model')
			if st.button('SAVE MODEL'):
				#Exporting the trained model
				#joblib.dump(model_griddtree,'model/DTgridTreeModel.ml')
				  with open('model/model_griddtree.pkl', 'wb') as file: 
					  pickle.dump(model_griddtree, file)

		elif pred_type=="RF":
			st.write("### 4. Running Random Forest with GridSearchCV Algorithm on Sample")
			#Random Forest with GridSearchCV Model
			from sklearn.ensemble import RandomForestClassifier
			classifier_rfg=RandomForestClassifier(random_state=33,n_estimators=23)
			parameters=[{'min_samples_split':[2,3],'criterion':['gini','entropy'],'min_samples_leaf':[1,2]}]
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
										 index = ['Actual Low Diabetes', 'Actual High Diabetes'], 
										 columns = ['Predicted Low Diabetes', 'Predicted High Diabetes'])
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
			index = ['Actual Low Diabetes', 'Actual High Diabetes']
			st.text(classification_report(y_test,y_predict_rf, target_names= index))
			st.write('The accuracy score for the application of RFwithGridSearchCV algorithm is ', (accuracy_score(y_test,y_predict_rf)))
			st.write('So from the above results from the Precision, Recall, F1-Score, were displayed.  The support is the number of data samples used in the report')
			st.markdown('### Save Model')
			if st.button('SAVE MODEL'):
				#Exporting the trained model
				#joblib.dump(model_gridrf,'model/RFgridTreeModel.ml')
				with open('model/model_gridrf.pkl', 'wb') as file: 
					  pickle.dump(model_gridrf, file)


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
										 index = ['Actual Low Diabetes', 'Actual High Diabetes'], 
										 columns = ['Predicted Low Diabetes', 'Predicted High Diabetes'])
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
			index = ['Actual Low Diabetes', 'Actual High Diabetes']
			st.text(classification_report(y_test,y_predict_nb, target_names=index))
			st.write('The accuracy score for the application of Naive Bayes Algorithm is ', (accuracy_score(y_test,y_predict_nb)))
			st.write('So from the above results from the Precision, Recall, F1-Score, were displayed.  The support is the number of data samples used in the report')
			st.markdown('### Save Model')
			if st.button('SAVE MODEL'):
				#Exporting the trained model
				#joblib.dump(model_nb,'model/NBModel.ml')
				with open('model/model_nb.pkl', 'wb') as file: 
					  pickle.dump(model_nb, file)
				

		elif pred_type=="KNN":
			st.write("### 6. Running K-Nearest Neighbour Model on Sample")
			#K-Nearest Neighbour Model
			from sklearn.neighbors import KNeighborsClassifier
			model_knn = KNeighborsClassifier(n_neighbors=10,metric='euclidean', algorithm='brute') # Maximum accuracy for n=10
			model_knn.fit(X_train,y_train)
			#Predicting the model
			y_predict_knn = model_knn.predict(X_test)
			#Finding accuracy, precision, recall and confusion matrix
			st.markdown('### Confusion Matrix for K-Nearest Neighbour')
			st.write('The Confusion Matrix is used to know the performance of a Machine learning classification. It is represented in a matrix form. Confusion Matrix gives a comparison between Actual and predicted values.')
			knncf=confusion_matrix(y_test,y_predict_knn)
			knncf_data = pd.DataFrame(knncf,
										 index = ['Actual Low Diabetes', 'Actual High Diabetes'], 
										 columns = ['Predicted Low Diabetes', 'Predicted High Diabetes'])
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
			index = ['Actual Low Diabetes', 'Actual High Diabetes']
			st.text(classification_report(y_test,y_predict_knn, target_names=index))
			st.write('The accuracy score for the application of K-Nearest Neighbour Algorithm is ', (accuracy_score(y_test,y_predict_knn)))
			st.write('So from the above results from the Precision, Recall, F1-Score, were displayed.  The support is the number of data samples used in the report')
			
			st.markdown('### Save Model')
			if st.button('SAVE MODEL'):
				#Exporting the trained model
				#joblib.dump(model_knn,'model/KNNModel.ml')
				with open('model/model_knn.pkl', 'wb') as file: 
					  pickle.dump(model_knn, file)

		elif pred_type=="XGBoost":
			st.write("### 7. Running XGBoost Classifer Algorithm on Sample")
			#XGBOOST Model
			from sklearn.preprocessing import LabelEncoder
			encoder = LabelEncoder()
			y_train = encoder.fit_transform(y_train)
			y_test = encoder.transform(y_test)
			from xgboost import XGBClassifier
			from sklearn.model_selection import RandomizedSearchCV
			model_xgb = XGBClassifier()
			param_grid_xgb = {
				'n_estimators': [100, 200, 300],
				'learning_rate': [0.01, 0.1, 0.2, 0.3],
				'max_depth': [3, 4, 5, 6, 7],
				'min_child_weight': [1, 3, 5],
				'subsample': [0.5, 0.7, 1.0],
				'colsample_bytree': [0.5, 0.7, 1.0],
				'gamma': [0, 0.1, 0.2, 0.3]
				}
			xgb_random = RandomizedSearchCV(estimator=model_xgb, param_distributions=param_grid_xgb, n_iter=100, cv=3, verbose=2, n_jobs=-1)
			print("Tuning XGBoost...")
			xgb_random.fit(X_train, y_train)
			best_xgb = xgb_random.best_estimator_
			# Predictions and evaluation
			y_predict_xgb = best_xgb.predict(X_test)
			# model_xgb.fit(X_train,y_train)
			#Predicting the model
			# y_predict_xgb = model_xgb.predict(X_test)
			#Finding accuracy, precision, recall and confusion matrix
			st.write('The Confusion Matrix is used to know the performance of a Machine learning classification. It is represented in a matrix form. Confusion Matrix gives a comparison between Actual and predicted values.')
			st.markdown('### Confusion Matrix for XGBoost')
			xgbcf=confusion_matrix(y_test,y_predict_xgb)
			xgbcf_data = pd.DataFrame(xgbcf,
										 index = ['Actual Low Diabetes', 'Actual High Diabetes'], 
										 columns = ['Predicted Low Diabetes', 'Predicted High Diabetes'])
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
			index = ['Actual Low Diabetes', 'Actual High Diabetes']
			st.text(classification_report(y_test,y_predict_xgb, target_names=index))
			st.write('The accuracy score for the application of XGBoost Algorthm is ', (accuracy_score(y_test,y_predict_xgb)))
			st.write('So from the above results from the Precision, Recall, F1-Score, were displayed.  The support is the number of data samples used in the report')
			st.markdown('### Save Model')
			if st.button('SAVE MODEL'):
				#Exporting the trained model

				#model_xgb.save_model('model/XGBModel.json')
				#joblib.dump(model_xgb,'model/XGBModel.ml')
				with open('model/model_xgb.pkl', 'wb') as file: 
					  pickle.dump(model_xgb, file)

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
										 index = ['Actual Low Diabetes', 'Actual High Diabetes'], 
										 columns = ['Predicted Low Diabetes', 'Predicted High Diabetes'])
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
			index = ['Actual Low Diabetes', 'Actual High Diabetes']
			st.text(classification_report(y_test,y_predict_mlp, target_names=index))
			st.write('The accuracy score for the application of Artificial Neural Network Algorithm is ', (accuracy_score(y_test,y_predict_mlp)))
			st.write('So from the above results from the Precision, Recall, F1-Score, were displayed.  The support is the number of data samples used in the report')
			st.markdown('### Save Model')
			if st.button('SAVE MODEL'):
				#Exporting the trained model
				#joblib.dump(model_mlp,'model/ANNModel.ml')
				with open('model/model_mlp.pkl', 'wb') as file: 
					  pickle.dump(model_mlp, file)
					  
		
		elif pred_type == "Ensemble":
			st.write("### 10 Running Ensemble Learning with Hyperparameter Tuned Models")
			# Hyperparameter Tuned Decision Tree

			# Import necessary libraries
			import numpy as np
			import pandas as pd
			from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
			from sklearn.tree import DecisionTreeClassifier
			from sklearn.linear_model import LogisticRegression
			from sklearn.neighbors import KNeighborsClassifier
			from sklearn.naive_bayes import GaussianNB
			from sklearn.model_selection import RandomizedSearchCV
			# Define the hyperparameter search spaces for the models
			rf_param_dist = {'n_estimators': np.random.randint(100, 1000, 10),
                 'max_depth': np.random.randint(5, 20, 5),
                 'min_samples_split': np.random.randint(2, 11, 5)}
			gb_param_dist = {'n_estimators': np.random.randint(100, 1000, 10),
                 'max_depth': np.random.randint(3, 10, 5),
                 'learning_rate': np.random.uniform(0.01, 0.3, 5)}
			dt_param_dist = {'max_depth': np.random.randint(3, 10, 5)}
			lr_param_dist = {'C': np.random.uniform(0.1, 10, 5)}
			#knn_param_dist = {'n_neighbors': np.random.randint(1, 21, 10)}
			# Perform hyperparameter tuning using RandomizedSearchCV
			
			rf_model = RandomForestClassifier(random_state=42)
			rf_search = RandomizedSearchCV(rf_model, param_distributions=rf_param_dist, n_iter=3, cv=3, scoring='accuracy', random_state=42)
			rf_search.fit(X_train, y_train)
			best_rf_model = rf_search.best_estimator_
			
			gb_model = GradientBoostingClassifier(random_state=42)
			gb_search = RandomizedSearchCV(gb_model, param_distributions=gb_param_dist, n_iter=3, cv=3, scoring='accuracy', random_state=42)
			gb_search.fit(X_train, y_train)
			best_gb_model = gb_search.best_estimator_
			
			dt_model = DecisionTreeClassifier(random_state=42)
			dt_search = RandomizedSearchCV(dt_model, param_distributions=dt_param_dist, n_iter=3, cv=3, scoring='accuracy', random_state=42)
			dt_search.fit(X_train, y_train)
			best_dt_model = dt_search.best_estimator_
			
			lr_model = LogisticRegression(random_state=42)
			lr_search = RandomizedSearchCV(lr_model, param_distributions=lr_param_dist, n_iter=3, cv=3, scoring='accuracy', random_state=42)
			lr_search.fit(X_train, y_train)
			best_lr_model = lr_search.best_estimator_
			
			
			nb_model = GaussianNB()
			nb_model.fit(X_train, y_train)
			
			# Create the ensemble model
			ensemble_model = VotingClassifier(estimators=[('rf', best_rf_model), ('gb', best_gb_model),
                                              ('dt', best_dt_model), ('lr', best_lr_model),
                                               ('nb', nb_model)],
                                  voting='soft')
			ensemble_model.fit(X_train, y_train)
			
			# Evaluate the ensemble model
			
			y_pred_ensemble = ensemble_model.predict(X_test)
			# Display results
			st.markdown('### Confusion Matrix for Ensemble Learning')
			st.write('The Confusion Matrix is used to know the performance of a Machine learning classification. It is represented in a matrix form. Confusion Matrix gives a comparison between Actual and predicted values.')
			ensemble_cf = confusion_matrix(y_test, y_pred_ensemble)
			ensemble_cf_data = pd.DataFrame(ensemble_cf, index=['Actual Low Diabetes', 'Actual High Diabetes'], columns=['Predicted Low Diabetes', 'Predicted High Diabetes'])
			#st.dataframe(ensemble_cf_data)
			# Plotting the Confusion Matrix
			f4, ax = plt.subplots(figsize=(15, 10))
			sns.heatmap(ensemble_cf_data, annot=True, fmt="d")
			ax.set_title('Ensemble Learning Confusion Matrix')
			ax.set_ylabel('Actual Values')
			ax.set_xlabel('Predicted Values')
			st.pyplot(f4)
			# Classification Report
			st.markdown('### Classification Report for the application of Ensemble Learning')
			st.write('A Classification report is used to measure the quality of predictions from a classification algorithm. How many predictions are True and how many are False. More specifically, True Positives, False Positives, True negatives and False Negatives are used to predict the metrics of a classification report as shown below')
			st.markdown('### Classification Result')
			index = ['Actual Low Diabetes', 'Actual High Diabetes']
			st.text(classification_report(y_test, y_pred_ensemble, target_names=index))
			st.write('The accuracy score for the application of Ensemble Learning is ', (accuracy_score(y_test, y_pred_ensemble)))
			st.write('So from the above results from the Precision, Recall, F1-Score, were displayed.  The support is the number of data samples used in the report')
			st.markdown('### Save Model')
			if st.button('SAVE MODEL'):
				with open('model/ensemble_model.pkl', 'wb') as file:
					pickle.dump(ensemble_model, file)



	else:
		import pandas as pd
		import numpy as np
		import joblib
		from sklearn.preprocessing import OneHotEncoder, StandardScaler
		from sklearn.compose import ColumnTransformer
		from sklearn.pipeline import Pipeline
		import logging # Set up logging
		logging.basicConfig(level=logging.DEBUG) # Function to preprocess data
		def preprocess_data(data):
			categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
			numerical_cols = [col for col in data.columns if data[col].dtype in ['int64', 'float64']] # Pipelines for numerical and categorical preprocessing
			numerical_transformer = StandardScaler()
			categorical_transformer = OneHotEncoder(handle_unknown='ignore')
			preprocessor = ColumnTransformer(
				transformers=[
					('num', numerical_transformer, numerical_cols),
					('cat', categorical_transformer, categorical_cols)
					]) # Apply preprocessing
			data_preprocessed = preprocessor.fit_transform(data)
			return data_preprocessed, preprocessor
		st.title('Prediction') # Load your data (adjust path accordingly)
		if st.checkbox('Show raw data'):
			st.dataframe(data) # Split data into features (X) and target variable (y)
		X = data.iloc[:, :-1]  # Features
		y = data.iloc[:, -1]   # Target variable, assuming it's the last column
		# Preprocess data
		X_preprocessed, preprocessor = preprocess_data(X) # Create sliders for feature input
		feature_input = {}
		for col in X.columns:
			feature_input[col] = st.select_slider(f"Select {col} value for prediction:", options=np.unique(X[col])) 
			# Load or upload model
		model_file = st.file_uploader("Upload model", type=['pkl', 'joblib'])
		if model_file:
			model = joblib.load(model_file)
		else:
			model = joblib.load('model/ensemble_model.pkl') # Adjust the threshold for prediction (optional, adjust as needed)
		high_risk_threshold = 0.99
		if st.button('Predict'): 
			# Convert feature input to DataFrame
			input_df = pd.DataFrame([feature_input]) 
			# Preprocess input to match training configuration
			input_preprocessed = preprocessor.transform(input_df) # Prediction
			pred = model.predict(input_preprocessed)
			confidence_scores = model.predict_proba(input_preprocessed) # Logging for debugging
			logging.debug("Original Input: %s", feature_input)
			logging.debug("Preprocessed Input: %s", input_preprocessed)
			logging.debug("Prediction: %s", pred)
			logging.debug("Confidence Scores: %s", confidence_scores)

			# Display results using adjusted threshold
			if confidence_scores[0][0]: #>= high_risk_threshold:
				st.warning('Predicted diabetes risk: High')
				st.markdown(f'Confidence score: {confidence_scores[0][0]:.2f}')
				st.markdown('Advice: Please consult a healthcare professional for further evaluation.')
			else:
				st.success('Predicted diabetes risk: Low')
				st.markdown(f'Confidence score: {confidence_scores[0][1]:.2f}')
				st.markdown('Advice: Maintain a healthy lifestyle.')

		
if __name__ == '__main__':
    main()
