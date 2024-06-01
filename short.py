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

		elif pred_type == "Ensemble":
			st.write("### 10 Running Ensemble Learning with Hyperparameter Tuned Models")
			# Hyperparameter Tuned Decision Tree
			from sklearn.tree import DecisionTreeClassifier
			dt_params = {'min_samples_split': [2, 3], 'criterion': ['gini', 'entropy']}
			dt_model = GridSearchCV(DecisionTreeClassifier(random_state=42, splitter='best'), param_grid=dt_params, scoring='accuracy', cv=2)
			dt_model.fit(X_train, y_train)  # Add this line to fit the model
			# Hyperparameter Tuned Random Forest
			from sklearn.ensemble import RandomForestClassifier
			rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
			rf_model = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=rf_params, scoring='accuracy', cv=2)
			rf_model.fit(X_train, y_train)  # Add this line to fit the model
			from sklearn.naive_bayes import GaussianNB
			nb_model = GaussianNB()
			nb_model.fit(X_train, y_train)  # Add this line to fit the model
			from sklearn.neighbors import KNeighborsClassifier
			knn_model = KNeighborsClassifier()
			knn_model.fit(X_train, y_train)  # Add this line to fit the model
			from sklearn.linear_model import LogisticRegression
			lr_model = LogisticRegression()
			lr_model.fit(X_train, y_train)  # Add this line to fit the model
			# Create the ensemble model with Hard Voting Classifier
			from sklearn.ensemble import VotingClassifier
			classifiers = [('dt', dt_model), ('rf', rf_model), ('nb', nb_model), ('knn', knn_model), ('lr', lr_model)]
			ensemble_model = VotingClassifier(estimators=classifiers, voting='hard')
			# Train the ensemble model
			ensemble_model.fit(X_train, y_train)
			y_pred_ensemble = ensemble_model.predict(X_test)
			# Convert X_predict to a dense numpy array
			X_predict_array = X_predict.toarray()
			# Predict using the ensemble model
			y_pred_ensemble = ensemble_model.predict(X_predict_array)
			
			    # Display results
			st.markdown('### Confusion Matrix for Ensemble Learning')
			st.write('The Confusion Matrix is used to know the performance of a Machine learning classification. It is represented in a matrix form. Confusion Matrix gives a comparison between Actual and predicted values.')
			ensemble_cf = confusion_matrix(y_test, y_pred_ensemble)
			ensemble_cf_data = pd.DataFrame(ensemble_cf, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
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
			index = ['Actual 0', 'Actual 1']
			st.text(classification_report(y_test, y_pred_ensemble, target_names=index))
			st.write('The accuracy score for the application of Ensemble Learning is ', (accuracy_score(y_test, y_pred_ensemble)))
			st.write('So from the above results from the Precision, Recall, F1-Score, were displayed.  The support is the number of data samples used in the report')
			st.markdown('### Save Model')
			if st.button('SAVE MODEL'):
				with open('model/ensemble_model.pkl', 'wb') as file:
					pickle.dump(ensemble_model, file)