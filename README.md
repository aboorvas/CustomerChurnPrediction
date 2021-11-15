<h1> PREDICTING CUSTOMER CHURN </h1>
    
<h2> PROBLEM SETTING </h2> 
<h4> For any business to run in a successful manner, it matters a lot to retain the old customers. A customer once lost is always lost. It is much harder to gain the trust and attention of the customer again. Such a scenario is common in almost all the industries. Among all such industries, telecom industry faces the huge problem of customer churn each year. This is because a certain saturation point has been reached already in telecom industry. Competition is at an all-time high and many companies are recognizing the need to improve customer experience and service in order to lower customer churn rates and compete effectively. A company which identifies the causes for churning accurately and takes retention measures immediately is likely to be the giant in the telecom industry. </h4>
    
<h2> PROBLEM DEFINITION </h2> 
<h4> Customer churn is a huge problem of concern for large companies nowadays. It directly impacts the company revenues. It is always more difficult and expensive to acquire a new customer than it is to retain a current paying customer. Hence, finding the factors that leads to churn could help in developing business strategies to retain the customers. A robust and accurate predictive model could be built to predict the customers who are about to churn, and the company could offer them with benefits to retain them. </h4> 
    
<h2> DATA SOURCE </h2>
<h4> This data set is taken from Kaggle under the topic Telco Customer Churn. The dataset can be accessed using the following link - https://www.kaggle.com/blastchar/telco-customer-churn <h4>
    
    <h2> DATA DESCRIPTION </h2>
This dataset has records of 7043 customers with 20 features (Independent variables) and 1 target (Dependent Variable). The target variable (Churn) represents whether a customer has left a company or not. So, the target variable has two states (Yes/No). Hence this can be treated as a Binary Classification Problem. The predictor variables present in the data set are customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService,
OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges. It is obvious that the predictor “customerID” seems not necessary for our Churn prediction model. The other variables should be further analyzed to find if they impact the performance of our model in predicting churning. There are no outliers present in the numerical columns of the data set, but the data set had few nulls. 11 nulls in the “TotalCharges” variable are filled using mean imputation. Few variables were converted from Object to Numeric data type while few were converted to categorical datatype.
    
    <h2> DATA EXPLORATION </h2>
    
1) Correlation Analysis :
Correlation between numerical variables and categorical variables are found separately.
The numerical variables are Tenure, MonthlyCharges and TotalCharges. Tenure and MonthlyCharges showcased more correlation with Total Charges. On further analysis, it was found that the amount in TotalCharges column is proportional to MonthlyCharges multiplied by Tenure. So its redunctant to include TotalCharges in the model. Hence, TotalCharges column is removed to avoid redundancy.
    
2) Target Variable Analysis :
The target Variable “Churn” has showcased a imbalanced Class Distribution. The Yes class is very less than the No class. Since imbalance in class affects the Performance of the model negatively, oversampling technique should be incorporated before model building. SMOTE oversampling technique was used for this purpose as random oversampling didn’t seem like a good option because random oversampling just duplicates the information and causes the model to overfit.
3) Binary Categorical Variable Analysis :
Initially, variables with two outcomes are analysed. It is found that SeniorCitizen and PhoneService variables have a high imbalance and its found that most of the Customers are not senior citizens and also most of the customers have a Phone Service. Further, these binary features are analyzed with respect to Target Variable (Churn). For Calculation purpose, the values of Target Variable are changed to 1 and 0 instead of Yes and No. It ws found that, for gender variable, the average churn rate for both Male and Female is approximately the same which shows that this variable dosen’t bring a valuable predictive power to the machine learning model. Similarly, PhoneService variable also shows similar results. Hence gender and PhoneService variables are ignored before model building.
    
4) Internet Service Variable Analysis:
The 6 variables that comes under Internet Service are StreamingTV, StreamingMovies, OnlineSecurity, OnlineBackup, DeviceProtection and TechSupport. All these variables seem to
have different Churn Rates for their respetive factors because Churn Rate changes accoridng to customers having these services. Its evitable that there is no much difference in StreamingMovies and StreamingTV but it can still bring a good addition to the Model. Hence, these variables could be included. Next, we found out few interesting insights by analyzing the type of Internet service the customer had.
It is found that the most adopted connection is Fiber optic but it has the most churn rate too. Hypothesizing that it may be due to cost related factor, we analyzed the monthly charges variable. It is found that the average monthly charges for fibre optic connection is very high. It might be the reason for the churning. Hence, we decide to keep this variable.
    
5) Phone Service Variable Analysis :
We know that if a customer does not have PhoneService, they cannot have MultipleLines. Therefore, MultipleLines variable includes all the information present in PhoneService by default. This is evident by the sum of values of Yes and No factors in MultipleLines variable which is equal
to the Yes factor of the PhonesService variable. Hence, PhoneService variable can be ignored before model building to avoid redundancy.
    
6) Contract Variable Analysis :
As expected, customers with less duration contract seems to churn more than customers who stay for long duration. This is the main reason why the companies wishes to have long term relationship with the customers.
    
7) Payment Variable Analysis :
It’s so distinct that Customers who pay using Electronice Check are more likely to Churn and it’s the most common Payment Method than others. This finding should be further analyzed to check if they impact anything more with churning.
    
8) Tenure and Monthly Charges Variable Analysis :
The tenure variable distribution shows that, most of the customers are either very new or they have stayed in the company for a long time.The ulimate aim is to keep these cusomers with a tenure up to a few months. MonthlyCharges variable also shows similar distribution and there seems a gap between low and high rates in MonthlyCharges.
    
Analysing the churn rate with MonthlyCharges and tenure variables, it is evident that the customers who is in the company for a long time tend to stay longer. Also, the average tenure(in months) for the people who left (Churn = 1) is 20 months less than the people who stayed (Churn =1). Hence, MonthlyCharges also shows an effect on churn rate.
By analysing Contract and tenure features, both are found to be highly correlated. Customers with long term contract stay in the company for a long period than customers with Short term contracts. Since both variables showcase reduncdant information, Contract variable is removed (Categorical) and tenure is retained (Numerical).
    
    
DATA MINING TASKS
    
• Dropping Unwanted Variables:
After EDA we decided to remove the following variables before training the model.
1. CustomerID
2. Gender
3. Contract
4. TotalCharges
5. PhoneService
    
• One Hot Encoding:
Except Tenure, Monthly Charges and Total Charges, all of the predictors were categorical. Hence, we created dummies for them. We used m-1 dummies for all the models, except KNN as KNN could give different results for m and m-1 dummies.
    
• Label Encoding:
The target variable CHURN had factors as Yes, No. We encoded 1 and 0 for Yes and No respectively.
    
• Scaling:
Since the scales of the variables could impact the results, we scaled the numerical variables like Tenure and Monthly charges to a scale of (0,1) to match the ranges of other variables. Min Max scaling was utilized for the scaling purpose.
    
• Train Test Split:
The data was split into 80% training and 20% testing using sklearn.model_selection in a stratified manner. The training data was then oversampled due to class imbalance. The testing data was not oversampled, as the model has to be tested in an environment representing the real-world proportion.
    
• Over Sampling:
As the target variable is imbalanced with only 26.5% values of the success class, we oversampled the success class using SMOTE sampling. 
    
DATA MINING MODELS 
    
• Random Forest
One of the main reasons why we decided to go with random forest is that it is relatively easier to understand the logic behind the model in classification.
Also, based on our research we found that random forest always does a good job with respect to classification tasks.
We thought of understanding the feature importance of each of the predictors in churning estimation which could be achieved using random forest.
We trained the model using default set of parameters and then hyper tuned the parameters to get the best results. 
    
• KNN Method
We decided to go with knn, as knn has almost no assumptions to be satisfied concerning the data.
We trained the model on default neighbors of 5 intitially.
Later, we hyper tuned the parameters to identify the best k value, weights and metric. 
    
• Logistic Regression
We used logistic regression, as we could interpret the feature importance of each of the predictors using the model co-efficients.
Also, considering the fact that it is easier to implement, interpret, very efficient to train and makes no assumptions about distributions of classes in feature space, we decided to use it.
We trained the model using default values and then hyper tuned the parameters like C, solver, class_weight to obtain the best possible results. 
    
• Support Vector Machine  
Based on our study, we understood that SVM does a good job in classification by creating a line or a hyperplane which separates the data into classes. Also, it had lot of parameters to tune upon which we thought of making use of to get the best possible results.
We trained the model using default values and then hyper tuned the parameters like C, Kernel, Gamma, degree to obtain the best possible results.
    
    
PERFORMANCE EVALUATION
The models were evaluated using three key metrics namely,
1. Sensitivity – Our target is to determine the customers who are about to churn and come up with preventive measures to retain them. Therefore, the overall accuracy doesn’t matter for us. We need to identify the maximum number of churning customers which makes Sensitivity as the key metric to focus upon.
2. ROC Curve – ROC Curve with maximum area under the curve indicates better separation between two classes in terms if probability distribution obtained from the model. Therefore, this metric can be used to compare the performance of different models.
3. Net gain – This is a derived metric which takes into account the cost incurred in retaining an old customer and the cost of finding a new customer. Few assumptions were made in order to use this metric such as,
- It costs the company $500 to find a new customer
- It costs the company $100 to retain an existing customer
Now, we analyze the models deployed using the following parameters. 
    
ROC Curve
Almost all the 4 models (trained on oversampled data) had a good ROC curve. All the 4 models seemed to be a good classifier. It was not possible to choose the best model among the 4 models by using ROC curve. The Area under the curve was also almost the same for each of the 4 models.
The results are shown below:
1) Random Forest
Class Separation : GOOD
AUC : 0.83
Therefore, good classifier.
2) Logistic Regression
Class Separation : GOOD
AUC : 0.83
Therefore, good classifier.
3) SVM
Class Separation : GOOD
AUC : 0.80
Therefore, good classifier.
4) KNN
Class Separation : GOOD
AUC : 0.73
Therefore, moderate classifier.
    
Since we couldn’t differentiate the performance of the models much using the ROC, we look for the next method of evaluating i.e., Sensitivity.
    
SENSITIVITY
Oversampling the training data before training the model is supposed to increase the performance of the model in terms of identifying the success class which is smaller in proportion. This was our initial hypothesis and in-order to check that, we tested the two models, one which was trained on original data (non-oversampled) and other model which was trained on oversampled data. This approach was applied to all 4 of our data mining models i.e. Random Forest, KNN, Logistic Regression, SVM. 

Net Gain
    
For the net gain calculation, we compare 3 scenarios based on the cost assumptions we made initially. The 3 scenarios are as follows:
1) Do nothing – Assume none of the customers will churn
2) Retain everyone – Assume everyone will churn and try to retain everyone
3) Model estimate – Use the model predictions as estimate
    
Case1: Do Nothing
- When we do nothing, we assume none of the customers will churn and encode the target variable as 0.
- Create the classification summary by comparing these encoded values to the original values.
- Use the following formula to calculate the Net Gain.
- The net gain calculated from the above formula is -$747500, which indicates a loss.
    
Case2: Retain everyone
- We assume that all the customers will churn and encode the target variable as 1.
- Create the classification summary by comparing these values to original values.
- The net gain obtained for this strategy is $184100 which means that we will save $184100 if we assume everyone would churn and spend $100 to retain everyone.
    
Case 3: Model estimate
- We use the model to predict the target variable.
- Then, the predicted probabilities for success class is obtained.
- We calculate the classification summary for various cutoffs between 0 to 1.
- After obtaining multiple classification summaries, we calculate the net gain for each of the classification summaries.
- The maximum net gain possible is found from the obtained values.
- This approach is applied to all the 4 models and the best model is chosen based on the maximum net gain value obtained from each model.
The results of the model are shown below:
Logistic Regression:
Maximum Net Gain= $269300
KNN:
Maximum Net Gain=$258300
Random Forest:
Maximum Net Gain=$268500
SVM:
Maximum Net Gain=$288100
We have obtained the best result from SVM. We could save $288100 by choosing the cut off as 0.12. This is clearly visible from the graph shown above.
    
    
PROJECT RESULTS:
We could find that SVM model initially gave a better sensitivity after hyper tuning the parameters. Also, it is the model which gave us the maximum net gain possible for a cut-off 0.12. This net gain is the parameter which a company would be focusing upon to maximize. Therefore, SVM is our winning model.
    
    
INSIGHTS FOR DECISION MAKING:
Hence, SVM Model can be used to identify the Churning Customers with
• Maximum Net Gain
• High Sensitivity
    
IMPACT OF PROJECT OUTCOMES:
By implementing this model, we can generate a report of future churning customers and this report can be sent directly to the customer service team who can then contact customers on the list to better understand their needs or propose new offers, different products or strategy. This would greatly benefit the company in terms of profit and economic development.
