# Fraud-Detection-Using-K-Nearest-Neighbors

**Data Science for Business Final Project: Fraud Detection Using K-Nearest Neighbors (K-NN): A Machine Learning Approach**

Brock Pinagel

College of Business & IT, Lawrence Technological University

INT 7623: Data Science for Business

Dr. Kamel Rushaidat

December 16, 2024

**Introduction**

This project addresses the critical problem of fraud detection in financial transactions using the K-Nearest Neighbors (K-NN) algorithm. Fraud detection is vital for maintaining financial security and trust, as fraudulent activities directly impact both customers and financial institutions. The dataset used for this project comprises 2,512 rows and 16 columns, including transaction details such as TransactionAmount, AccountBalance, TransactionType, and TransactionDate. The target variable, TransactionType, is a binary classification problem with two labels: "Fraud" and "Non-Fraud."

**Load the Data and Visualize Insights**

The data was analyzed to uncover patterns and relationships between features. A correlation heatmap was created to identify dependencies between numerical features, revealing potential predictors of fraudulent transactions. A scatter plot was created to visualize the relationship between TransactionAmount and AccountBalance, stratified by transaction type. Additionally, a bar plot was used to display the distribution of the target variable, showing a balanced class distribution. 

**Prepare the Dataset**

Data preprocessing was conducted to ensure compatibility with the K-NN algorithm. Numerical features such as TransactionAmount and AccountBalance were standardized to have zero mean and unit variance. Missing values in numerical features were imputed using the median, while categorical variables were one-hot encoded. Time-based features were extracted from the TransactionDate column, including the hour of the transaction and the day of the week. Outliers were handled by robust scaling to ensure the model's stability.

**Data Partitioning**

The preprocessed dataset was partitioned into training and testing sets using an 80-20 split. Stratification was applied to ensure that the class distribution in the target variable was maintained in both subsets. This step was critical to prevent bias during model evaluation and to reflect real-world distributions.

**Choosing k Values**

Three values for were selected: 3, 5, and 7. These values were chosen to explore the trade-off between overfitting and underfitting. A smaller focuses on local patterns, which may help capture subtle fraud behaviors. In contrast, larger values smooth predictions and reduce sensitivity to noise. Odd values of were used to avoid ties during classification.

**Training Phase**

The K-NN model was trained on the training dataset for each value (3, 5, and 7). During training, the model calculated the distance between the data points and identified the -nearest neighbors. The majority class among the neighbors was used to assign labels to each transaction. This step emphasized the importance of correctly preprocessing the data to ensure accurate distance calculations.

**Testing Phase**

The trained models were evaluated on the test set. For each , predictions were made by calculating distances to the training points, identifying the nearest neighbors, and assigning the majority class label. The accuracy of each model was computed by comparing the predictions to the actual labels in the test set. This phase provided insights into how well the models generalized to unseen data.

**Evaluation Phase**

The models were further evaluated using confusion matrices, precision, and recall scores. The confusion matrix provided a detailed breakdown of true positives, true negatives, false positives, and false negatives. Precision quantified the proportion of correctly identified fraudulent transactions among all predicted fraudulent transactions. Recall measured the ability of the model to identify all actual fraudulent transactions. These metrics highlighted the strengths and weaknesses of each value in balancing false positives and false negatives.

**Present the Best Model**

The best model was selected based on a combination of accuracy, precision, and recall scores. The model with achieved the best balance between accuracy and the ability to detect fraudulent transactions without generating excessive false positives. This value of offered a compromise between sensitivity to noise and capturing meaningful patterns in the data.

**Conclusion**

This project successfully demonstrated the application of the K-NN algorithm for fraud detection. The preprocessing steps ensured that the data was suitable for distance-based calculations, while the evaluation phase provided insights into the model’s performance. The best model achieved a high level of accuracy and maintained a reasonable balance between precision and recall. Future improvements could include feature engineering to incorporate additional transaction metadata, advanced algorithms such as Random Forest or Gradient Boosting, and hyperparameter tuning to further enhance performance.
