from itertools import combinations

import openpyxl
import pip
from sklearn.tree import DecisionTreeClassifier


# Function to evaluate model with different combinations of features
def evaluate_feature_combinations(X, y, features):
    results = {}
    for r in range(1, len(features) + 1):
        for subset in combinations(features, r):
            # Selecting the current combination of features
            X_subset = X[list(subset)]
            X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42)
            model = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=20, min_samples_leaf=10, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[subset] = accuracy
    return results

# List of features to consider
features = X.columns.tolist()

# Evaluating combinations
results = evaluate_feature_combinations(X, y, features)

# Finding the best combination
best_combination = max(results, key=results.get)
best_accuracy = results[best_combination]
best_combination, best_accuracy


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np

# Loading data from the Excel file
data = pd.read_excel('CW_Data.xlsx')

# Selecting specific columns for the experiment
columns_to_use = ['Total', 'MCQ', 'Q2', 'Q3', 'Q4', 'Q5', 'Grade']
X = data[columns_to_use]
y = data['Programme']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# RandomForest Parameters
n_estimators = 50
max_depth = 8
min_samples_split = 20
min_samples_leaf = 15

# Initialize RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=42)
# Train the model
rf_clf.fit(X_train, y_train)
# Make predictions
predictions_rf = rf_clf.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, predictions_rf)
f1 = f1_score(y_test, predictions_rf, average='weighted')

# Display results
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np

# Loading data from the Excel file
data = pd.read_excel('CW_Data.xlsx')

# Selecting specific columns for the experiment
columns_to_use = ['Total', 'MCQ', 'Q2', 'Q3', 'Q4', 'Q5', 'Grade']
X = data[columns_to_use]
y = data['Programme']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# RandomForest Parameters
n_estimators = 50
max_depth = 8
min_samples_split = 20
min_samples_leaf = 15

# Initialize RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=42)
# Train the model
rf_clf.fit(X_train, y_train)
# Make predictions
predictions_rf = rf_clf.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, predictions_rf)
f1 = f1_score(y_test, predictions_rf, average='weighted')

# Display results
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')

fig, axs = plt.subplots(4, 1, figsize=(5,10))
parameters = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
parameter_titles = ['Number of Estimators', 'Max Depth', 'Min Samples Split', 'Min Samples Leaf']

for i, parameter in enumerate(parameters):
    axs[i].plot(n_estimators_range if parameter == 'n_estimators' else max_depth_range if parameter == 'max_depth' else min_samples_split_range if parameter == 'min_samples_split' else min_samples_leaf_range,
                results_accuracy[parameter], label='Accuracy', marker='o')
    axs[i].plot(n_estimators_range if parameter == 'n_estimators' else max_depth_range if parameter == 'max_depth' else min_samples_split_range if parameter == 'min_samples_split' else min_samples_leaf_range,
                results_f1[parameter], label='F1 Score', marker='s')
    axs[i].set_title(parameter_titles[i])
    axs[i].set_xlabel(parameter_titles[i])
    axs[i].set_ylabel('Score')
    axs[i].legend()

plt.tight_layout()
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_excel('CW_Data.xlsx')

columns_to_use = ['Total', 'MCQ', 'Q2', 'Q3', 'Q4', 'Q5', 'Grade']
X = data[columns_to_use]

# Proceed with the train-test split, model training, and predictions as before
y = data['Programme']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

n_estimators =50  # Number of trees in the forest
max_depth = 5  # Maximum depth of the tree
min_samples_split = 20  # Minimum number of samples required to split an internal node
min_samples_leaf = 10


# Training the RandomForestClassifier with the selected columns
rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=42)
rf_clf.fit(X_train, y_train)

# Making predictions with the selected columns
predictions_rf = rf_clf.predict(X_test)

# Calculating accuracy and F1 score for each program with the selected columns
accuracy_rf = accuracy_score(y_test, predictions_rf)
f1_scores_rf = f1_score(y_test, predictions_rf, average=None)

# Calculating mean accuracy and mean F1 score with the selected columns
mean_accuracy_rf = np.mean(accuracy_rf)
mean_f1_rf = np.mean(f1_scores_rf)

# Calculating variance of F1 scores with the selected columns
var_f1_rf = np.var(f1_scores_rf)

# Plotting the confusion matrix as a heatmap
cm = confusion_matrix(y_test, predictions_rf)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

{'Overall Accuracy': accuracy_rf, 'F1 Scores by Program': f1_scores_rf, 'Mean Accuracy': mean_accuracy_rf, 'Mean F1 Score': mean_f1_rf, 'Variance of F1 Scores': var_f1_rf}

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the data from the Excel file
data = pd.read_excel('CW_Data.xlsx')

# Selecting features for PCA
features = ['Total', 'MCQ',  'Q2', 'Q3', 'Q4', 'Q5', 'Grade']
X = data[features]

# Standardizing the features for PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying PCA to reduce dimensions to 2
pca = PCA(n_components=2)
X_pca_transformed = pca.fit_transform(X_scaled)

# Training the polynomial SVM model
svm_poly = SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo')
svm_poly.fit(X_pca_transformed, data['Programme'])

# Creating a mesh to plot in
x_min, x_max = X_pca_transformed[:, 0].min() - 1, X_pca_transformed[:, 0].max() + 1
y_min, y_max = X_pca_transformed[:, 1].min() - 1, X_pca_transformed[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predicting and plotting
Z = svm_poly.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
plt.scatter(X_pca_transformed[:, 0], X_pca_transformed[:, 1], c=data['Programme'], s=20, edgecolor='k', cmap='viridis')
plt.title('Polynomial SVM on PCA-transformed Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the data from the Excel file
data = pd.read_excel('CW_Data.xlsx')

# Selecting features for PCA
features = ['Total', 'MCQ',  'Q2', 'Q3', 'Q4', 'Q5', 'Grade']
X = data[features]

# Standardizing the features for PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying PCA to reduce dimensions to 2
pca = PCA(n_components=2)
X_pca_transformed = pca.fit_transform(X_scaled)

# Training the linear SVM model
svm_linear = SVC(kernel='linear', C=1, decision_function_shape='ovo')
svm_linear.fit(X_pca_transformed, data['Programme'])

# Creating a mesh to plot in
x_min, x_max = X_pca_transformed[:, 0].min() - 1, X_pca_transformed[:, 0].max() + 1
y_min, y_max = X_pca_transformed[:, 1].min() - 1, X_pca_transformed[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predicting and plotting
Z = svm_linear.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
plt.scatter(X_pca_transformed[:, 0], X_pca_transformed[:, 1], c=data['Programme'], s=20, edgecolor='k', cmap='viridis')
plt.title('Linear SVM on PCA-transformed Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

!pip install openpyxl
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Load data

df = pd.read_excel('CW_Data.xlsx', engine='openpyxl')

# Define a function to evaluate the model given a set of features
def evaluate_features(features):
    X = df[list(features)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    f1_scores_all = f1_score(y_test, y_pred, average=None)
    f1_variance = np.var(f1_scores_all)
    return accuracy, f1, f1_variance

# List of all possible features excluding 'Index' and 'Programme'
all_features = ['Gender', 'Grade', 'Total', 'MCQ', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']

# Define 'y' as the target variable
y = df['Programme']

# Evaluate all combinations of features
results = []
for r in range(1, len(all_features) + 1):
    for combo in combinations(all_features, r):
        accuracy, f1, f1_variance = evaluate_features(combo)
        results.append((combo, accuracy, f1, f1_variance))

# Find the combination with the highest weighted F1 score and accuracy
best_combination = max(results, key=lambda x: (x[2], x[1]))  # Prioritize F1 score, then accuracy

best_combination

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Load the dataset
data = pd.read_csv('CW_Data.csv')

# Assuming 'Programme' is the target column
X = data.drop(['Programme'], axis=1)
X = X[['Grade', 'MCQ', 'Q2', 'Q3', 'Q4', 'Q5']]
y = data['Programme']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# Define the classifiers
clf1_rf = RandomForestClassifier(n_estimators=100, random_state=1)
clf2_svc = SVC(probability=True, random_state=1)
clf3_nb = GaussianNB()

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a voting classifier
ensemble_clf_rf_svc_nb = VotingClassifier(estimators=[('rf', clf1_rf), ('svc', clf2_svc), ('nb', clf3_nb)], voting='soft')

# Fit the ensemble classifier
ensemble_clf_rf_svc_nb.fit(X_train_scaled, y_train)

# Predictions
y_pred_ensemble = ensemble_clf_rf_svc_nb.predict(X_test_scaled)

# Accuracy and F1 Score
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
f1_ensemble = f1_score(y_test, y_pred_ensemble, average='macro')

# Variance of F1 Scores
f1_scores = [f1_score(y_test, ensemble_clf_rf_svc_nb.predict(X_test_scaled), average=None)]
variance_f1_scores = np.var(np.concatenate(f1_scores))

print(accuracy_ensemble, f1_ensemble, variance_f1_scores)