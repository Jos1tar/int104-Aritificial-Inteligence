from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from scipy.stats import mode
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Assuming the DataFrame should be created or loaded here
feature_importance_df = pd.DataFrame()  # Placeholder for actual data loading or creation


# Load dataset
df = pd.read_csv('CW_Data.csv')

# Define features and target
y = df['Programme']
X = df.drop(columns=['Programme'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train a RandomForest Classifier to evaluate feature importance
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Get feature importances
importances = clf.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Load the dataset
feature_importance_df_with_grade = pd.read_csv('CW_Data.csv')

# Select features for clustering
features = feature_importance_df_with_grade.drop(['Index','Gender'], axis=1)

# Normalize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of components for GMM
# For example purposes, let's choose n_components=4
optimal_n_components = 4


# Fit GMM
gmm = GaussianMixture(n_components=optimal_n_components, random_state=0)
gmm.fit(scaled_features)
y_gmm = gmm.predict(scaled_features)

# Perform PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaled_features)

# Plot PCA components with GMM clustering
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_gmm, cmap='viridis', edgecolor='k', s=20)
plt.title('GMM Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# Plot PCA components colored by Programme
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=feature_importance_df_with_grade['Programme'], cmap='viridis', edgecolor='k', s=20)
plt.title('Actual Programmes')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.tight_layout()
plt.show()

# Load the data
df = pd.read_csv('CW_Data.csv')

# Prepare the data
X = df.drop(columns=['Index','Gender'])
y = df['Programme']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit GMM
optimal_n_components = 4 # Assuming the optimal number of components is 3 based on prior analysis
gmm = GaussianMixture(n_components=optimal_n_components, random_state=0)
gmm.fit(X_train_scaled)

# Predict on training data to find the mapping between GMM clusters and actual programmes
y_train_pred = gmm.predict(X_train_scaled)

# We will need to map the GMM cluster labels to actual programmes for accurate prediction. This requires0 additional steps.

# Load the data
df = pd.read_csv('CW_Data.csv')

# Prepare the data
X = df.drop(columns=['Index','Gender','Programme'])  # Dropped an additional 'Grade' column for consistency
Y = df['Programme']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit GMM
optimal_n_components = 4  # Assuming the optimal number of components is 4 based on prior analysis
gmm = GaussianMixture(n_components=optimal_n_components, random_state=0)
gmm.fit(X_train_scaled)

# Predict on training data to find the mapping between GMM clusters and actual programmes
y_train_pred = gmm.predict(X_train_scaled)

# Map GMM cluster labels to actual programmes
labels = np.zeros_like(y_train_pred)
for i in range(optimal_n_components):
    mask = (y_train_pred == i)
    labels[mask] = mode(y_train[mask])[0]

# Predict on test data using GMM
y_test_pred = gmm.predict(X_test_scaled)

# Map predicted cluster labels to actual programme labels
y_test_mapped = np.zeros_like(y_test_pred)
for i in range(optimal_n_components):
    mask = (y_test_pred == i)
    y_test_mapped[mask] = mode(y_train[y_train_pred == i])[0]

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_test_mapped)
conf_matrix = confusion_matrix(y_test, y_test_mapped)

accuracy = round(accuracy, 3)

accuracy, conf_matrix

# Number of components range
n_components_range = range(1, 11)

# Lists to store metrics
aics = []
bics = []
log_likelihoods = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=0).fit(X_train_scaled)
    aics.append(gmm.aic(X_train_scaled))
    bics.append(gmm.bic(X_train_scaled))
    log_likelihoods.append(gmm.score(X_train_scaled) * len(X_train_scaled))

# Plotting
plt.figure(figsize=(10, 7))
plt.plot(n_components_range, aics, label='AIC')
plt.plot(n_components_range, bics, label='BIC')
plt.plot(n_components_range, log_likelihoods, label='Log Likelihood', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Scores')
plt.legend()
plt.title('GMM Metrics vs Number of Components')
plt.show()

# Load the dataset
feature_importance_df_with_grade = pd.read_csv('CW_Data.csv')

# Select features for clustering
features = feature_importance_df_with_grade.drop(['Index','Gender'], axis=1)

# Normalize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of components for GMM
# For example purposes, let's choose n_components=4
optimal_n_components = 6

# Fit GMM
gmm = GaussianMixture(n_components=optimal_n_components, random_state=0)
gmm.fit(scaled_features)
y_gmm = gmm.predict(scaled_features)

# Perform PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaled_features)

# Plot PCA components with GMM clustering
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
scatter1 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_gmm, cmap='viridis', edgecolor='k', s=20)
plt.title('GMM Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(*scatter1.legend_elements(), title="Clusters")

# Plot PCA components colored by Programme
plt.subplot(1, 2, 2)
scatter2 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=feature_importance_df_with_grade['Programme'], cmap='viridis', edgecolor='k', s=20)
plt.title('Actual Programmes')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(*scatter2.legend_elements(), title="Programmes")

plt.tight_layout()
plt.show()

# Load the data
df = pd.read_csv('CW_Data.csv')

# Prepare the data
X = df.drop(columns=['Index','Gender'])
y = df['Programme']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit GMM
optimal_n_components = 6  # Assuming the optimal number of components is based on prior analysis
gmm = GaussianMixture(n_components=optimal_n_components, random_state=0)
gmm.fit(X_train_scaled)

# Predict on training data to find the mapping between GMM clusters and actual programmes
y_train_pred = gmm.predict(X_train_scaled)

# Map GMM cluster labels to actual programmes
labels = np.zeros_like(y_train_pred)
for i in range(optimal_n_components):
    mask = (y_train_pred == i)
    labels[mask] = mode(y_train[mask])[0]

# Manual override: Map cluster 4 to program 2
labels[y_train_pred == 4] = 2

# Predict on test data using GMM
y_test_pred = gmm.predict(X_test_scaled)

# Map predicted cluster labels to actual programme labels
y_test_mapped = np.zeros_like(y_test_pred)
for i in range(optimal_n_components):
    mask = (y_test_pred == i)
    y_test_mapped[mask] = mode(y_train[y_train_pred == i])[0]

# Manual override: Map cluster 4 to program 2
y_test_mapped[y_test_pred == 4] = 2

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_test_mapped)
conf_matrix = confusion_matrix(y_test, y_test_mapped)

accuracy, conf_matrix

# Drop 'Gender', 'Index', and 'Grade' columns
df.drop(['Gender', 'Index', 'Q1'], axis=1, inplace=True)

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Initialize lists to store the inertia and silhouette coefficients
inertia = []
silhouette_coefficients = []

# Determine the optimal number of clusters using the elbow method and silhouette score
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)
    score = silhouette_score(df_scaled, kmeans.labels_)
    silhouette_coefficients.append(score)

# Plotting the elbow graph and the silhouette scores
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(range(2, 11), inertia, marker='o', linestyle='--')
ax1.set_title('Elbow Method For Optimal k')
ax1.set_xlabel('Number of clusters')
ax1.set_ylabel('Inertia')

ax2.plot(range(2, 11), silhouette_coefficients, marker='o', linestyle='--')
ax2.set_title('Silhouette Coefficient For Optimal k')
ax2.set_xlabel('Number of clusters')
ax2.set_ylabel('Silhouette Coefficient')

plt.tight_layout()
plt.show()

# Choosing the optimal number of clusters from the elbow graph
optimal_clusters = 4 # based on the elbow method and silhouette score result

# Apply KMeans with the optimal number of clusters
classifier = KMeans(n_clusters=optimal_clusters, random_state=0)
df['Cluster'] = classifier.fit_predict(df_scaled)

# Apply PCA to reduce dimensions to 2 for visualization
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Adding the cluster information to the PCA reduced dataframe
df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
df_pca['Cluster'] = df['Cluster']

# Plotting Cluster Scatter Plot
plt.figure(figsize=(10,6))
for cluster in df_pca['Cluster'].unique():
    cluster_data = df_pca[df_pca['Cluster'] == cluster]
    plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {cluster}')
plt.title('PCA of Dataset with Cluster Labels')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Plotting Programme Scatter Plot
plt.figure(figsize=(10,6))
for programme in df['Programme'].unique():
    programme_data = df_pca[df['Programme'] == programme]
    plt.scatter(programme_data['PC1'], programme_data['PC2'], label=f'Programme {programme}')
plt.title('PCA of Dataset with Programme Labels')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Choosing the optimal number of clusters from the elbow graph
optimal_clusters = 3 # based on the elbow method and silhouette score result

# Apply KMeans with the optimal number of clusters
classifier = KMeans(n_clusters=optimal_clusters, random_state=0)
df['Cluster'] = classifier.fit_predict(df_scaled)

# Apply PCA to reduce dimensions to 2 for visualization
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Adding the cluster information to the PCA reduced dataframe
df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
df_pca['Cluster'] = df['Cluster']

# Plotting Cluster Scatter Plot
plt.figure(figsize=(10,6))
for cluster in df_pca['Cluster'].unique():
    cluster_data = df_pca[df_pca['Cluster'] == cluster]
    plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {cluster}')
plt.title('PCA of Dataset with Cluster Labels')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Plotting Programme Scatter Plot
plt.figure(figsize=(10,6))
for programme in df['Programme'].unique():
    programme_data = df_pca[df['Programme'] == programme]
    plt.scatter(programme_data['PC1'], programme_data['PC2'], label=f'Programme {programme}')
plt.title('PCA of Dataset with Programme Labels')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


# Map each cluster to the program that is most frequent within the cluster
cluster_to_program = {}
for cluster in np.unique(df['Cluster']):
    cluster_data = df[df['Cluster'] == cluster]
    most_frequent_program = cluster_data['Programme'].mode()[0]
    cluster_to_program[cluster] = most_frequent_program

# Create new predicted labels using the cluster_to_program mapping
predicted_labels_mapped = df['Cluster'].map(cluster_to_program)

# Calculating confusion matrix
true_labels = df['Programme']
cm = confusion_matrix(true_labels, predicted_labels_mapped)

# Plotting the confusion matrix
plt.figure(figsize=(5, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(predicted_labels_mapped), yticklabels=np.unique(true_labels))
plt.title('Confusion Matrix')
plt.ylabel('Actual Programme')
plt.xlabel('Predicted Programme')
plt.show()

# Calculating accuracy for each program
program_accuracy = {}
for programme in np.unique(true_labels):
    program_indices = true_labels == programme
    program_accuracy[programme] = accuracy_score(true_labels[program_indices], predicted_labels_mapped[program_indices])

print('Accuracy for each programme:')
for programme, accuracy in program_accuracy.items():
    print(f'Programme {programme}: {accuracy:.2f}')

# Load the data
X_train = pd.read_csv('CW_Data.csv')

# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Adjusting the code to remove the centroid linkage since it's not supported

# Define the cluster range (for instance from 2 to 10 clusters)
cluster_range = range(2, 11)

# Initialize lists to store silhouette scores for different linkage methods
silhouette_scores_average = []
silhouette_scores_complete = []
silhouette_scores_ward = []

for n_clusters in cluster_range:
    # Average
    clusterer_average = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='average')
    cluster_labels_average = clusterer_average.fit_predict(X_train_scaled)
    silhouette_avg_average = silhouette_score(X_train_scaled, cluster_labels_average)
    silhouette_scores_average.append(silhouette_avg_average)

    # Complete
    clusterer_complete = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='complete')
    cluster_labels_complete = clusterer_complete.fit_predict(X_train_scaled)
    silhouette_avg_complete = silhouette_score(X_train_scaled, cluster_labels_complete)
    silhouette_scores_complete.append(silhouette_avg_complete)

    # Ward
    clusterer_ward = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    cluster_labels_ward = clusterer_ward.fit_predict(X_train_scaled)
    silhouette_avg_ward = silhouette_score(X_train_scaled, cluster_labels_ward)
    silhouette_scores_ward.append(silhouette_avg_ward)

# Plot the silhouette scores for the available linkage methods
plt.figure(figsize=(10, 7))
plt.plot(cluster_range, silhouette_scores_average, label='Average')
plt.plot(cluster_range, silhouette_scores_complete, label='Complete')
plt.plot(cluster_range, silhouette_scores_ward, label='Ward')
plt.title('Silhouette Scores for Different Linkage Methods')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.legend()
plt.show()

# Load the dataset
CW_Data = pd.read_csv('CW_Data.csv')

# Split the data into training and test sets
X = CW_Data.drop(['Index', 'Gender'], axis=1)  # Assuming 'Gender' and 'Index' are not features
y = CW_Data['Programme']  # Correct the target column name
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=2)
train_pca = pca.fit_transform(X_train_scaled)
test_pca = pca.transform(X_test_scaled)

# Perform hierarchical clustering
cluster_4 = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
train_clusters = cluster_4.fit_predict(X_train_scaled)

# Plot the hierarchical clustering
plt.figure(figsize=(10, 7))
sns.scatterplot(x=train_pca[:, 0], y=train_pca[:, 1], hue=train_clusters, palette='deep')
plt.title('Hierarchical Clustering (Training Data)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()

# Plot the distribution of actual programmes in the training set
plt.figure(figsize=(10, 7))
sns.scatterplot(x=train_pca[:, 0], y=train_pca[:, 1], hue=y_train, palette='deep')
plt.title('Actual Program Distribution (Training Data)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Program')
plt.show()

# Predict the clusters on the test set
test_clusters_4 = cluster_4.fit_predict(X_test_scaled)

# Create a DataFrame to analyze the cluster distribution for the Programme
cluster_program_df = pd.DataFrame({'Cluster': test_clusters_4, 'Programme': y_test.values, 'PCA1': test_pca[:, 0], 'PCA2': test_pca[:, 1]})

# Add code for bar plot

# Plot the distribution of Programmes across the clusters using a bar plot
plt.figure(figsize=(10, 6))
cluster_program_counts = cluster_program_df.groupby(['Cluster', 'Programme']).size().unstack(fill_value=0)
cluster_program_counts.plot(kind='bar', stacked=True)
plt.title('Program Distribution Across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Frequency')
plt.show()

# Assign each cluster to the most frequent actual programme
cluster_program_mode = cluster_program_df.groupby('Cluster')['Programme'].agg(lambda x: x.value_counts().index[0])

# Avoid reindexing y_test to prevent NaNs
predicted_program = cluster_program_df['Cluster'].map(cluster_program_mode)

# Align the predicted program indices with y_test
predicted_program.index = y_test.index

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predicted_program)

# Convert confusion matrix to text format
conf_matrix_text = '\n'.join(['\t'.join(map(str, row)) for row in conf_matrix])
print(f'Confusion Matrix:\n{conf_matrix_text}')

# Calculate and print program-wise accuracies
accuracies = {}
for program in y_test.unique():
    program_idx = y_test[y_test == program].index
    accuracies[program] = accuracy_score(y_test.loc[program_idx], predicted_program.loc[program_idx])
print(f'Accuracy by Program: {accuracies}')

# Plot the mean Total, Grade, and Gender for each Programme using bar plots
plt.figure(figsize=(18, 6))

# Plot for Total
plt.subplot(1, 3, 1)
sns.barplot(x='Programme', y='Total', data=df_grouped)
plt.title('Mean Total Score by Programme')

# Plot for Grade
plt.subplot(1, 3, 2)
sns.barplot(x='Programme', y='Grade', data=df_grouped)
plt.title('Mean Grade by Programme')

# Plot for Gender
plt.subplot(1, 3, 3)
sns.barplot(x='Programme', y='Gender', data=df_grouped)
plt.title('Mean Gender by Programme')

plt.tight_layout()
plt.show()