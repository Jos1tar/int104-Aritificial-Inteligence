import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import seaborn as sns

df = pd.read_csv('CW_Data.csv')


# Dropping the 'Index' column as it's just an identifier
plot_data = df.drop('Index', axis=1)

plt.figure(figsize=(12, 8))
plt.boxplot(plot_data, patch_artist=True, labels=plot_data.columns)
plt.xticks(rotation=45)
plt.title('Box plot of CW_Data columns')
plt.show()


# Standardizing the data
scaler = StandardScaler()
plot_data_scaled = scaler.fit_transform(plot_data)

# Applying PCA
pca = PCA()
pca_data = pca.fit_transform(plot_data_scaled)
pca_data_df = pd.DataFrame(data=pca_data, columns=[f'PC{i+1}' for i in range(pca_data.shape[1])])


plt.figure(figsize=(12, 8))
plt.boxplot(pca_data_df, patch_artist=True, labels=pca_data_df.columns)
plt.xticks(rotation=45)
plt.title('Box plot of PCA components')
plt.show()

# Load the data
df = pd.read_csv('CW_Data.csv')

# Remove the index column if it exists
if 'Index' in df.columns:
    df.drop(columns=['Index'], inplace=True)

# Scaling the data
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Plotting the box plot
plt.figure(figsize=(12, 6))
df_scaled.boxplot()
plt.xticks(rotation=45)
plt.title('Box Plot of Scaled Data')
plt.show()

# Removing the 'Grade', 'Gender', and 'Programme' columns
pca_data = df_scaled.drop(columns=['Grade', 'Gender', 'Programme'])

# Applying PCA and reducing to 2 dimensions
pca = PCA(n_components=2)
pca_result = pca.fit_transform(pca_data)

# Creating a DataFrame for the PCA result
pca_df = pd.DataFrame(data = pca_result, columns = ['principal component 1', 'principal component 2'])
# Adding back the 'Programme' column for coloring
pca_df['Programme'] = df['Programme'].values

# Plotting
plt.figure(figsize=(10, 8))
sns.scatterplot(x='principal component 1', y='principal component 2', hue='Programme', data=pca_df, palette='viridis')
plt.title('PCA of Scaled Data')
plt.show()

plt.figure(figsize=(12, 8))
plt.boxplot(pca_result_df, patch_artist=True, labels=pca_result_df.columns)
plt.xticks(rotation=45)
plt.title('Box plot of PCA components excluding Programme and Index')
plt.show()

colors = {1: 'red', 2: 'green', 3: 'blue', 4: 'purple'}

plt.figure(figsize=(8, 6))
for programme, group in pca_result_df_excluded.groupby('Programme'):
    plt.scatter(group['PC1'], group['PC4'], label=f'Programme {programme}', color=colors[programme])
plt.xlabel('PC1')
plt.ylabel('PC4')
plt.title('Scatter plot of PC1 vs PC4 with Programme color coding (excluding Gender and Grade)')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
for programme, group in pca_result_df_excluded.groupby('Programme'):
    plt.scatter(group['PC1'], group['PC3'], label=f'Programme {programme}', color=colors[programme])
plt.xlabel('PC1')
plt.ylabel('PC3')
plt.title('Scatter plot of PC1 vs PC3 with Programme color coding (excluding Gender and Grade)')
plt.legend()
plt.show()

# Removing the 'Grade', 'Gender', and 'Programme' columns
pca_data = df_scaled.drop(columns=['Gender', 'Programme'])

# Applying PCA and reducing to 2 dimensions
pca = PCA(n_components=2)
pca_result = pca.fit_transform(pca_data)

# Creating a DataFrame for the PCA result
pca_df = pd.DataFrame(data = pca_result, columns = ['principal component 1', 'principal component 2'])
# Adding back the 'Programme' column for coloring
pca_df['Programme'] = df['Programme'].values

# Plotting
plt.figure(figsize=(10, 8))
sns.scatterplot(x='principal component 1', y='principal component 2', hue='Programme', data=pca_df, palette='viridis')
plt.title('PCA of Scaled Data')
plt.show()

plot_data = principalDf_all[['PC1', 'PC3']].copy()
plot_data['Programme'] = df['Programme_encoded']

# Plotting scatter plot with PC1 and PC3, coloring by Programme
cmap = plt.cm.get_cmap('viridis', len(plot_data['Programme'].unique()))
plt.figure(figsize=(8,6))
scatter = plt.scatter(plot_data['PC1'], plot_data['PC3'], c=plot_data['Programme'], cmap=cmap)
plt.colorbar(scatter)
plt.xlabel('PC1')
plt.ylabel('PC3')
plt.title('Scatter plot of PC1 vs PC3 colored by Programme')
plt.show()