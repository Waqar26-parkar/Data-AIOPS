import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the data
df = pd.read_csv('C:\\Project-completion\\Data-Modeling-Aiops\\New-Dataset_Aiops.csv')

# Convert 'clock' to datetime
df['clock'] = pd.to_datetime(df['clock'])

# Identify columns with missing data
columns_with_missing = df.columns[df.isnull().any()].tolist()
print("Columns with missing data:", columns_with_missing)

# Separate numeric and non-numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_columns = [col for col in df.columns if col not in numeric_columns]

# Remove 'server_events' and 'website_events' from imputation
columns_to_impute = [col for col in columns_with_missing if col not in ['server_events', 'website_events']]

print("Columns to impute:", columns_to_impute)

# Handle non-numeric columns (excluding 'server_events' and 'website_events')
for col in non_numeric_columns:
    if col in columns_to_impute:
        # For non-numeric columns, fill with the most frequent value
        df[col].fillna(df[col].mode()[0], inplace=True)

# Create dictionaries for server and website events
server_events_dict = {np.nan: 0}  # 0 for no event (NaN)
website_events_dict = {np.nan: 0}  # 0 for no event (NaN)

# Populate the dictionaries with events and their corresponding numbers
for col, event_dict in [('server_events', server_events_dict), ('website_events', website_events_dict)]:
    events = df[col].dropna().unique()
    for i, event in enumerate(events, start=1):
        event_dict[event] = i

# Apply the dictionaries to the dataframe
df['server_events'] = df['server_events'].map(server_events_dict).fillna(0).astype(int)
df['website_events'] = df['website_events'].map(website_events_dict).fillna(0).astype(int)

# Print the event mappings
print("\nServer Events Mapping:")
print({v: k for k, v in server_events_dict.items()})
print("\nWebsite Events Mapping:")
print({v: k for k, v in website_events_dict.items()})

# Select numeric features for KNN imputation
numeric_columns_to_impute = [col for col in numeric_columns if col in columns_to_impute]

if numeric_columns_to_impute:
    # Create a copy of the numeric columns to impute
    df_numeric = df[numeric_columns_to_impute].copy()
    
    # Handle outliers using IQR method
    for col in numeric_columns_to_impute:
        Q1 = df_numeric[col].quantile(0.25)
        Q3 = df_numeric[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_numeric[col] = df_numeric[col].clip(lower_bound, upper_bound)
    
    # Normalize the features using MinMaxScaler
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=numeric_columns_to_impute, index=df.index)
    
    # Apply KNN imputation
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df_scaled), 
                              columns=numeric_columns_to_impute,
                              index=df.index)
    
    # Inverse transform the imputed values
    df_imputed = pd.DataFrame(scaler.inverse_transform(df_imputed), 
                              columns=numeric_columns_to_impute,
                              index=df.index)
    
    # Replace only the missing values in the original dataframe
    for col in numeric_columns_to_impute:
        df.loc[df[col].isnull(), col] = df_imputed.loc[df[col].isnull(), col]

# Verify no missing data remains in imputed columns
print("\nMissing data after imputation:")
print(df[columns_to_impute].isnull().sum())

# Display summary statistics for numeric columns that were imputed
print("\nSummary statistics after imputation (numeric columns):")
print(df[numeric_columns_to_impute].describe())

# Feature Engineering
# Extract time-based features
df['hour'] = df['clock'].dt.hour
df['day_of_week'] = df['clock'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Create lag features for numeric columns
lag_columns = ['CPU_user_time', 'CPU_system_time', 'CPU_idle_time', 'Available_memory', 'Memory_utilization']
for col in lag_columns:
    df[f'{col}_lag1'] = df[col].shift(1)
    df[f'{col}_lag3'] = df[col].shift(3)

# Calculate rolling averages
for col in lag_columns:
    df[f'{col}_rolling_mean_3'] = df[col].rolling(window=3).mean()
    df[f'{col}_rolling_mean_6'] = df[col].rolling(window=6).mean()

# Drop rows with NaN values created by lag and rolling features
df.dropna(inplace=True)

# Correlation Analysis
correlation_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

print("Correlation heatmap saved as 'correlation_heatmap.png'")

# Dimensionality Reduction
# Select numeric columns for PCA
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Perform PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
pca_result = pca.fit_transform(df[numeric_cols])

# Create a DataFrame with PCA results
pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])

# Concatenate PCA results with original non-numeric columns
final_df = pd.concat([df[non_numeric_columns], pca_df], axis=1)

print(f"Reduced dimensions from {len(numeric_cols)} to {pca_result.shape[1]}")

# Anomaly Detection
# Perform Isolation Forest for anomaly detection
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomalies = iso_forest.fit_predict(df[numeric_cols])

# Add anomaly column to the dataset
df['is_anomaly'] = anomalies

# Analyze anomalies
print("Number of anomalies detected:", sum(anomalies == -1))
print("Percentage of anomalies:", sum(anomalies == -1) / len(anomalies) * 100)

# Time Series Decomposition
# Select a numeric column for decomposition (e.g., 'cpu_user')
column_to_decompose = 'CPU_user_time'

# Perform time series decomposition
result = seasonal_decompose(df[column_to_decompose], model='additive', period=24)  # Assuming hourly data

# Plot the decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
result.observed.plot(ax=ax1)
ax1.set_title('Observed')
result.trend.plot(ax=ax2)
ax2.set_title('Trend')
result.seasonal.plot(ax=ax3)
ax3.set_title('Seasonal')
result.resid.plot(ax=ax4)
ax4.set_title('Residual')
plt.tight_layout()
plt.savefig(f'{column_to_decompose}_decomposition.png')
plt.close()

print(f"Time series decomposition plot saved as '{column_to_decompose}_decomposition.png'")

# Save the final preprocessed dataset
final_df.to_csv('Final_Preprocessed_Dataset.csv', index=False)
print("Final preprocessed dataset saved as 'Final_Preprocessed_Dataset.csv'")

# Save the original dataset with added features and anomaly detection
df.to_csv('Enhanced_Dataset_with_Anomalies.csv', index=False)
print("Enhanced dataset with anomalies saved as 'Enhanced_Dataset_with_Anomalies.csv'")