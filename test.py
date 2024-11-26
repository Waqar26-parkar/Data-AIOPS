# # Cell 1: Import necessary libraries
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.utils import to_categorical
# import matplotlib.pyplot as plt

# # Cell 2: Load and prepare the data
# # Load your preprocessed data
# df = pd.read_csv('Enhanced_Dataset_with_Anomalies.csv')

# # Assuming 'website_events' is your target column
# X = df.drop(['website_events', 'server_events', 'clock'], axis=1)  # Adjust as needed
# y = df['website_events']

# # Display first few rows and info about the dataset
# print(df.head())
# print(df.info())

# # Cell 3: Encode the target variable
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
# num_classes = len(label_encoder.classes_)
# y_categorical = to_categorical(y_encoded)

# print(f"Number of classes: {num_classes}")

# # Cell 4: Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# # Reshape input data for LSTM (samples, time steps, features)
# X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
# X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

# print(f"Training data shape: {X_train.shape}")
# print(f"Testing data shape: {X_test.shape}")

# # Cell 5: Define the model
# model = Sequential([
#     LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
#     LSTM(32),
#     Dense(num_classes, activation='softmax')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Display model summary
# model.summary()

# # Cell 6: Train the model
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# history = model.fit(
#     X_train, y_train,
#     epochs=10,  # You can adjust this
#     batch_size=5,  # You can adjust this
#     validation_split=0.2,
#     callbacks=[early_stopping]
# )

# # Cell 7: Evaluate the model
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print(f"Test accuracy: {test_accuracy:.4f}")

# # Cell 8: Plot training history
# plt.figure(figsize=(12, 4))

# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# plt.tight_layout()
# plt.show()

# # Cell 9: Save the model
# model.save('event_prediction_model.h5')
# print("Model saved as 'event_prediction_model.h5'")

# # Cell 10: Make predictions (optional)
# y_pred = model.predict(X_test)
# y_pred_classes = np.argmax(y_pred, axis=1)
# y_true_classes = np.argmax(y_test, axis=1)

# # Print some sample predictions
# for i in range(10):
#     true_label = label_encoder.inverse_transform([y_true_classes[i]])[0]
#     pred_label = label_encoder.inverse_transform([y_pred_classes[i]])[0]
#     print(f"True: {true_label}, Predicted: {pred_label}")














import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Data Preparation
def prepare_data(csv_path):
    # Read the dataset
    df = pd.read_csv(csv_path)
    
    # Select features (metrics)
    feature_columns = [
        'Download_Speed', 'Reach_Time', 'Time_to_First_Byte', 
        'HTTP_Probe_Duration', 'Page_Load_Time', 'DNS_Query_Time',
        'Status_ID', 'Failed_step_of_scenario_WEB_HEALTH_CHECK',
        'Interrupts_per_second', 'Load_average_15m_avg', 
        'Load_average_1m_avg', 'Load_average_5m_avg',
        'CPU_utilization', 'CPU_idle_time', 'CPU_iowait_time',
        'CPU_system_time', 'CPU_user_time', 'xvda_Disk_utilization',
        'Boot_Space_Used_in_percent', 'Available_memory_in_percent',
        'Memory_utilization', 'Space_Available', 'Boot_Space_Available',
        'Available_memory', 'Total_memory'
    ]
    
    # Prepare features (X) and targets (y)
    X = df[feature_columns]
    y_website = df['website_events']
    y_server = df['server_events']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode targets
    le_website = LabelEncoder()
    le_server = LabelEncoder()
    
    y_website_encoded = le_website.fit_transform(y_website)
    y_server_encoded = le_server.fit_transform(y_server)
    
    # Convert to one-hot encoding
    y_website_onehot = tf.keras.utils.to_categorical(y_website_encoded)
    y_server_onehot = tf.keras.utils.to_categorical(y_server_encoded)
    
    return X_scaled, y_website_onehot, y_server_onehot, scaler, le_website, le_server

# 2. Create Deep Learning Model
def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 3. Train Models
def train_models(X, y_website, y_server):
    # Split data
    X_train, X_test, y_website_train, y_website_test, y_server_train, y_server_test = train_test_split(
        X, y_website, y_server, test_size=0.2, random_state=42
    )
    
    # Create and train website events model
    website_model = create_model(X.shape[1], y_website.shape[1])
    website_history = website_model.fit(
        X_train, y_website_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Create and train server events model
    server_model = create_model(X.shape[1], y_server.shape[1])
    server_history = server_model.fit(
        X_train, y_server_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate models
    website_eval = website_model.evaluate(X_test, y_website_test)
    server_eval = server_model.evaluate(X_test, y_server_test)
    
    print("\nWebsite Events Model Accuracy:", website_eval[1])
    print("Server Events Model Accuracy:", server_eval[1])
    
    return website_model, server_model, website_history, server_history

# 4. Prediction Function
def predict_events(website_model, server_model, scaler, le_website, le_server, input_metrics):
    # Scale input metrics
    input_scaled = scaler.transform([input_metrics])
    
    # Get predictions
    website_pred_proba = website_model.predict(input_scaled)
    server_pred_proba = server_model.predict(input_scaled)
    
    # Get top 3 most likely events for each
    top_website_indices = np.argsort(website_pred_proba[0])[-3:][::-1]
    top_server_indices = np.argsort(server_pred_proba[0])[-3:][::-1]
    
    # Convert indices back to event names
    website_events = le_website.inverse_transform(top_website_indices)
    server_events = le_server.inverse_transform(top_server_indices)
    
    return website_events, server_events, website_pred_proba, server_pred_proba

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    X, y_website, y_server, scaler, le_website, le_server = prepare_data('your_dataset.csv')
    
    # Train models
    website_model, server_model, website_history, server_history = train_models(X, y_website, y_server)
    
    # Example prediction with your test metrics
    test_metrics = {
        'Download_Speed': 44984.96562,
        'Reach_Time': 0.0042,
        'Time_to_First_Byte': 0.00537,
        'HTTP_Probe_Duration': 2.245,
        'Page_Load_Time': 0.002252,
        'DNS_Query_Time': 2885,
        'Status_ID': 1,
        'Failed_step_of_scenario_WEB_HEALTH_CHECK': 0,
        'Interrupts_per_second': 213.5782956,
        'Load_average_15m_avg': 0,
        'Load_average_1m_avg': 0,
        'Load_average_5m_avg': 0,
        'CPU_utilization': 0.818577,
        'CPU_idle_time': 99.181423,
        'CPU_iowait_time': 0.016706,
        'CPU_system_time': 0.233918,
        'CPU_user_time': 0.517962,
        'xvda_Disk_utilization': 0.04166729,
        'Boot_Space_Used_in_percent': 16.13612725,
        'Available_memory_in_percent': 41.499557,
        'Memory_utilization': 58.500443,
        'Space_Available': 9337511936,
        'Boot_Space_Available': 719982592,
        'Available_memory': 416624640,
        'Total_memory': 1003925504
    }
    
    # Make predictions
    website_events, server_events, website_probs, server_probs = predict_events(
        website_model, server_model, scaler, le_website, le_server, list(test_metrics.values())
    )
    
    # Print results
    print("\nPredicted Website Events:")
    for event, prob in zip(website_events, website_probs[0][np.argsort(website_probs[0])[-3:][::-1]]):
        print(f"Event {event}: {prob*100:.2f}% probability")
    
    print("\nPredicted Server Events:")
    for event, prob in zip(server_events, server_probs[0][np.argsort(server_probs[0])[-3:][::-1]]):
        print(f"Event {event}: {prob*100:.2f}% probability")