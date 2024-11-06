import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam

import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load concatenated data from CSV file
data = pd.read_csv('pcos_csv.csv', low_memory=False) 

# Separate numerical and categorical features
numerical_features = data.select_dtypes(include=['float64', 'int64'])
categorical_features = data.select_dtypes(include=['object'])

# Preprocess numerical features
if not numerical_features.empty:
    scaler = StandardScaler()
    scaled_numerical_features = scaler.fit_transform(numerical_features)
    scaled_numerical_features = pd.DataFrame(scaled_numerical_features, columns=numerical_features.columns)
else:
    scaled_numerical_features = pd.DataFrame()

# Preprocess categorical features
if not categorical_features.empty:
    encoder = LabelEncoder()
    encoded_categorical_features = categorical_features.apply(encoder.fit_transform)
else:
    encoded_categorical_features = pd.DataFrame()

# Concatenate preprocessed features
preprocessed_data = pd.concat([scaled_numerical_features, encoded_categorical_features], axis=1)

# Split data into training and testing sets
X_train, X_test = train_test_split(preprocessed_data, test_size=0.2, random_state=42)

# Define and compile discriminator model
def build_discriminator(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Define and compile generator model
def build_generator(latent_dim, output_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(output_dim, activation='sigmoid'))
    return model

# Combine generator and discriminator into a GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

# Train GAN model
def train_gan(generator, discriminator, gan, X_train, epochs=100, batch_size=128, latent_dim=100):
    for epoch in range(epochs):
        # Generate fake samples
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_samples = generator.predict(noise)
        
        # Combine real and fake samples
        X_combined = np.concatenate([X_train, fake_samples])
        y_combined = np.concatenate([np.ones((len(X_train), 1)), np.zeros((batch_size, 1))])
        
        # Train discriminator
        d_loss = discriminator.train_on_batch(X_combined, y_combined)
        
        # Train generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        y_gen = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, y_gen)
        
        # Print progress
        print(f'Epoch: {epoch+1}, Discriminator Loss: {d_loss[0]}, Generator Loss: {g_loss}')

# Define parameters
input_dim = X_train.shape[1]
latent_dim = 100
gan_epochs = 100
gan_batch_size = 128

# Build and compile models
discriminator = build_discriminator(input_dim)
generator = build_generator(latent_dim, input_dim)
gan = build_gan(generator, discriminator)

# Train GAN
train_gan(generator, discriminator, gan, X_train, epochs=gan_epochs, batch_size=gan_batch_size)

# Generate synthetic data
synthetic_data = generator.predict(np.random.normal(0, 1, (len(X_test), latent_dim)))

# Calculate regression metrics
mse = mean_squared_error(X_test, synthetic_data)
mae = mean_absolute_error(X_test, synthetic_data)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

# Calculate classification metrics (assuming X_test and synthetic_data are compatible for classification metrics)
precision = precision_score(np.argmax(X_test.values, axis=1), np.argmax(synthetic_data, axis=1), average='weighted', zero_division=1)
recall = recall_score(np.argmax(X_test.values, axis=1), np.argmax(synthetic_data, axis=1), average='weighted', zero_division=0)
f1 = f1_score(np.argmax(X_test.values, axis=1), np.argmax(synthetic_data, axis=1), average='weighted', zero_division=0)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Plot metrics
plt.figure(figsize=(10, 6))

# Plot regression metrics
plt.subplot(2, 2, 1)
plt.bar(['MSE', 'MAE'], [mse, mae])
plt.xlabel('Metric')
plt.ylabel('Error')
plt.title('Regression Metrics')

# Plot classification metrics
plt.subplot(2, 2, 2)
plt.bar(['Precision', 'Recall', 'F1 Score'], [precision, recall, f1])
plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('Classification Metrics')

plt.tight_layout()
plt.show()