import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class SignLanguageModel:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
    
    def load_data(self):
        """Load processed features"""
        X = np.load('data/processed/X.npy', allow_pickle=True)
        y = np.load('data/processed/y.npy', allow_pickle=True)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded)
        
        # Pad sequences to same length
        max_len = max([x.shape[0] for x in X])
        X_padded = np.array([
            np.pad(x, ((0, max_len - x.shape[0]), (0, 0)), mode='constant')
            for x in X
        ])
        
        return train_test_split(X_padded, y_categorical, test_size=0.2, random_state=42)
    
    def build_model(self, input_shape, num_classes):
        """Build LSTM model for sequence classification"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, epochs=50):
        """Train the model"""
        X_train, X_test, y_train, y_test = self.load_data()
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Number of classes: {y_train.shape[1]}")
        
        self.model = self.build_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            num_classes=y_train.shape[1]
        )
        
        print("\nModel Summary:")
        self.model.summary()
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        # Save model
        self.model.save('models/sign_classifier.h5')
        np.save('models/label_encoder.npy', self.label_encoder.classes_)
        
        # Plot training history
        self.plot_history(history)
        
        return history
    
    def plot_history(self, history):
        """Plot training metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(history.history['accuracy'], label='Train')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        ax2.plot(history.history['loss'], label='Train')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('models/training_history.png')
        print("✓ Training history saved to models/training_history.png")

# Usage
if __name__ == "__main__":
    trainer = SignLanguageModel()
    history = trainer.train(epochs=50)