import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
def load_data():
    # URL for the Heart Disease UCI dataset
    url = "https://gist.githubusercontent.com/vivek2606/88ec0800798ee7bf8540193acde83553/raw/"
    df = pd.read_csv(url)
    return df

def perform_eda(df):
    print("\nExploratory Data Analysis:")
    print("\nDataset Info:")
    df.info()
    print("\nDescriptive Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nTarget Variable Distribution:")
    print(df['heart disease'].value_counts())

    # Visualize the distribution of the target variable
    plt.figure(figsize=(6, 4))
    sns.countplot(x='heart disease', data=df)
    plt.title('Distribution of Heart Disease')
    plt.savefig('heart_disease_distribution.png')
    plt.close()

    # Visualize relationships between a few key features
    print("\nGenerating pairplot for EDA...")
    sns.pairplot(df[['age', 'BP', 'cholestrol', 'max heart rate', 'heart disease']], hue='heart disease')
    plt.savefig('eda_pairplot.png')
    plt.close()
    print("EDA visualizations saved to 'heart_disease_distribution.png' and 'eda_pairplot.png'")

def preprocess_data(df):
    print("\nPreprocessing Data...")
    # The target variable 'heart disease' has values 1 (no disease) and 2 (disease).
    # We will map it to 0 (no disease) and 1 (disease) for consistency.
    df['heart disease'] = df['heart disease'].replace({1: 0, 2: 1})

    # Separate features and target
    X = df.drop('heart disease', axis=1)
    y = df['heart disease']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Data preprocessing complete.")
    return X_train, X_test, y_train, y_test

def plot_roc_curve(y_test, y_pred_proba, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    print("\nTraining and Evaluating Models...")
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        print(f"\n--- {name} ---")
        # Train the model
        model.fit(X_train, y_train)
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Evaluate the model
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print(f"Precision: {precision_score(y_test, y_pred):.2f}")
        print(f"Recall: {recall_score(y_test, y_pred):.2f}")
        print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.2f}")

        # Plot ROC curve
        plot_roc_curve(y_test, y_pred_proba, name)

    # Train and evaluate ANN
    print("\n--- Artificial Neural Network (ANN) ---")
    ann = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    ann.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
    y_pred_proba_ann = ann.predict(X_test).ravel()
    y_pred_ann = (y_pred_proba_ann > 0.5).astype(int)

    print(f"Accuracy: {accuracy_score(y_test, y_pred_ann):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred_ann):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred_ann):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred_ann):.2f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba_ann):.2f}")
    plot_roc_curve(y_test, y_pred_proba_ann, "ANN")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='best')
    plt.savefig('roc_curves.png')
    plt.close()
    print("\nROC curves plot saved to 'roc_curves.png'")

if __name__ == "__main__":
    print("Disease Prediction System")
    # Load the data
    df = load_data()
    print("Dataset loaded successfully:")
    print(df.head())

    # Perform EDA
    perform_eda(df)

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(df.copy()) # Use a copy to avoid changing the original df

    # Train and evaluate ML models
    train_and_evaluate_models(X_train, X_test, y_train, y_test)
