import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,precision_score,recall_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib 

def perform_descriptive_analysis(url : str):
    """
    Perform descriptive analysis on the provided DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    """
    df=pd.read_csv(url)

    print("loaded data sucessfully ")
    print("DataFrame Size:", df.size)  # Total number of elements in the DataFrame
    print("DataFrame Shape:", df.shape)  # Dimensions of the DataFrame (rows, columns)
    
    print("DataFrame Info:")
    print(df.info())  # Information about data types and non-null values
    
    print("\nSummary Statistics for Numerical Columns:")
    print(df.describe())  # Summary statistics for numerical columns
    
    print("\nMissing Values per Column:")
    print(df.isnull().sum()) 
    return df


def perform_eda(df: pd.DataFrame):
    """
    Perform exploratory data analysis on the provided DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    """
    # Dropping the first column which contains only row numbers
    df = df.drop(columns=['Unnamed: 0'])

    # Dropping null values in the 'Phising' column
    df = df.dropna()

    print("Missing values per column after dropping nulls:")
    print(df.isnull().sum())

    # Handling columns with high Standard Deviation
    plt.figure(figsize=(10, 6))
    sns.histplot(df['UrlLength'], bins=30, kde=True)
    plt.title("Distribution of URL Length")
    plt.xlabel("UrlLength")
    plt.ylabel("Frequency")
    plt.show()
    print("Displayed distribution of URL Length.")

    # Applying log transformation
    df['LogUrlLength'] = np.log1p(df['UrlLength'])

    # Plot the transformed distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['LogUrlLength'], bins=30, kde=True)
    plt.title("Distribution of Log-Transformed URL Length")
    plt.xlabel("LogUrlLength")
    plt.ylabel("Frequency")
    plt.show()
    print("Displayed distribution of log-transformed URL Length.")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Distribution of Path Length (Histogram)
    sns.histplot(df['PathLength'], bins=30, kde=True, ax=axes[0])
    axes[0].set_title("Distribution of Path Length")
    axes[0].set_xlabel("PathLength")
    axes[0].set_ylabel("Frequency")

    # Plot 2: Box Plot of Path Length
    sns.boxplot(x=df['PathLength'], ax=axes[1])
    axes[1].set_title("Box Plot of Path Length")

    plt.tight_layout()  
    plt.show()
    print("Displayed distribution and box plot of Path Length.")

    # Drop rows where PathLength is greater than 500
    df = df[df['PathLength'] <= 500]

    # Check the new shape of the DataFrame
    print(f"Original DataFrame shape: {df.shape}")
    print(f"DataFrame shape after dropping outliers: {df.shape}")

    # Plot the new box plot for PathLength after removing outliers
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['PathLength'])
    plt.title("Box Plot of Path Length After Dropping Outliers (>500)")
    plt.show()
    print("Displayed box plot of Path Length after dropping outliers.")

    # Apply log transformation for the column
    df['LogPathLength'] = np.log1p(df['PathLength'])

    # Plot the transformed distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['LogPathLength'], bins=30, kde=True)
    plt.title("Log-Transformed Distribution of Path Length")
    plt.xlabel("LogPathLength")
    plt.ylabel("Frequency")
    plt.show()
    print("Displayed log-transformed distribution of Path Length.")

    # Remove rows where UrlLength is greater than 1000
    df = df[df['UrlLength'] <= 600]

    # Scatter plot between two numerical variables (UrlLength vs PathLength)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='UrlLength', y='PathLength', hue='Phising', data=df)
    plt.title("Scatter Plot: URL Length vs Path Length")
    plt.xlabel("UrlLength")
    plt.ylabel("PathLength")
    plt.show()
    print("Displayed scatter plot between URL Length and Path Length.")

    # Correlation matrix to check the relationships between features
    corr_matrix = df.corr()

    # Heatmap of the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Matrix of Features")
    plt.show()
    print("Displayed heatmap of the correlation matrix.")

    return df



def apply_machine_learning(df: pd.DataFrame):
    """
    Apply various machine learning algorithms to the given DataFrame and evaluate their performance.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the features and target variable.
    """
    model_results = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': []
    }

    # Features and target variable
    X = df.drop(columns=['Phising'])  # target variable 
    y = df['Phising']

    # Split the data into train and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=500, class_weight='balanced')
    log_reg.fit(X_train_scaled, y_train)
    y_pred_log_reg = log_reg.predict(X_test_scaled)
    model_results['Model'].append('Logistic Regression')
    model_results['Accuracy'].append(accuracy_score(y_test, y_pred_log_reg))
    model_results['Precision'].append(precision_score(y_test, y_pred_log_reg))
    model_results['Recall'].append(recall_score(y_test, y_pred_log_reg))
    print(f"Logistic Regression - Accuracy: {model_results['Accuracy'][-1]}, Precision: {model_results['Precision'][-1]}, Recall: {model_results['Recall'][-1]}")

    # K-Nearest Neighbors
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    y_pred_knn = knn_model.predict(X_test_scaled)
    model_results['Model'].append('K-Nearest Neighbors')
    model_results['Accuracy'].append(accuracy_score(y_test, y_pred_knn))
    model_results['Precision'].append(precision_score(y_test, y_pred_knn))
    model_results['Recall'].append(recall_score(y_test, y_pred_knn))
    print(f"K-Nearest Neighbors - Accuracy: {model_results['Accuracy'][-1]}, Precision: {model_results['Precision'][-1]}, Recall: {model_results['Recall'][-1]}")

    # Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)
    y_pred_nb = nb_model.predict(X_test_scaled)
    model_results['Model'].append('Naive Bayes')
    model_results['Accuracy'].append(accuracy_score(y_test, y_pred_nb))
    model_results['Precision'].append(precision_score(y_test, y_pred_nb))
    model_results['Recall'].append(recall_score(y_test, y_pred_nb))
    print(f"Naive Bayes - Accuracy: {model_results['Accuracy'][-1]}, Precision: {model_results['Precision'][-1]}, Recall: {model_results['Recall'][-1]}")

    # Decision Tree
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    model_results['Model'].append('Decision Tree')
    model_results['Accuracy'].append(accuracy_score(y_test, y_pred_dt))
    model_results['Precision'].append(precision_score(y_test, y_pred_dt))
    model_results['Recall'].append(recall_score(y_test, y_pred_dt))
    print(f"Decision Tree - Accuracy: {model_results['Accuracy'][-1]}, Precision: {model_results['Precision'][-1]}, Recall: {model_results['Recall'][-1]}")

    # Random Forest
    rf_model = RandomForestClassifier(class_weight='balanced')
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    model_results['Model'].append('Random Forest')
    model_results['Accuracy'].append(accuracy_score(y_test, y_pred_rf))
    model_results['Precision'].append(precision_score(y_test, y_pred_rf))
    model_results['Recall'].append(recall_score(y_test, y_pred_rf))
    print(f"Random Forest - Accuracy: {model_results['Accuracy'][-1]}, Precision: {model_results['Precision'][-1]}, Recall: {model_results['Recall'][-1]}")

    # Convert results dictionary to a DataFrame
    results_df = pd.DataFrame(model_results)

    # Display the results
    print(results_df)

    # Plot a scatter plot between Accuracy and Precision for each model
    plt.figure(figsize=(8, 6))
    plt.scatter(results_df['Accuracy'], results_df['Precision'], color='blue', s=100)

    # Annotate each point with the model name
    for i, model in enumerate(results_df['Model']):
        plt.text(results_df['Accuracy'][i] + 0.001, results_df['Precision'][i], model, fontsize=10)

    plt.title('Scatter Plot: Accuracy vs Precision')
    plt.xlabel('Accuracy')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.show()
    print("Displayed scatter plot between Accuracy and Precision.")


def evaluate_best_model(df: pd.DataFrame):
    """
    Evaluate the Random Forest model on the given DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing features and target variable.
    """
    # Features and target variable
    X = df.drop(columns=['Phising'])  # Target variable
    y = df['Phising']

    # Split the data into train and test sets (70% train, 30% test)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Initialize and train the Random Forest model
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    # Predictions on the training set
    y_train_pred = rf_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    # Predictions on the test set
    y_test_pred = rf_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Print train and test accuracies
    print(f"Random Forest Train Accuracy: {train_accuracy:.4f}")
    print(f"Random Forest Test Accuracy: {test_accuracy:.4f}")

    # Print classification report for test set
    print("Test Set Classification Report:\n", classification_report(y_test, y_test_pred))

    # Save the trained model to a file
    joblib.dump(rf_model, 'phishing_url_model.pkl')
    print("Model saved successfully!")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    cm_normalized = cm.astype('float') / cm.sum() * 100  # Normalize

    # Plot the normalized confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=rf_model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix - Random Forest (in %)')
    plt.show()

    # ROC Curve
    y_pred_prob = rf_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Random Forest')
    plt.legend(loc='lower right')
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)

    plt.figure()
    plt.plot(recall, precision, color='b', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Random Forest')
    plt.show()

    # Feature Importance
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance - Random Forest")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

