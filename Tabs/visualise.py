import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from web_functions import train_model  # Sesuaikan dengan nama file dan fungsi yang sesuai

# Fungsi untuk membuat heatmap confusion matrix
def plot_confusion_matrix_heatmap(y_true, y_pred, title=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Female', 'Male'], yticklabels=['Female', 'Male'])

    # Set the title if provided
    if title:
        plt.title(title)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot()

# Fungsi untuk membuat scatter plot K-Nearest Neighbors
def plot_kneighbors_scatter(model, x, features, y_test):
    neighbors = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(x)
    distances, indices = neighbors.kneighbors(x)

    plt.figure(figsize=(10, 6))

    if len(features) == 1:
        plt.scatter(x[features[0]], x[features[0]], marker='o', s=50, label='Data Points')  # Gunakan fitur yang sama untuk kedua sumbu
        for i in range(len(x)):
            label = np.bincount(y_test[indices[i]]).argmax()
            color = 'blue' if label == 1 else 'pink'
            plt.scatter(
                x.iloc[indices[i], x.columns.get_loc(features[0])],
                x.iloc[indices[i], x.columns.get_loc(features[0])],
                marker='x', s=50, color=color, alpha=0.5
            )
        plt.xlabel(f'{features[0]}')
        plt.ylabel(f'{features[0]}')
        plt.title("Visualisasi K-Nearest Neighbors")
        st.pyplot()

    elif len(features) == 2:
        plt.scatter(x[features[0]], x[features[1]], marker='o', s=50, label='Data Points')
        for i in range(len(x)):
            color = 'blue' if np.bincount(y_test[indices[i]]).argmax() == 1 else 'pink'
            plt.scatter(
                x.iloc[indices[i], x.columns.get_loc(features[0])],
                x.iloc[indices[i], x.columns.get_loc(features[1])],
                marker='x', s=50, color=color, alpha=0.5
            )
        plt.xlabel(f'{features[0]}')
        plt.ylabel(f'{features[1]}')
        plt.title("Visualisasi K-Nearest Neighbors")
        st.pyplot()

    else:
        st.warning("Pilih satu atau dua fitur untuk visualisasi.")

# Fungsi untuk membuat ROC Curve
def plot_roc_curve(model, x, y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    y_probs = model.predict_proba(x)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_encoded, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    st.pyplot()

# Fungsi untuk membuat bar plot akurasi
def plot_accuracy(model, x_train, y_train, x_test, y_test):
    train_accuracy = model.score(x_train, y_train) * 100
    test_accuracy = model.score(x_test, y_test) * 100

    plt.figure(figsize=(10, 6))
    plt.bar(['Training Accuracy', 'Testing Accuracy'], [train_accuracy, test_accuracy], color=['skyblue', 'lightgreen'])
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    st.pyplot()

    st.write(f"Training Accuracy: {train_accuracy:.2f}%")
    st.write(f"Testing Accuracy: {test_accuracy:.2f}%")

# Fungsi untuk membuat plot akurasi dan grafik lainnya untuk k-NN
def plot_knn_accuracy(x_train, y_train, x_test, y_test, max_neighbors):
    neighbors = range(1, max_neighbors + 1)
    train_accuracy = []
    test_accuracy = []

    for neighbor in neighbors:
        model = KNeighborsClassifier(n_neighbors=neighbor)
        model.fit(x_train, y_train)
        train_accuracy.append(model.score(x_train, y_train) * 100)
        test_accuracy.append(model.score(x_test, y_test) * 100)

    # Plot k-NN Accuracy
    plt.figure(figsize=(10, 6))
    plt.title('k-NN Varying number of neighbors')
    plt.plot(neighbors, test_accuracy, label='Testing Accuracy', marker='o', linestyle='-', color='orange')
    plt.plot(neighbors, train_accuracy, label='Training accuracy', marker='o', linestyle='-', color='lightblue')
    plt.legend()
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    st.pyplot()

    # Print the accuracy values
    for neighbor, test_acc, train_acc in zip(neighbors, test_accuracy, train_accuracy):
        st.write(f"Neighbors: {neighbor}, Testing Accuracy: {test_acc:.2f}%, Training Accuracy: {train_acc:.2f}%")

    # Plot Confusion Matrix for k-NN with max_neighbors
    model_k = KNeighborsClassifier(n_neighbors=max_neighbors)
    model_k.fit(x_train, y_train)
    y_pred_k = model_k.predict(x_test)

    # Set the title for Confusion Matrix
    conf_matrix_title = f'KNN Classifier Confusion Matrix (Neighbors={max_neighbors})'

    # Plot the confusion matrix
    plot_confusion_matrix_heatmap(y_test, y_pred_k, title=conf_matrix_title)

    # Plot ROC Curve for k-NN with max_neighbors
    model = KNeighborsClassifier(n_neighbors=max_neighbors)
    model.fit(x_train, y_train)
    plot_roc_curve(model, x_test, y_test)

    # Print AUC ROC for k-NN with max_neighbors
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)
    y_probs = model.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test_encoded, y_probs)
    roc_auc = auc(fpr, tpr)
    st.write(f"AUC ROC for k-NN with {max_neighbors} neighbors: {roc_auc:.4f}")

    # Plot Error Rate vs K
    error_rate = [1 - acc / 100 for acc in test_accuracy]
    plt.figure(figsize=(10, 6))
    plt.plot(neighbors, error_rate, marker='o', linestyle='--', color='red')
    plt.title('Error Rate vs K')
    plt.xlabel('Number of neighbors')
    plt.ylabel('Error Rate')
    st.pyplot()

    # Print Error Rate for each K
    for k, error in zip(neighbors, error_rate):
        st.write(f"K = {k}, Error Rate = {error:.4f}")

# Fungsi utama aplikasi
def app(df, x, y):
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("Halaman Visualisasi Prediksi Gender")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

    if st.checkbox("Plot Confusion Matrix"):
        st.write("Menggunakan data yang dihasilkan dari model")
        model, score = train_model(x_train, y_train)
        y_pred = model.predict(x_test)
        plot_confusion_matrix_heatmap(y_test, y_pred)

    if st.checkbox("Plot K-Nearest Neighbors"):
        model, score = train_model(x_train, y_train)
        st.write("Visualisasi K-Nearest Neighbors")
        st.write("Menggunakan data yang dihasilkan dari model")
        feature_options = x.columns.tolist()
        features = st.multiselect('Pilih fitur untuk visualisasi', feature_options, default=[feature_options[0], feature_options[1]])

        st.write("X Biru = Male")
        st.write("X Pink = Female")

        plot_kneighbors_scatter(model, x_test, features, y_test)

    if st.checkbox("Plot ROC Curve"):
        st.write("Menggunakan data yang dihasilkan dari model")
        model, score = train_model(x_train, y_train)
        plot_roc_curve(model, x_test, y_test)

    if st.checkbox("Plot Accuracy Model"):
        st.write("Menggunakan data yang dihasilkan dari model")
        model, score = train_model(x_train, y_train)
        plot_accuracy(model, x_train, y_train, x_test, y_test)

    if st.checkbox("Plot Berdasarkan Input Nilai K untuk k-NN"):
        st.write("Input Nilai K yang Diinginkan")
        max_neighbors = st.slider('Select nilai K of neighbors', 1, 20, 3)
        plot_knn_accuracy(x_train, y_train, x_test, y_test, max_neighbors)
