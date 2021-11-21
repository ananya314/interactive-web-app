from numpy.lib.function_base import average
from sklearn import neighbors
import streamlit as st
from sklearn import datasets
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, confusion_matrix 

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# can make train test split user interactive
# can make accuracy_score user interactive

st.write("""
# Exploring Different Classifiers and Datasets
""")
st.write("""This is an interactive project to explore the accuracy of popular machine learning models using classic datasets. To begin, choose the **dataset, model, and model parameters**. Have fun!""")
dataset_name = st.sidebar.selectbox("Select Dataset", ("Wine Dataset", "Iris Dataset", "Breast Cancer Dataset"))
st.write("""
#### Dataset 
_____________
""")
st.write("""**Name:**""", dataset_name) #streamlit uses intelligent chaching 
EDA = st.sidebar.checkbox('Show EDA')


def get_dataset(dataset_name):

    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name ==  "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    X = data.data
    col_names = data.feature_names
    y = data.target
    return X, y, col_names

X, y, col_names = get_dataset(dataset_name)
df = pd.DataFrame(X, columns = col_names)


if EDA:
    X, y, col_names = get_dataset(dataset_name)
    df = pd.DataFrame(X, columns = col_names)
    st.write("""**Shape of dataset:**""", X.shape)
    st.write("**Number of classes:**", len(np.unique(y)))
    
    plot_PCA = st.sidebar.checkbox('Show Plot of Principal Components')
    # st.write(col_names)
    st.line_chart(data=df, use_container_width=700)


    if(plot_PCA):
        pca = PCA(2)
        X_projected = pca.fit_transform(X)

        x1 = X_projected[:, 0]
        x2 = X_projected[:, 1]
        d = {"Principal Component 1": x1, "Principal Component 2": x2}
        df_pca = pd.DataFrame(data=d)

        fig = px.scatter(df_pca, x="Principal Component 1", y = "Principal Component 2", color=y, color_continuous_scale=px.colors.sequential.Viridis)
        st.write("""##### Principal Components""")
        st.write("The principal Components after the dataset undergoes principal component analysis (PCA) for feature reduction are shown below: ")
        st.plotly_chart(fig)  

classifier_name = st.sidebar.selectbox("Select Classifier", ("K-Nearest Neighbors", "Support Vector Machine", "Decision Tree", "Random Forest", "Logistic Regression"))

def parameter_sidebar(clf_name):
    params = dict()
    if clf_name == "K-Nearest Neighbors":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "Support Vector Machine":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    elif clf_name == "Random Forest":
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    elif clf_name == "Decision Tree":
        criterion = st.sidebar.radio("criterion", ("Gini", "Entropy"))
        if criterion == "Gini":
            params["criterion"] = "gini"
        else:
            params["criterion"] = "entropy"

        splitter = st.sidebar.radio("Splitter", ("Best", "Random"))
        if splitter == "Best":
            params["splitter"] = "best"
        else:
            params["splitter"] = "random"
        max_depth_dt = st.sidebar.slider("max_depth", 2, 15)
        params["max_depth_dt"] = max_depth_dt
    
    else:
        C_log = st.sidebar.slider("C", 0.01, 10.0)
        params["C_log"] = C_log
        
    return params

params = parameter_sidebar(classifier_name)

def get_classifier(clf_name, params):

    if clf_name == "K-Nearest Neighbors":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    
    elif clf_name == "Support Vector Machine":
        clf = SVC(C=params["C"])

    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=42)

    elif clf_name == "Decision Tree":
        clf = DecisionTreeClassifier(max_depth=params["max_depth_dt"], splitter=params["splitter"], criterion=params["criterion"])

    else:
        clf = LogisticRegression(C=params["C_log"])
    return clf

clf = get_classifier(classifier_name, params)

########## CLASSIFICATION ###################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write("""
#### Classifier 
_____________
""")
st.write(f"**Classifier Name:** {classifier_name}")
st.write(f"**Accuracy:** {acc}")

list_metrics = st.sidebar.checkbox("Show Other Metrics")
if list_metrics:
    p = st.sidebar.checkbox("Precision")
    r = st.sidebar.checkbox("Recall")
    f1 = st.sidebar.checkbox("F1 Score")
    m = st.sidebar.checkbox("MSE")
    ma = st.sidebar.checkbox("MAE")
    conf_matrix = st.sidebar.checkbox("Plot Confusion Matrix")

    if p:
        acc = precision_score(y_test, y_pred, average=None)
        st.write(f"**Precision:**")
        for i in range(len(np.unique(y))):
            st.write(f"Class {i+1}: {acc[i]}")
    if r:
        acc = recall_score(y_test, y_pred, average=None)
        st.write(f"**Recall:**")
        for i in range(len(np.unique(y))):
            st.write(f"Class {i+1}: {acc[i]}")
    if f1:
        acc = f1_score(y_test, y_pred, average=None)
        st.write(f"**F1-Score:**")
        for i in range(len(np.unique(y))):
            st.write(f"Class {i+1}: {acc[i]}")
    if m:
        acc = mean_squared_error(y_test, y_pred)
        st.write(f"**MSE:** {acc}")
    if ma:
        acc = mean_absolute_error(y_test, y_pred)
        st.write(f"**MAE:** {acc}")
    if conf_matrix:
        conf = confusion_matrix(y_test, y_pred)
        lst = []
        for i in range(len(np.unique(y))):
            lst.append(f"Class {i+1}")
        st.write("**Confusion Matrix Table:**")
        st.write(conf)

        st.write("**Confusion Matrix Plot:**")
        fig = px.imshow(conf, x=lst, y=lst, color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig)

st.sidebar.title('Contact:')
st.sidebar.write("Ananya Devarakonda")
st.sidebar.markdown('[https://ananya314.github.io/contact.html](https://ananya314.github.io/contact.html)')

