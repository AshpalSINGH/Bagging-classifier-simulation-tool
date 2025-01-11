import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier  # Added for KNN
from sklearn.datasets import make_classification
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # sample data
    st.set_option('deprecation.showPyplotGlobalUse', False)
    df=pd.read_csv("Social_Network_Ads.csv")
    x= df.iloc[:, 2:4]
    y= df.iloc[:,-1]

    x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)
    scalar= StandardScaler()
    x_train= scalar.fit_transform(x_train)
    x_test = scalar.transform(x_test)

    # Streamlit app
    with st.sidebar:
        st.title("Bagging Classifier")
        base_estimator_type = st.selectbox("Select base estimator", ["Decision Tree", "SVM", "KNN"])
        num_of_est = st.number_input("Enter the number of estimators", 0, 10000)
        samples = st.slider("Max samples", 0, x_train.shape[0])
        boot_samples = st.radio("Bootstrap Samples", ["True", "False"])
        features = st.slider("Max features", 0, x_train.shape[1])

        boot_features = st.radio("Bootstrap features", ["True", "False"])
        button = st.button("Run Algorithm")

    base_estimator = None

    # Create base estimator outside the button block
    if base_estimator_type == "Decision Tree":
        base_estimator = DecisionTreeClassifier()
    elif base_estimator_type == "SVM":
        base_estimator = SVC()
    elif base_estimator_type == "KNN":  # Added for KNN
        base_estimator = KNeighborsClassifier()
    else:
        raise ValueError("Unsupported base estimator type:", base_estimator_type)
    
    st.subheader(base_estimator_type)
    # Check if the user has clicked the "Run Algorithm" button
    if button:
        col1, col2 = st.columns(2)

        with col1:
            # Train the base estimator
            base_estimator.fit(x_train, y_train)
            y_pred= base_estimator.predict(x_test)
            # Plot Decision Boundary for Bagging Classifier
            a=np.arange(start= x_train[:, 0].min()-1, stop= x_train[:, 0].max()+1,step=0.01)
            b=np.arange(start= x_train[:, 1].min()-1, stop= x_train[:, 1].max()+1,step=0.01)
            xx, yy = np.meshgrid(a,b)
            input_array = np.array([xx.ravel(),yy.ravel()]).T
            labels= base_estimator.predict(input_array)
            custom_cmap = plt.cm.get_cmap('rainbow')
            plt.contourf(xx,yy,labels.reshape(xx.shape),cmap= custom_cmap,alpha=0.4)
            plt.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap=custom_cmap,alpha=0.5)
            plt.title("Decision Boundary - Without Bagging Classifier")
            st.pyplot()
            st.write("Accuracy:", accuracy_score(y_test, y_pred))
            # Add conditions for plotting decision boundary for other base estimators
            # (SVM, KNN, etc.) if needed

        with col2:
            # Train a Bagging Classifier
            bag = BaggingClassifier(
                base_estimator=base_estimator,
                n_estimators=num_of_est,
                max_samples=samples,
                max_features=features,
                bootstrap=boot_samples,
                bootstrap_features=boot_features,
            )
            bag.fit(x_train, y_train)
            y_pred_bag = bag.predict(x_test)

            # Plot Decision Boundary for Bagging Classifier
            a=np.arange(start= x_train[:, 0].min()-1, stop= x_train[:, 0].max()+1,step=0.01)
            b=np.arange(start= x_train[:, 1].min()-1, stop= x_train[:, 1].max()+1,step=0.01)
            xx, yy = np.meshgrid(a,b)
            input_array = np.array([xx.ravel(),yy.ravel()]).T
            labels= bag.predict(input_array)
            custom_cmap = plt.cm.get_cmap('rainbow')
            plt.contourf(xx,yy,labels.reshape(xx.shape),cmap= custom_cmap,alpha=0.4)
            plt.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap=custom_cmap,alpha=0.5)
            plt.title("Decision Boundary - With Bagging Classifier")
            st.pyplot()
            # Display accuracy
            st.write("Accuracy:", accuracy_score(y_test, y_pred_bag))
    st.subheader("A sample of data")
    st.dataframe(df.sample(5))