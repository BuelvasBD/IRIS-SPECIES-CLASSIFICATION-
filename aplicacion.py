import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import plotly.express as px
import plotly.graph_objects as go

from streamlit_option_menu import option_menu

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Iris Classification",
    page_icon="",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 1rem;
}

div[data-testid="stMetric"] {
    background-color: #1c1f26;
    border: 1px solid #2d333b;
    padding: 15px;
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target
df["species_name"] = df["species"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

# ------------------------------------------------
# MODEL
# ------------------------------------------------
X = df[iris.feature_names]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

# ------------------------------------------------
# TITLE
# ------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'> Iris Species Classification</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;color:gray;'>Interactive Machine Learning Dashboard</p>",
    unsafe_allow_html=True
)

# ------------------------------------------------
# TOP NAVIGATION MENU
# ------------------------------------------------
selected = option_menu(
    menu_title=None,
    options=[
        "Dashboard",
        "Prediction",
        "3D Plot",
        "Histograms",
        "Scatter Matrix",
        "Feature Importance",
        "Dataset"
    ],
    icons=[
        "bar-chart",
        "cpu",
        "globe",
        "graph-up",
        "grid",
        "star",
        "table"
    ],
    orientation="horizontal"
)

# ------------------------------------------------
# DASHBOARD PAGE
# ------------------------------------------------
if selected == "Dashboard":

    st.subheader(" Model Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy:.2f}")
    col2.metric("Precision", f"{precision:.2f}")
    col3.metric("Recall", f"{recall:.2f}")
    col4.metric("F1 Score", f"{f1:.2f}")

    st.divider()

    st.subheader(" Species Distribution")

    species_count = df["species_name"].value_counts().reset_index()
    species_count.columns = ["Species", "Count"]

    fig = px.pie(
        species_count,
        names="Species",
        values="Count",
        hole=0.5
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# PREDICTION PAGE
# ------------------------------------------------
elif selected == "Prediction":

    st.subheader(" Predict Flower Species")

    col1, col2 = st.columns(2)

    with col1:

        sepal_length = st.slider(
            "Sepal Length",
            float(df["sepal length (cm)"].min()),
            float(df["sepal length (cm)"].max()),
            5.4
        )

        sepal_width = st.slider(
            "Sepal Width",
            float(df["sepal width (cm)"].min()),
            float(df["sepal width (cm)"].max()),
            3.4
        )

    with col2:

        petal_length = st.slider(
            "Petal Length",
            float(df["petal length (cm)"].min()),
            float(df["petal length (cm)"].max()),
            1.3
        )

        petal_width = st.slider(
            "Petal Width",
            float(df["petal width (cm)"].min()),
            float(df["petal width (cm)"].max()),
            0.2
        )

    input_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = model.predict(input_data)[0]

    species_names = {
        0: " Setosa",
        1: " Versicolor",
        2: "Virginica"
    }

    st.success(f"Predicted Species: {species_names[prediction]}")

# ------------------------------------------------
# 3D PLOT PAGE
# ------------------------------------------------
elif selected == "3D Plot":

    st.subheader(" 3D Scatter Plot")

    fig_3d = px.scatter_3d(
        df,
        x="sepal length (cm)",
        y="sepal width (cm)",
        z="petal length (cm)",
        color="species_name",
        opacity=0.7
    )

    st.plotly_chart(fig_3d, use_container_width=True)

# ------------------------------------------------
# HISTOGRAM PAGE
# ------------------------------------------------
elif selected == "Histograms":

    st.subheader(" Feature Histograms")

    feature = st.selectbox(
        "Select Feature",
        iris.feature_names
    )

    fig = px.histogram(
        df,
        x=feature,
        color="species_name",
        marginal="box"
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# SCATTER MATRIX PAGE
# ------------------------------------------------
elif selected == "Scatter Matrix":

    st.subheader(" Scatter Matrix")

    fig = px.scatter_matrix(
        df,
        dimensions=iris.feature_names,
        color="species_name"
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# FEATURE IMPORTANCE PAGE
# ------------------------------------------------
elif selected == "Feature Importance":

    st.subheader(" Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": iris.feature_names,
        "Importance": model.feature_importances_
    })

    fig = px.bar(
        importance_df,
        x="Feature",
        y="Importance"
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# DATASET PAGE
# ------------------------------------------------
elif selected == "Dataset":

    st.subheader(" Iris Dataset")

    st.dataframe(df, use_container_width=True)

# ------------------------------------------------
# FOOTER
# ------------------------------------------------
st.divider()

st.markdown(
    "<p style='text-align:center;color:gray;'>Developed for Data Mining - Universidad de la Costa</p>",
    unsafe_allow_html=True
)
