import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Iris Species Classification",
    page_icon="🌸",
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
    padding-top: 2rem;
}

div[data-testid="stMetric"] {
    background-color: #1c1f26;
    border: 1px solid #2d333b;
    padding: 15px;
    border-radius: 15px;
}

h1, h2, h3 {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD DATASET
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
# TITLE
# ------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🌸 Iris Species Classification Dashboard</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;color:gray;'>Machine Learning Classification using Random Forest</p>",
    unsafe_allow_html=True
)

st.divider()

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
st.sidebar.title("🌿 Flower Measurements")

sepal_length = st.sidebar.slider(
    "Sepal Length",
    float(df["sepal length (cm)"].min()),
    float(df["sepal length (cm)"].max()),
    5.4
)

sepal_width = st.sidebar.slider(
    "Sepal Width",
    float(df["sepal width (cm)"].min()),
    float(df["sepal width (cm)"].max()),
    3.4
)

petal_length = st.sidebar.slider(
    "Petal Length",
    float(df["petal length (cm)"].min()),
    float(df["petal length (cm)"].max()),
    1.3
)

petal_width = st.sidebar.slider(
    "Petal Width",
    float(df["petal width (cm)"].min()),
    float(df["petal width (cm)"].max()),
    0.2
)

# ------------------------------------------------
# MODEL TRAINING
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

# ------------------------------------------------
# MODEL EVALUATION
# ------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

# ------------------------------------------------
# KPI METRICS
# ------------------------------------------------
st.subheader("📊 Model Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy:.2f}")
col2.metric("Precision", f"{precision:.2f}")
col3.metric("Recall", f"{recall:.2f}")
col4.metric("F1 Score", f"{f1:.2f}")

st.divider()

# ------------------------------------------------
# USER INPUT PREDICTION
# ------------------------------------------------
input_data = np.array([[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]])

prediction = model.predict(input_data)[0]

species_names = {
    0: "🌱 Setosa",
    1: "🌿 Versicolor",
    2: "🌸 Virginica"
}

predicted_species = species_names[prediction]

# ------------------------------------------------
# PREDICTION DISPLAY
# ------------------------------------------------
st.subheader("🔮 Prediction")

st.success(f"The predicted flower species is: {predicted_species}")

st.divider()

# ------------------------------------------------
# 3D SCATTER PLOT
# ------------------------------------------------
st.subheader("🌐 3D Scatter Visualization")

fig_3d = px.scatter_3d(
    df,
    x="sepal length (cm)",
    y="sepal width (cm)",
    z="petal length (cm)",
    color="species_name",
    opacity=0.7,
    title="3D Distribution of Iris Species"
)

# Add new sample
fig_3d.add_trace(
    go.Scatter3d(
        x=[sepal_length],
        y=[sepal_width],
        z=[petal_length],
        mode="markers",
        marker=dict(
            size=10,
            symbol="diamond"
        ),
        name="New Sample"
    )
)

st.plotly_chart(fig_3d, use_container_width=True)

st.divider()

# ------------------------------------------------
# HISTOGRAMS
# ------------------------------------------------
st.subheader("📈 Feature Distributions")

feature = st.selectbox(
    "Select Feature",
    iris.feature_names
)

hist_fig = px.histogram(
    df,
    x=feature,
    color="species_name",
    marginal="box",
    title=f"Distribution of {feature}"
)

st.plotly_chart(hist_fig, use_container_width=True)

st.divider()

# ------------------------------------------------
# SCATTER MATRIX
# ------------------------------------------------
st.subheader("🔍 Scatter Matrix")

scatter_matrix = px.scatter_matrix(
    df,
    dimensions=iris.feature_names,
    color="species_name",
    title="Scatter Matrix of Iris Features"
)

st.plotly_chart(scatter_matrix, use_container_width=True)

st.divider()

# ------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------
st.subheader("⭐ Feature Importance")

importance_df = pd.DataFrame({
    "Feature": iris.feature_names,
    "Importance": model.feature_importances_
})

importance_fig = px.bar(
    importance_df,
    x="Feature",
    y="Importance",
    title="Feature Importance"
)

st.plotly_chart(importance_fig, use_container_width=True)

st.divider()

# ------------------------------------------------
# DATASET PREVIEW
# ------------------------------------------------
st.subheader("📄 Dataset Preview")

st.dataframe(df, use_container_width=True)

# ------------------------------------------------
# FOOTER
# ------------------------------------------------
st.markdown(
    "<p style='text-align:center;color:gray;'>Developed for Data Mining - Universidad de la Costa</p>",
    unsafe_allow_html=True
)