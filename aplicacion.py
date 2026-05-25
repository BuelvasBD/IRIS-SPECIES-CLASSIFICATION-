import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import plotly.express as px
from streamlit_option_menu import option_menu

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Iris Dashboard",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: white !important;
}

/* ------------------------------------------------ */
/* BACKGROUND */
/* ------------------------------------------------ */

.stApp {
    background-color: #0F1117;
}

/* ------------------------------------------------ */
/* MAIN CONTAINER */
/* ------------------------------------------------ */

.main .block-container {
    padding-top: 1rem;
    max-width: 1400px;
}

/* ------------------------------------------------ */
/* STREAMLIT HEADER */
/* ------------------------------------------------ */

[data-testid="stHeader"] {
    background: transparent;
}

/* ------------------------------------------------ */
/* TITLES */
/* ------------------------------------------------ */

.main-title {
    font-size: 42px;
    font-weight: 600;
    color: white;
    margin-bottom: 5px;
}

.sub-title {
    color: #B8BCC8;
    margin-bottom: 25px;
}

/* ------------------------------------------------ */
/* NAVIGATION */
/* ------------------------------------------------ */

.nav-link {
    border-radius: 10px !important;
    transition: 0.2s !important;
    color: #D1D5DB !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}

.nav-link:hover {
    background-color: #1E293B !important;
}

.nav-link-selected {
    background-color: #6366F1 !important;
    color: white !important;
}

/* ------------------------------------------------ */
/* METRICS */
/* ------------------------------------------------ */

div[data-testid="stMetric"] {

    background: #1A1D29;

    border-radius: 14px;

    padding: 18px;

    border: 1px solid #2A2F3D;
}

/* Metric labels */

div[data-testid="stMetricLabel"] {
    color: #B8BCC8 !important;
}

/* Metric values */

div[data-testid="stMetricValue"] {
    color: white !important;
}

/* ------------------------------------------------ */
/* CHARTS */
/* ------------------------------------------------ */

.js-plotly-plot {

    background: #1A1D29;

    border-radius: 14px;

    padding: 10px;

    border: 1px solid #2A2F3D;
}

/* ------------------------------------------------ */
/* DATAFRAME */
/* ------------------------------------------------ */

[data-testid="stDataFrame"] {

    border-radius: 14px;

    border: 1px solid #2A2F3D;

    overflow: hidden;
}

/* ------------------------------------------------ */
/* SECTION TITLES */
/* ------------------------------------------------ */

h1, h2, h3, h4, h5, h6 {
    color: white !important;
}

/* ------------------------------------------------ */
/* SLIDERS */
/* ------------------------------------------------ */

.stSlider label {
    color: white !important;
}

/* ------------------------------------------------ */
/* SELECTBOX */
/* ------------------------------------------------ */

.stSelectbox label {
    color: white !important;
}

/* ------------------------------------------------ */
/* SUCCESS BOX */
/* ------------------------------------------------ */

.stSuccess {
    background-color: #1E293B !important;
    color: white !important;
    border: 1px solid #374151 !important;
}

/* ------------------------------------------------ */
/* DIVIDER */
/* ------------------------------------------------ */

hr {
    border-color: #2A2F3D;
}

/* ------------------------------------------------ */
/* FOOTER */
/* ------------------------------------------------ */

footer {
    color: #B8BCC8 !important;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.markdown("""
<div class="main-title">
    Iris Species Dashboard
</div>

<div class="sub-title">
    Machine Learning Classification using Random Forest
</div>
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
# MENU
# ------------------------------------------------

selected = option_menu(
    menu_title=None,
    options=[
        "Dashboard",
        "Prediction",
        "3D Plot",
        "Histograms",
        "Feature Importance",
        "Dataset"
    ],
    orientation="horizontal",
    styles={
        "container": {
            "padding": "0!important",
            "background-color": "transparent"
        },
        "icon": {
            "display": "none"
        },
        "nav-link": {
            "text-align": "center",
            "margin": "0px 6px",
        },
        "nav-link-selected": {
            "background-color": "#111827",
        },
    }
)

st.divider()

# ------------------------------------------------
# DASHBOARD
# ------------------------------------------------

if selected == "Dashboard":

    st.subheader("Model Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy:.2f}")
    col2.metric("Precision", f"{precision:.2f}")
    col3.metric("Recall", f"{recall:.2f}")
    col4.metric("F1 Score", f"{f1:.2f}")

    st.divider()

    left, right = st.columns(2)

    with left:

        species_count = df["species_name"].value_counts().reset_index()

        species_count.columns = ["Species", "Count"]

        fig = px.pie(
            species_count,
            names="Species",
            values="Count",
            hole=0.55
        )

        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(
                family="Inter",
                size=14
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        fig_box = px.box(
            df,
            y=iris.feature_names
        )

        fig_box.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(
                family="Inter",
                size=14
            )
        )

        st.plotly_chart(fig_box, use_container_width=True)

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------

elif selected == "Prediction":

    st.subheader("Prediction")

    col1, col2 = st.columns(2)

    with col1:

        sepal_length = st.slider(
            "Sepal Length",
            4.0,
            8.0,
            5.4
        )

        sepal_width = st.slider(
            "Sepal Width",
            2.0,
            4.5,
            3.4
        )

    with col2:

        petal_length = st.slider(
            "Petal Length",
            1.0,
            7.0,
            1.3
        )

        petal_width = st.slider(
            "Petal Width",
            0.1,
            2.5,
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
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    st.success(
        f"Predicted Species: {species_names[prediction]}"
    )

# ------------------------------------------------
# 3D PLOT
# ------------------------------------------------

elif selected == "3D Plot":

    st.subheader("3D Visualization")

    fig_3d = px.scatter_3d(
        df,
        x="sepal length (cm)",
        y="sepal width (cm)",
        z="petal length (cm)",
        color="species_name"
    )

    fig_3d.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(
            family="Inter",
            size=14
        )
    )

    st.plotly_chart(fig_3d, use_container_width=True)

# ------------------------------------------------
# HISTOGRAMS
# ------------------------------------------------

elif selected == "Histograms":

    st.subheader("Feature Histograms")

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

    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(
            family="Inter",
            size=14
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------

elif selected == "Feature Importance":

    st.subheader("Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": iris.feature_names,
        "Importance": model.feature_importances_
    })

    fig = px.bar(
        importance_df,
        x="Feature",
        y="Importance",
        color="Importance",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(
            family="Inter",
            size=14
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# DATASET
# ------------------------------------------------

elif selected == "Dataset":

    st.subheader("Dataset")

    st.dataframe(df, use_container_width=True)

# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.divider()

st.caption(
    "Developed for Data Mining • Universidad de la Costa"
)
