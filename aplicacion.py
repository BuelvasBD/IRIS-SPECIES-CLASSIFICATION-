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
    page_title="Iris Floral Experience",
    page_icon="🌸",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600;700&family=Montserrat:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif;
}

/* ------------------------------------------------ */
/* BACKGROUND */
/* ------------------------------------------------ */

.stApp {
    background-color: #F8F3F7;
}

/* ------------------------------------------------ */
/* REMOVE STREAMLIT HEADER */
/* ------------------------------------------------ */

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"] {
    right: 2rem;
}

/* ------------------------------------------------ */
/* HERO SECTION */
/* ------------------------------------------------ */

.hero-section {

    height: 430px;

    border-radius: 35px;

    background-image: url("https://images.unsplash.com/photo-1490750967868-88aa4486c946?q=80&w=1974");

    background-size: cover;

    background-position: center;

    position: relative;

    overflow: hidden;

    margin-bottom: 40px;

    box-shadow: 0px 15px 40px rgba(0,0,0,0.12);
}

.hero-overlay {

    width: 100%;

    height: 100%;

    background: rgba(0,0,0,0.28);

    display: flex;

    flex-direction: column;

    justify-content: center;

    align-items: center;

    backdrop-filter: blur(2px);
}

.hero-title {

    color: white !important;

    font-size: 90px !important;

    font-family: 'Cormorant Garamond', serif !important;

    font-weight: 700 !important;

    margin-bottom: 10px;
}

.hero-subtitle {

    color: rgba(255,255,255,0.92);

    font-size: 22px;

    letter-spacing: 5px;

    text-transform: uppercase;
}

/* ------------------------------------------------ */
/* MAIN CONTAINER */
/* ------------------------------------------------ */

.main .block-container {

    padding-top: 2rem;

    padding-bottom: 4rem;

    max-width: 1400px;
}

/* ------------------------------------------------ */
/* NAVIGATION */
/* ------------------------------------------------ */

.nav-link {

    border-radius: 16px !important;

    transition: 0.35s !important;

    color: #6B4C68 !important;

    font-size: 15px !important;

    font-weight: 500 !important;

    padding-top: 14px !important;

    padding-bottom: 14px !important;
}

.nav-link:hover {

    background-color: #EADFEB !important;

    transform: translateY(-2px);
}

.nav-link-selected {

    background-color: #B68CB8 !important;

    color: white !important;
}

/* ------------------------------------------------ */
/* METRIC CARDS */
/* ------------------------------------------------ */

div[data-testid="stMetric"] {

    background: white;

    border-radius: 24px;

    padding: 30px;

    box-shadow: 0px 10px 30px rgba(0,0,0,0.06);

    transition: 0.3s;

    border: none;
}

div[data-testid="stMetric"]:hover {

    transform: translateY(-6px);
}

div[data-testid="stMetricLabel"] {

    color: #8D6C8B !important;

    font-size: 15px !important;
}

div[data-testid="stMetricValue"] {

    color: #4B2E5E !important;

    font-size: 34px !important;
}

/* ------------------------------------------------ */
/* PLOTLY */
/* ------------------------------------------------ */

.js-plotly-plot {

    background: white;

    border-radius: 28px;

    padding: 15px;

    box-shadow: 0px 10px 30px rgba(0,0,0,0.05);
}

/* ------------------------------------------------ */
/* DATAFRAME */
/* ------------------------------------------------ */

[data-testid="stDataFrame"] {

    background: white;

    border-radius: 25px;

    padding: 15px;

    box-shadow: 0px 10px 30px rgba(0,0,0,0.05);
}

/* ------------------------------------------------ */
/* TITLES */
/* ------------------------------------------------ */

h2, h3 {

    color: #4B2E5E !important;

    font-family: 'Cormorant Garamond', serif !important;

    font-size: 42px !important;
}

/* ------------------------------------------------ */
/* SLIDERS */
/* ------------------------------------------------ */

.stSlider {

    padding-top: 20px;
}

/* ------------------------------------------------ */
/* BUTTONS */
/* ------------------------------------------------ */

.stButton>button {

    background-color: #B68CB8;

    color: white;

    border-radius: 14px;

    border: none;

    padding: 12px 22px;

    transition: 0.3s;
}

.stButton>button:hover {

    background-color: #9F73A3;

    transform: scale(1.03);
}

/* ------------------------------------------------ */
/* DIVIDER */
/* ------------------------------------------------ */

hr {

    border-color: rgba(120,90,120,0.15);
}

/* ------------------------------------------------ */
/* SIDEBAR */
/* ------------------------------------------------ */

section[data-testid="stSidebar"] {

    background-color: #F4ECF4;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# HERO SECTION
# ------------------------------------------------

st.markdown("""
<div class="hero-section">

    <div class="hero-overlay">

        <h1 class="hero-title">
            Iris Species
        </h1>

        <p class="hero-subtitle">
            Floral Classification Experience
        </p>

    </div>

</div>
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
        "Scatter Matrix",
        "Feature Importance",
        "Dataset"
    ],
    orientation="horizontal",
    styles={
        "container": {
            "padding": "0!important",
            "background-color": "transparent",
        },
        "icon": {
            "display": "none"
        },
        "nav-link": {
            "text-align": "center",
            "margin": "0px 8px",
        },
        "nav-link-selected": {
            "background-color": "#B68CB8",
        },
    }
)

st.divider()

# ------------------------------------------------
# DASHBOARD
# ------------------------------------------------

if selected == "Dashboard":

    st.subheader("Model Performance Overview")

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
            hole=0.58,
            color_discrete_sequence=[
                "#E7C6FF",
                "#C8B6FF",
                "#FFD6E0"
            ]
        )

        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(
                family="Montserrat",
                color="#4B2E5E",
                size=16
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        box_fig = px.box(
            df,
            y=iris.feature_names,
            color_discrete_sequence=["#C8B6FF"]
        )

        box_fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(
                family="Montserrat",
                color="#4B2E5E",
                size=16
            )
        )

        st.plotly_chart(box_fig, use_container_width=True)

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------

elif selected == "Prediction":

    st.subheader("Flower Prediction")

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

    st.markdown(f"""
    <div style="
        background:white;
        border-radius:28px;
        padding:50px;
        text-align:center;
        margin-top:30px;
        box-shadow:0px 10px 30px rgba(0,0,0,0.05);
    ">

        <h2 style="
            color:#4B2E5E;
            margin-bottom:10px;
        ">
            Predicted Species
        </h2>

        <h1 style="
            color:#B68CB8;
            font-size:65px;
            font-family:'Cormorant Garamond', serif;
        ">
            {species_names[prediction]}
        </h1>

    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------
# 3D PLOT
# ------------------------------------------------

elif selected == "3D Plot":

    st.subheader("3D Flower Distribution")

    fig_3d = px.scatter_3d(
        df,
        x="sepal length (cm)",
        y="sepal width (cm)",
        z="petal length (cm)",
        color="species_name",
        color_discrete_sequence=[
            "#E7C6FF",
            "#C8B6FF",
            "#FFD6E0"
        ]
    )

    fig_3d.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(
            family="Montserrat",
            color="#4B2E5E"
        )
    )

    st.plotly_chart(fig_3d, use_container_width=True)

# ------------------------------------------------
# HISTOGRAMS
# ------------------------------------------------

elif selected == "Histograms":

    st.subheader("Feature Distributions")

    feature = st.selectbox(
        "Select Feature",
        iris.feature_names
    )

    fig = px.histogram(
        df,
        x=feature,
        color="species_name",
        marginal="box",
        color_discrete_sequence=[
            "#E7C6FF",
            "#C8B6FF",
            "#FFD6E0"
        ]
    )

    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(
            family="Montserrat",
            color="#4B2E5E"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# SCATTER MATRIX
# ------------------------------------------------

elif selected == "Scatter Matrix":

    st.subheader("Scatter Matrix")

    fig = px.scatter_matrix(
        df,
        dimensions=iris.feature_names,
        color="species_name",
        color_discrete_sequence=[
            "#E7C6FF",
            "#C8B6FF",
            "#FFD6E0"
        ]
    )

    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(
            family="Montserrat",
            color="#4B2E5E"
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
        color_continuous_scale="Purples"
    )

    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(
            family="Montserrat",
            color="#4B2E5E"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# DATASET
# ------------------------------------------------

elif selected == "Dataset":

    st.subheader("Iris Dataset")

    st.dataframe(df, use_container_width=True)

# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.divider()

st.markdown("""
<p style='text-align:center; color:#7A6178; font-size:14px;'>
Developed for Data Mining • Universidad de la Costa
</p>
""", unsafe_allow_html=True)
