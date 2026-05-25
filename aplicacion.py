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
    page_title="Iris Species Classification",
    page_icon="🌸",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Cormorant Garamond', serif;
}

.stApp {
    background-image: url("https://images.unsplash.com/photo-1490750967868-88aa4486c946?q=80&w=1920");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.main > div {
    background: rgba(255,255,255,0.82);
    padding: 2rem;
    border-radius: 30px;
    backdrop-filter: blur(12px);
}

.block-container {
    padding-top: 1rem;
}

h1, h2, h3 {
    color: #4B2E5E;
}

p {
    color: #5C5470;
}

section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(15px);
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.65);
    border: 1px solid rgba(255,255,255,0.4);
    padding: 20px;
    border-radius: 20px;
    transition: 0.3s;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
}

div[data-testid="stMetric"]:hover {
    transform: scale(1.03);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    background-color: transparent;
    border-radius: 12px;
    color: #4B2E5E;
    font-size: 18px;
    transition: 0.3s;
}

.stTabs [aria-selected="true"] {
    background-color: #B48CC7 !important;
    color: white !important;
}

div[data-testid="stPlotlyChart"] {
    background: rgba(255,255,255,0.55);
    padding: 15px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
}

[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.6);
    border-radius: 20px;
    padding: 10px;
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
# HEADER
# ------------------------------------------------
st.markdown("""
<h1 style='text-align:center; font-size:60px; margin-bottom:0;'>
Iris Species
</h1>

<h3 style='text-align:center; font-weight:300; margin-top:0;'>
Classification Dashboard
</h3>
""", unsafe_allow_html=True)

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
            "background-color": "transparent"
        },
        "nav-link": {
            "font-size": "18px",
            "text-align": "center",
            "margin": "0px",
            "color": "#4B2E5E",
            "--hover-color": "#EADCF2",
        },
        "nav-link-selected": {
            "background-color": "#B48CC7",
            "color": "white",
            "border-radius": "12px",
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

    left, right = st.columns([1, 1])

    with left:

        species_count = df["species_name"].value_counts().reset_index()
        species_count.columns = ["Species", "Count"]

        fig = px.pie(
            species_count,
            names="Species",
            values="Count",
            hole=0.55,
            color_discrete_sequence=[
                "#D9B8F3",
                "#C79AD9",
                "#F3C6D3"
            ]
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(
                family="Cormorant Garamond",
                size=18
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        box_fig = px.box(
            df,
            y=iris.feature_names,
            color_discrete_sequence=["#C79AD9"]
        )

        box_fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(
                family="Cormorant Garamond",
                size=18
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
        background: rgba(180,140,199,0.25);
        padding: 30px;
        border-radius: 25px;
        text-align: center;
        margin-top: 30px;
        animation: fadeIn 1s;
    ">
        <h2 style="color:#4B2E5E;">
            Predicted Species
        </h2>

        <h1 style="
            color:#8B5FBF;
            font-size:60px;
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
            "#D9B8F3",
            "#C79AD9",
            "#F3C6D3"
        ]
    )

    fig_3d.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Cormorant Garamond",
            size=16
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
            "#D9B8F3",
            "#C79AD9",
            "#F3C6D3"
        ]
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Cormorant Garamond",
            size=16
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
            "#D9B8F3",
            "#C79AD9",
            "#F3C6D3"
        ]
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Cormorant Garamond",
            size=15
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
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Cormorant Garamond",
            size=16
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
<p style='text-align:center; color:#5C5470;'>
Developed for Data Mining • Universidad de la Costa
</p>
""", unsafe_allow_html=True)
