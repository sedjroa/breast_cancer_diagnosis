import streamlit as st
import pickle as pk
import pandas as pd
from plotly import graph_objects as go
import numpy as np

def add_sidebar():
    st.sidebar.header("Cell Nuclei Details")

    with open("data/labels.pkl", "rb") as f:
        labels = pk.load(f)

    sidebar_inputs = {}  
    for label, lab_name, minval, meanval, maxval in labels:
        sidebar_inputs[lab_name] = st.sidebar.slider(
            label=label,
            min_value=float(minval),
            max_value=float(maxval),
            value=float(meanval)
        )
    
    return sidebar_inputs

def sidebar_input_scale(sidebar_inputs):
    labels = pk.load(open("data/labels.pkl","rb"))
    for label, value  in sidebar_inputs.items():
        match = next((vals_tpl for vals_tpl in labels if vals_tpl[1] == label), None)
        scale_value =( value - match[2] )/ (match[4] -match[2])
        sidebar_inputs[label] = scale_value
    return sidebar_inputs

def add_prediction(sidebar_infos):
    scaler = pk.load(open("model/scaler.pkl","rb"))
    model = pk.load(open("model/model.pkl","rb"))
    user_inputs = np.array(list(sidebar_infos.values())).reshape(1, -1)
    scaled_user_inputs = scaler.transform(user_inputs)
    pred = model.predict(scaled_user_inputs)
    return pred   

def get_radar_chart(sidebar_inputs):
    fig = go.Figure()
    
    measurements = [(key, val) for key, val in sidebar_inputs.items() if key.endswith("mean")]
    fig.add_trace(go.Scatterpolar(
        r=[measure[1] for measure in measurements],
        theta=[label[0].split("_")[0] for label in measurements],
        fill='toself',
        name='Mean Value'
    ))

    measurements = [(key, val) for key, val in sidebar_inputs.items() if key.endswith("se")]
    fig.add_trace(go.Scatterpolar(
        r=[measure[1] for measure in measurements],
        theta=[label[0].split("_")[0] for label in measurements],
        fill='toself',
        name='Std Value'
    ))

    measurements = [(key, val) for key, val in sidebar_inputs.items() if key.endswith("worst")]
    fig.add_trace(go.Scatterpolar(
        r=[measure[1] for measure in measurements],
        theta=[label[0].split("_")[0] for label in measurements],
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    showlegend=False
    )

    return fig

def main():
    st.set_page_config(page_title='Breast Cancer Diagnosis', 
                        page_icon=":female-doctor:",
                        layout="wide",
                        initial_sidebar_state="expanded")

    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Please connect this app to your cytology lab to help diagnore breast cancer form you "
        "tissue sample. This app predicts using a machine learning model whether a breast mass"
        "is malign or benign based on the measurements it receives from your cytosis lab. "
        "Your can also update the measurements by hand using the sliders in the sidebar.")

    col1, col2 = st.columns([4,1])

    sidebar_infos = add_sidebar()
    
    with col1:
        scale_sidebar_infos = sidebar_input_scale(sidebar_infos.copy())
        radar_chart = get_radar_chart(scale_sidebar_infos)  
        st.plotly_chart(radar_chart)
        
    with col2:
        prediction = add_prediction(sidebar_infos)
        if prediction[0] == 0:
            st.write("Benign")
        else:
            st.write("Malicious")


if __name__ == "__main__":
    main()