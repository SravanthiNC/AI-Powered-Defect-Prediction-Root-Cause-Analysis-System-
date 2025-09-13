import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from datetime import datetime
import numpy as np
import os

# Function to load models and data
@st.cache_resource
def load_models():
    try:
        model = joblib.load('manufacturing_quality_model.joblib')
        scaler = joblib.load('feature_scaler.joblib')
        with open('feature_columns.json', 'r') as f:
            feature_cols = json.load(f)
        return model, scaler, feature_cols
    except FileNotFoundError:
        st.error("Model files not found. Please run the model training notebook first.")
        return None, None, None

@st.cache_data
def load_data():
    try:
        main_unit_df = pd.read_csv('main_unit_assembly_data.csv')
        component_df = pd.read_csv('component_assembly_data.csv')
        return main_unit_df, component_df
    except FileNotFoundError:
        st.error("Data files not found. Please run the data generation notebook first.")
        return None, None

# Load models and data
model, scaler, feature_cols = load_models()
main_unit_df, component_df = load_data()

# Check if data and models are available
if main_unit_df is None or component_df is None:
    st.warning("Please generate the data first by running the generate_data.ipynb notebook.")
    st.stop()

if model is None or scaler is None or feature_cols is None:
    st.warning("Please train the models first by running the model_training.ipynb notebook.")
    st.stop()

# Load data
main_unit_df = pd.read_csv('main_unit_assembly_data.csv')
component_df = pd.read_csv('component_assembly_data.csv')

# Set page config
st.set_page_config(page_title="Manufacturing Quality Analysis", layout="wide")

# Title
st.title("Manufacturing Quality Analysis Dashboard")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Overview", "Failure Analysis", "Predictions", "Cost Impact"])

if page == "Overview":
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_units = len(main_unit_df['USN'].unique())
        st.metric("Total Units", f"{total_units:,}")
    
    with col2:
        failure_rate = len(main_unit_df[main_unit_df['RESULTFLAG'] == 'F']) / len(main_unit_df)
        st.metric("Failure Rate", f"{failure_rate:.2%}")
    
    with col3:
        total_vendors = main_unit_df['VENDOR'].nunique()
        st.metric("Total Vendors", total_vendors)
    
    with col4:
        total_stages = main_unit_df['STAGE'].nunique()
        st.metric("Total Stages", total_stages)
    
    # Failure Trends
    st.subheader("Failure Trends")
    main_unit_df['TRNDATE'] = pd.to_datetime(main_unit_df['TRNDATE'])
    daily_failures = main_unit_df.groupby(main_unit_df['TRNDATE'].dt.date)['RESULTFLAG'].apply(
        lambda x: (x == 'F').mean()
    ).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_failures['TRNDATE'], 
                            y=daily_failures['RESULTFLAG'],
                            mode='lines+markers'))
    fig.update_layout(title="Daily Failure Rate",
                     xaxis_title="Date",
                     yaxis_title="Failure Rate")
    st.plotly_chart(fig)

elif page == "Failure Analysis":
    st.subheader("Failure Analysis by Stage and Vendor")
    
    # Stage-Vendor Heatmap
    stage_vendor_failures = pd.crosstab(
        main_unit_df[main_unit_df['RESULTFLAG'] == 'F']['STAGE'],
        main_unit_df[main_unit_df['RESULTFLAG'] == 'F']['VENDOR']
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=stage_vendor_failures.values,
        x=stage_vendor_failures.columns,
        y=stage_vendor_failures.index,
        colorscale='Reds'))
    
    fig.update_layout(title="Stage-Vendor Failure Heatmap",
                     xaxis_title="Vendor",
                     yaxis_title="Stage")
    st.plotly_chart(fig)
    
    # Error Code Analysis
    st.subheader("Error Code Analysis")
    error_counts = main_unit_df[main_unit_df['A_ERRORCODE'] != '']['A_ERRORCODE'].value_counts()
    
    fig = go.Figure(data=go.Bar(x=error_counts.index, y=error_counts.values))
    fig.update_layout(title="Error Code Distribution",
                     xaxis_title="Error Code",
                     yaxis_title="Count")
    st.plotly_chart(fig)

elif page == "Predictions":
    st.subheader("Failure Prediction Tool")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_stage = st.selectbox("Select Stage", sorted(main_unit_df['STAGE'].unique()))
        selected_vendor = st.selectbox("Select Vendor", sorted(main_unit_df['VENDOR'].unique()))
    
    with col2:
        selected_date = st.date_input("Select Date")
        selected_time = st.time_input("Select Time")
    
    if st.button("Predict Failure Probability"):
        # Prepare features
        datetime_str = f"{selected_date} {selected_time}"
        dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        
        features = pd.DataFrame({
            'STAGE_encoded': [main_unit_df[main_unit_df['STAGE'] == selected_stage].index[0]],
            'VENDOR_encoded': [main_unit_df[main_unit_df['VENDOR'] == selected_vendor].index[0]],
            'TRNDATE_hour': [dt.hour],
            'TRNDATE_day': [dt.day],
            'TRNDATE_month': [dt.month]
        })
        
        # Scale features and predict
        features_scaled = scaler.transform(features)
        failure_prob = model.predict_proba(features_scaled)[0][1]
        
        st.metric("Failure Probability", f"{failure_prob:.2%}")
        
        if failure_prob > 0.7:
            st.error("⚠️ High risk of failure! Recommend immediate inspection.")
        elif failure_prob > 0.3:
            st.warning("⚠️ Moderate risk of failure. Monitor closely.")
        else:
            st.success("✅ Low risk of failure.")

elif page == "Cost Impact":
    st.subheader("Cost Impact Analysis")
    
    # Cost parameters
    cost_per_failure = st.number_input("Cost per Failure ($)", value=1000, step=100)
    
    # Calculate costs
    vendor_costs = main_unit_df[main_unit_df['RESULTFLAG'] == 'F'].groupby('VENDOR').agg({
        'USN': 'count'
    }).reset_index()
    vendor_costs['cost_impact'] = vendor_costs['USN'] * cost_per_failure
    
    # Vendor cost impact
    fig = go.Figure(data=go.Bar(
        x=vendor_costs['VENDOR'],
        y=vendor_costs['cost_impact'],
        text=vendor_costs['cost_impact'].apply(lambda x: f"${x:,.0f}"),
        textposition='auto',
    ))
    
    fig.update_layout(title="Cost Impact by Vendor",
                     xaxis_title="Vendor",
                     yaxis_title="Cost Impact ($)")
    st.plotly_chart(fig)
    
    # Total cost impact
    st.metric("Total Cost Impact", f"${vendor_costs['cost_impact'].sum():,.2f}")

# Footer
st.markdown("---")
st.markdown("TCS AI Hackathon - Manufacturing Quality Analysis System")