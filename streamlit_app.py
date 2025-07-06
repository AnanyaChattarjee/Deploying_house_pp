import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.title("🏠 House Price Prediction")

# Load model and metadata
model = joblib.load("house_price_model.pkl")
columns = joblib.load("model_columns.pkl")
defaults = joblib.load("model_defaults.pkl")

qual_map = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1}

# User inputs
st.header("Enter house features")
overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (sqft)", min_value=300, value=1500)
exter_qual = st.selectbox("Exterior Quality", list(qual_map.keys()))
kitchen_qual = st.selectbox("Kitchen Quality", list(qual_map.keys()))

if st.button("Predict Price"):
    # Step 1: Build user input
    user_input = pd.DataFrame({
        'GrLivArea': [gr_liv_area],
        'OverallQual': [overall_qual],
        'ExterQual': [qual_map[exter_qual]],
        'KitchenQual': [qual_map[kitchen_qual]]
    })

    # Step 2: Fill missing columns from defaults
    full_input = pd.DataFrame([0] * len(columns), index=columns).T
    for col in columns:
        if col in user_input.columns:
            full_input[col] = user_input[col].values[0]
        else:
            full_input[col] = defaults.get(col, 0)

    # Step 3: Predict
    prediction = model.predict(full_input)[0]
    st.success(f"💰 Predicted House Price: ₹{int(prediction):,}")

    # Step 4: Plot predictions for different GrLivArea values
    size_range = [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600]
    demo_rows = []

    for s in size_range:
        row = {
            'GrLivArea': s,
            'OverallQual': overall_qual,
            'ExterQual': qual_map[exter_qual],
            'KitchenQual': qual_map[kitchen_qual]
        }

        # Fill any remaining columns with defaults
        for col in columns:
            if col not in row:
                row[col] = defaults.get(col, 0)

        demo_rows.append(row)

    demo_df = pd.DataFrame(demo_rows)
    predicted_prices = model.predict(demo_df)

    demo_data = pd.DataFrame({
        'GrLivArea': size_range,
        'PredictedPrice': predicted_prices
    })

    # Plot
    fig = px.scatter(demo_data, x='GrLivArea', y='PredictedPrice', title='GrLivArea vs Predicted Price')
    fig.add_scatter(x=[gr_liv_area], y=[prediction],
                    mode='markers', marker=dict(size=12, color='red'), name='Your Input')
    st.plotly_chart(fig)
