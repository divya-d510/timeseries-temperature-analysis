import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
import io
import base64
from PIL import Image
import calendar

# Set page configuration
st.set_page_config(
    page_title="Temperature Prediction App",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 36px;
#         font-weight: bold;
#         color: #1E88E5;
#         margin-bottom: 10px;
#     }
#     .sub-header {
#         font-size: 24px;
#         font-weight: bold;
#         color: #0D47A1;
#         margin-top: 30px;
#         margin-bottom: 15px;
#     }
#     .metric-container {
#         background-color: #f0f8ff;
#         padding: 15px;
#         border-radius: 10px;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#         margin-bottom: 20px;
#     }
#     .metric-label {
#         font-weight: bold;
#         color: #555;
#     }
#     .metric-value {
#         font-size: 28px;
#         font-weight: bold;
#         color: #1E88E5;
#     }
#     .warning {
#         color: #FF5722;
#         font-weight: bold;
#     }
#     .success {
#         color: #4CAF50;
#         font-weight: bold;
#     }
#     .info-box {
#         background-color: #e3f2fd;
#         padding: 10px;
#         border-radius: 5px;
#         margin-bottom: 15px;
#     }
# </style>
# """, unsafe_allow_html=True)

st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #90caf9;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #42a5f5;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .metric-container {
        background-color: #1c1f26;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.4);
        margin-bottom: 20px;
    }
    .metric-label {
        font-weight: bold;
        color: #cccccc;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #90caf9;
    }
    .warning {
        color: #ff9800;
        font-weight: bold;
    }
    .success {
        color: #66bb6a;
        font-weight: bold;
    }
    .info-box {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-left: 4px solid #90caf9;
        border-radius: 5px;
        margin-bottom: 15px;
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)


# Helper functions for model loading/prediction
@st.cache_resource
def load_model():
    """Load the pre-trained time series model."""
    try:
        # Try to load the pre-trained model from pickle file
        with open("C:\\Users\\dayad\\Downloads\\AirTempTimeSeries\\best_air_temp_model.pkl", 'rb') as f:
            trained_model = pickle.load(f)
            
        # Check if we have historical data from the model context
        # This assumes your pickle contains a fitted SARIMA model
        # You might need to adjust based on how your model is structured
        if hasattr(trained_model, 'data'):
            historical_data = trained_model.data.orig_endog
            dates = pd.date_range(start='2010-01-01', periods=len(historical_data), freq='MS')
            
            # Create DataFrame from the model's data
            df = pd.DataFrame({
                'Date': dates,
                'Temperature': historical_data
            })
            
            # Add month and year columns
            df['Month'] = df['Date'].dt.month
            df['Year'] = df['Date'].dt.year
            df['MonthName'] = df['Date'].dt.strftime('%b')
            
            return trained_model, df
        else:
            # If model doesn't contain data, use synthetic data for display
            # but still use the trained model for predictions
            df = create_synthetic_data()
            return trained_model, df
            
    except Exception as e:
        st.sidebar.warning(f"Could not load model from pickle file: {str(e)}")
        st.sidebar.info("Using synthetic model for demonstration. Upload your model.pkl file to the same folder as this app for real predictions.")
        
        # Fall back to synthetic model
        df = create_synthetic_data()
        
        # Create a simple SARIMA model on the synthetic data
        model_data = {
            'order': (1, 1, 1),
            'seasonal_order': (1, 1, 1, 12),
            'trend': 'c',
            'measurement_error': True,
        }
        
        model = SARIMAX(
            df['Temperature'], 
            order=model_data['order'], 
            seasonal_order=model_data['seasonal_order'],
            trend=model_data['trend']
        )
        
        fitted_model = model.fit(disp=False)
        
        return fitted_model, df

@st.cache_data
def create_synthetic_data():
    """Create synthetic temperature data for demo purposes."""
    # Generate dates from 2010 to current date
    start_date = pd.to_datetime('2010-01-01')
    end_date = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # Generate synthetic temperature data with seasonal patterns
    n = len(date_range)
    trend = np.linspace(0, 5, n)  # Increasing trend over time (climate change)
    seasonal = 15 * np.sin(np.linspace(0, 2*np.pi*10, n))  # Seasonal cycle
    noise = np.random.normal(0, 2, n)  # Random variations
    
    # Combine components
    temp = 22 + trend + seasonal + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': date_range,
        'Temperature': temp
    })
    
    # Add month and year columns
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['MonthName'] = df['Date'].dt.strftime('%b')
    
    return df

def predict_temperature(model, start_date, end_date, df):
    """Generate temperature predictions for a given date range."""
    # Get prediction
    try:
        # Calculate number of steps
        steps = ((end_date.year - start_date.year) * 12 + 
                end_date.month - start_date.month) + 1
        
        # Check if using trained model from pickle or synthetic model
        if hasattr(model, 'get_forecast'):
            # If it's a SARIMAX model
            forecast = model.get_forecast(steps=steps)
            
            # Extract prediction and confidence intervals
            pred_mean = forecast.predicted_mean
            pred_ci = forecast.conf_int()
        elif hasattr(model, 'predict'):
            # If it's another type of model (like Prophet)
            # Create future dataframe
            future_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Make prediction
            forecast = model.predict(future_df)
            
            # Extract prediction and confidence intervals
            pred_mean = forecast['yhat']
            pred_ci = pd.DataFrame({
                'lower': forecast['yhat_lower'],
                'upper': forecast['yhat_upper']
            })
        else:
            # Generic fallback for other model types
            st.warning("Using basic prediction method. Your model type may not be fully compatible.")
            # Simulate prediction (replace with appropriate method for your model)
            future_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
            pred_mean = pd.Series(np.zeros(len(future_dates)))
            
            for i, date in enumerate(future_dates):
                month = date.month
                # Simple prediction based on historical averages plus trend
                month_data = df[df['Month'] == month]
                if not month_data.empty:
                    trend = np.polyfit(month_data['Year'].values, month_data['Temperature'].values, 1)[0]
                    base = month_data['Temperature'].mean()
                    years_from_last = date.year - df['Year'].max()
                    pred_mean.iloc[i] = base + (trend * years_from_last)
                else:
                    pred_mean.iloc[i] = df['Temperature'].mean()  # Fallback
                    
            # Create simple confidence intervals
            pred_ci = pd.DataFrame({
                'lower': pred_mean - 2.5,
                'upper': pred_mean + 2.5
            })
        
        # Create date range for the forecast period
        forecast_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Temperature': pred_mean.values,
            'Lower_CI': pred_ci.iloc[:, 0].values,
            'Upper_CI': pred_ci.iloc[:, 1].values
        })
        
        # Add month and year columns
        forecast_df['Month'] = forecast_df['Date'].dt.month
        forecast_df['Year'] = forecast_df['Date'].dt.year
        forecast_df['MonthName'] = forecast_df['Date'].dt.strftime('%b')
        
        return forecast_df
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def get_month_statistics(df, month):
    """Calculate statistics for a specific month across all years."""
    month_data = df[df['Month'] == month]
    stats = {
        'mean': month_data['Temperature'].mean(),
        'min': month_data['Temperature'].min(),
        'max': month_data['Temperature'].max(),
        'std': month_data['Temperature'].std(),
        'trend': np.polyfit(month_data['Year'].values, month_data['Temperature'].values, 1)[0]
    }
    return stats, month_data

def generate_alerts(prediction_df, historical_df, month, year):
    """Generate temperature alerts based on predictions vs historical data."""
    alerts = []
    
    # Get month name
    month_name = calendar.month_name[month]
    
    # Get prediction for the specified month and year
    prediction = prediction_df[(prediction_df['Month'] == month) & 
                              (prediction_df['Year'] == year)]['Temperature'].values
    
    if len(prediction) == 0:
        return ["No prediction available for the selected date."]
    
    prediction = prediction[0]
    
    # Get historical data for this month
    month_data = historical_df[historical_df['Month'] == month]
    month_avg = month_data['Temperature'].mean()
    month_std = month_data['Temperature'].std()
    month_max = month_data['Temperature'].max()
    
    # Calculate how many standard deviations from mean
    z_score = (prediction - month_avg) / month_std
    
    # Generate alerts based on predictions
    if z_score > 2:
        alerts.append(f"⚠️ **ALERT**: {month_name} {year} is predicted to be significantly hotter than usual " 
                     f"({prediction:.1f}°C vs {month_avg:.1f}°C average).")
                     
    if z_score < -2:
        alerts.append(f"❄️ **NOTICE**: {month_name} {year} is predicted to be significantly cooler than usual "
                     f"({prediction:.1f}°C vs {month_avg:.1f}°C average).")
    
    if prediction > month_max:
        alerts.append(f"🔥 **EXTREME HEAT WARNING**: {month_name} {year} could set a new record high temperature!")
        
    # Add general advice based on temperature
    if prediction > 30:
        alerts.append("🌞 **HEAT ADVICE**: Consider staying hydrated and avoiding extended outdoor activities.")
    elif prediction < 5:
        alerts.append("🧣 **COLD ADVICE**: Prepare for cold temperatures; dress in layers and limit exposure.")
    
    if not alerts:
        alerts.append(f"✅ {month_name} {year} is predicted to have typical temperatures for this time of year.")
    
    return alerts

def plot_prediction_vs_historical(historical_df, forecast_df, selected_month=None, selected_year=None):
    """Create an interactive plot comparing historical temperatures with predictions."""
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_df['Date'],
        y=historical_df['Temperature'],
        mode='lines',
        name='Historical Data',
        line=dict(color='royalblue')
    ))
    
    # Add forecast data
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Temperature'],
        mode='lines',
        name='Forecast',
        line=dict(color='firebrick')
    ))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
        y=forecast_df['Upper_CI'].tolist() + forecast_df['Lower_CI'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(231,107,243,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=True,
        name='95% Confidence Interval'
    ))
    
    # Highlight the selected month if provided
    if selected_month and selected_year:
        highlight_date = pd.Timestamp(f"{selected_year}-{selected_month:02d}-01")
        selected_forecast = forecast_df[
            (forecast_df['Month'] == selected_month) & 
            (forecast_df['Year'] == selected_year)
        ]
        
        if not selected_forecast.empty:
            fig.add_trace(go.Scatter(
                x=[highlight_date],
                y=[selected_forecast['Temperature'].values[0]],
                mode='markers',
                marker=dict(size=12, color='red', symbol='star'),
                name=f'Selected Month ({calendar.month_name[selected_month]} {selected_year})'
            ))
    
    # Update layout
    fig.update_layout(
        title='Temperature Forecast vs Historical Data',
        xaxis_title='Date',
        yaxis_title='Temperature (°C)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
    )
    
    return fig

def plot_monthly_trends(df, month):
    """Plot the temperature trend for a specific month across years."""
    month_data = df[df['Month'] == month].copy()
    month_name = calendar.month_name[month]
    
    # Calculate the trend line
    if len(month_data) > 1:
        x = month_data['Year'].values
        y = month_data['Temperature'].values
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        trend_line = p(x)
    
        fig = go.Figure()
        
        # Add actual temperatures
        fig.add_trace(go.Scatter(
            x=month_data['Year'],
            y=month_data['Temperature'],
            mode='markers+lines',
            name=f'{month_name} Temperatures',
            line=dict(color='royalblue')
        ))
        
        # Add trend line
        fig.add_trace(go.Scatter(
            x=month_data['Year'],
            y=trend_line,
            mode='lines',
            name='Trend Line',
            line=dict(color='red', dash='dash')
        ))
        
        # Calculate trend rate per decade
        trend_per_decade = z[0] * 10
        trend_direction = "warming" if trend_per_decade > 0 else "cooling"
        
        fig.update_layout(
            title=f'{month_name} Temperature Trend ({abs(trend_per_decade):.2f}°C {trend_direction} per decade)',
            xaxis_title='Year',
            yaxis_title='Temperature (°C)',
            hovermode='x unified',
            height=400
        )
        
        return fig
    else:
        # Not enough data
        fig = go.Figure()
        fig.update_layout(
            title=f'Not enough historical data for {month_name}',
            xaxis_title='Year',
            yaxis_title='Temperature (°C)',
            height=400
        )
        return fig

def plot_monthly_distribution(df, month, prediction=None, year=None):
    """Plot the distribution of temperatures for a specific month with prediction highlight."""
    month_data = df[df['Month'] == month].copy()
    month_name = calendar.month_name[month]
    
    fig = go.Figure()
    
    # Create histogram with density curve
    fig.add_trace(go.Histogram(
        x=month_data['Temperature'],
        histnorm='probability density',
        name='Temperature Distribution',
        marker_color='royalblue',
        opacity=0.7
    ))
    
    # Add a kernel density estimate
    hist_data = [month_data['Temperature'].values]
    group_labels = ['Temperature']
    
    # Only add prediction marker if provided
    if prediction is not None and year is not None:
        fig.add_trace(go.Scatter(
            x=[prediction],
            y=[0],
            mode='markers',
            name=f'Prediction for {month_name} {year}',
            marker=dict(
                color='red',
                size=12,
                symbol='triangle-up',
                line=dict(width=2, color='DarkSlateGrey')
            )
        ))
        
        # Add annotation for percentile
        percentile = (month_data['Temperature'] < prediction).mean() * 100
        fig.add_annotation(
            x=prediction,
            y=0.03,
            text=f"{percentile:.0f}th percentile",
            showarrow=True,
            arrowhead=1
        )
    
    fig.update_layout(
        title=f'Temperature Distribution for {month_name}',
        xaxis_title='Temperature (°C)',
        yaxis_title='Probability Density',
        bargap=0.1,
        height=400
    )
    
    return fig

def plot_seasonal_patterns(df):
    """Plot the seasonal temperature patterns across years."""
    # Group by month and calculate statistics
    monthly_stats = df.groupby('Month').agg({
        'Temperature': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    monthly_stats.columns = ['Month', 'Mean', 'Std', 'Min', 'Max']
    monthly_stats['MonthName'] = monthly_stats['Month'].apply(lambda x: calendar.month_abbr[x])
    
    fig = go.Figure()
    
    # Add mean temperature line
    fig.add_trace(go.Scatter(
        x=monthly_stats['MonthName'],
        y=monthly_stats['Mean'],
        mode='lines+markers',
        name='Mean Temperature',
        line=dict(color='royalblue', width=3)
    ))
    
    # Add range (min to max)
    fig.add_trace(go.Scatter(
        x=monthly_stats['MonthName'].tolist() + monthly_stats['MonthName'].tolist()[::-1],
        y=monthly_stats['Max'].tolist() + monthly_stats['Min'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,176,246,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Temperature Range (Min-Max)'
    ))
    
    # Add confidence bands (mean ± std)
    fig.add_trace(go.Scatter(
        x=monthly_stats['MonthName'].tolist() + monthly_stats['MonthName'].tolist()[::-1],
        y=(monthly_stats['Mean'] + monthly_stats['Std']).tolist() + 
          (monthly_stats['Mean'] - monthly_stats['Std']).tolist()[::-1],
        fill='toself',
        fillcolor='rgba(231,107,243,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Standard Deviation Range'
    ))
    
    fig.update_layout(
        title='Seasonal Temperature Patterns',
        xaxis_title='Month',
        yaxis_title='Temperature (°C)',
        xaxis=dict(tickmode='array', tickvals=monthly_stats['MonthName']),
        hovermode='x unified',
        height=500
    )
    
    return fig

def compare_years(df, forecast_df, year1, year2):
    """Compare temperatures between two different years."""
    # Get data for the specified years
    data_year1 = df[df['Year'] == year1] if year1 < datetime.now().year else forecast_df[forecast_df['Year'] == year1]
    data_year2 = df[df['Year'] == year2] if year2 < datetime.now().year else forecast_df[forecast_df['Year'] == year2]
    
    # Create month order for proper sorting
    month_order = list(calendar.month_abbr)[1:]
    
    # Prepare figure
    fig = go.Figure()
    
    # Add line for year 1
    if not data_year1.empty:
        fig.add_trace(go.Scatter(
            x=data_year1['MonthName'],
            y=data_year1['Temperature'],
            mode='lines+markers',
            name=f'Year {year1}',
            line=dict(color='royalblue')
        ))
    
    # Add line for year 2
    if not data_year2.empty:
        fig.add_trace(go.Scatter(
            x=data_year2['MonthName'],
            y=data_year2['Temperature'],
            mode='lines+markers',
            name=f'Year {year2}',
            line=dict(color='firebrick')
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Temperature Comparison: {year1} vs {year2}',
        xaxis_title='Month',
        yaxis_title='Temperature (°C)',
        xaxis=dict(
            categoryorder='array',
            categoryarray=month_order
        ),
        hovermode='x unified',
        height=500
    )
    
    return fig

def get_travel_recommendations(forecast_df, min_preferred_temp=18, max_preferred_temp=28):
    """Generate travel recommendations based on forecasted temperatures."""
    
    # Filter future data (next 12 months)
    current_date = datetime.now()
    future_start = pd.Timestamp(f"{current_date.year}-{current_date.month:02d}-01")
    future_end = future_start + pd.DateOffset(months=12)
    
    future_data = forecast_df[
        (forecast_df['Date'] >= future_start) & 
        (forecast_df['Date'] <= future_end)
    ].copy()
    
    if future_data.empty:
        return "No future predictions available to make recommendations."
    
    # Calculate comfort score (closer to preferred range = higher score)
    future_data['TempScore'] = future_data['Temperature'].apply(
        lambda x: 100 - min(
            abs(x - min_preferred_temp), 
            abs(x - max_preferred_temp),
            max_preferred_temp - min_preferred_temp
        ) * 5
    )
    
    # Find best months
    best_months = future_data.sort_values('TempScore', ascending=False).head(3)
    
    recommendations = []
    for _, row in best_months.iterrows():
        month_year = row['Date'].strftime('%B %Y')
        temp = row['Temperature']
        
        # Generate recommendation text
        if min_preferred_temp <= temp <= max_preferred_temp:
            status = "ideal"
        elif temp < min_preferred_temp:
            status = "cool but pleasant"
        else:
            status = "warm but manageable"
            
        recommendations.append({
            'month_year': month_year,
            'temperature': temp,
            'status': status,
            'score': row['TempScore']
        })
    
    return recommendations

# Main application
def main():
    # Add a sidebar with instructions
    # st.sidebar.title("Temperature Prediction App")
    # st.sidebar.markdown("""
    # ### Instructions
    # 1. Place the `model.pkl` file in the same folder as this app
    # 2. The app will automatically load your model for predictions
    # 3. If no model is found, a demonstration model will be used
    # """)
    
    # Load model and data
    model, historical_df = load_model()
    
    # Show model load status
    # if "sidebar" not in st.session_state:
    #     st.session_state.sidebar = True
    #     # if isinstance(model, SARIMAX) or hasattr(model, 'params'):
    #     #     st.sidebar.success("✅ Using trained SARIMA model")
    #     # elif hasattr(model, 'predict'):
    #     #     st.sidebar.success("✅ Using trained model")
    #     # else:
    #     #     st.sidebar.warning("⚠️ Using demonstration model")
    #     if model is None:
    #         st.sidebar.warning("⚠️ No model found. Please place `model.pkl` in the app folder.")
    #     else:
    #         st.sidebar.success("✅ Model loaded successfully.")
    
    # Title and description
    st.markdown('<div class="main-header">🌡️ Temperature Prediction App</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    This app uses time series forecasting to predict future temperatures, analyze historical trends, 
    and provide insights to help with planning and climate awareness.
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different features
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Temperature Predictor", 
        "🔍 Historical Analysis", 
        "🧩 Comparison Tools",
        "ℹ️ About"
    ])
    
    # Get current date information for default values
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month
    
    # Generate forecasts (next 5 years by default)
    forecast_start = pd.Timestamp(f"{current_year}-{current_month:02d}-01")
    forecast_end = pd.Timestamp(f"{current_year + 5}-12-01")
    forecast_df = predict_temperature(model, forecast_start, forecast_end, historical_df)
    
    # Tab 1: Temperature Predictor
    with tab1:
        st.markdown('<div class="sub-header">Future Temperature Predictions</div>', unsafe_allow_html=True)
        
        # Input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                month = st.selectbox(
                    "Select Month", 
                    range(1, 13), 
                    format_func=lambda x: calendar.month_name[x],
                    index=current_month - 1
                )
            
            with col2:
                year = st.slider(
                    "Select Year", 
                    min_value=current_year,
                    max_value=current_year + 5, 
                    value=current_year + 1
                )
                
            # Submit button
            submitted = st.form_submit_button("Generate Prediction")
        
        if submitted or True:  # Always show initial prediction
            # Get prediction for selected month and year
            selected_prediction = forecast_df[(forecast_df['Month'] == month) & 
                                              (forecast_df['Year'] == year)]
            
            if not selected_prediction.empty:
                predicted_temp = selected_prediction['Temperature'].values[0]
                lower_ci = selected_prediction['Lower_CI'].values[0]
                upper_ci = selected_prediction['Upper_CI'].values[0]
                
                # Display prediction
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="metric-label">Predicted Temperature</div>
                    <div class="metric-value">{predicted_temp:.1f}°C</div>
                    <div>for {calendar.month_name[month]} {year}</div>
                    <div style="margin-top: 10px; font-size: 14px;">
                        95% confidence interval:<br>
                        {lower_ci:.1f}°C to {upper_ci:.1f}°C
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Get stats for this month historically
                    month_stats, _ = get_month_statistics(historical_df, month)
                    
                    st.markdown("""
                    <div style="font-size: 16px; margin-top: 15px;">
                    <b>Historical Context:</b>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Compare to historical average
                    diff_from_avg = predicted_temp - month_stats['mean']
                    diff_text = f"{abs(diff_from_avg):.1f}°C {'warmer' if diff_from_avg > 0 else 'cooler'}"
                    
                    st.markdown(f"""
                    <div style="margin-top: 10px; font-size: 14px;">
                        • Historical average: <b>{month_stats['mean']:.1f}°C</b><br>
                        • This prediction is <b>{diff_text}</b> than average<br>
                        • Historical range: {month_stats['min']:.1f}°C to {month_stats['max']:.1f}°C
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Display alerts
                    alerts = generate_alerts(forecast_df, historical_df, month, year)
                    
                    if alerts:
                        st.markdown("### Alerts & Recommendations")
                        for alert in alerts:
                            st.markdown(alert)
                    
                    # Plot prediction vs historical
                    fig = plot_prediction_vs_historical(historical_df, forecast_df, month, year)
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.error("No prediction available for the selected date.")
            
            # Additional insights
            st.markdown("### Additional Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Monthly distribution
                if not selected_prediction.empty:
                    fig = plot_monthly_distribution(
                        historical_df, 
                        month, 
                        prediction=predicted_temp, 
                        year=year
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Month trend over years
                fig = plot_monthly_trends(historical_df, month)
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Historical Analysis
    with tab2:
        st.markdown('<div class="sub-header">Historical Temperature Analysis</div>', unsafe_allow_html=True)
        
        # Month selector for analysis
        selected_month = st.selectbox(
            "Select month to analyze:", 
            range(1, 13), 
            format_func=lambda x: calendar.month_name[x],
            key="analysis_month"
        )
        
        # Get stats for selected month
        month_stats, month_data = get_month_statistics(historical_df, selected_month)
        
        # Display stats for the month
        st.markdown(f"### Historical Data for {calendar.month_name[selected_month]}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-label">Average Temperature</div>
            <div class="metric-value">{month_stats['mean']:.1f}°C</div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            trend_per_decade = month_stats['trend'] * 10
            trend_dir = "warming" if trend_per_decade > 0 else "cooling"
            
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-label">Temperature Trend</div>
            <div class="metric-value">{abs(trend_per_decade):.2f}°C</div>
            <div>per decade ({trend_dir})</div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-label">Temperature Range</div>
            <div class="metric-value">{month_stats['min']:.1f} - {month_stats['max']:.1f}°C</div>
            <div>min / max recorded</div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show trend plot
        st.markdown("### Temperature Trend Analysis")
        fig = plot_monthly_trends(historical_df, selected_month)
        st.plotly_chart(fig, use_container_width=True, key="trend_plot")
        
        # Show distribution
        st.markdown("### Temperature Distribution")
        fig = plot_monthly_distribution(historical_df, selected_month)
        st.plotly_chart(fig, use_container_width=True, key="distribution_plot")
        
        # Seasonal patterns
        st.markdown("### Seasonal Temperature Patterns")
        fig = plot_seasonal_patterns(historical_df)
        st.plotly_chart(fig, use_container_width=True, key="seasonal_plot")
    
    # Tab 3: Comparison Tools
    with tab3:
        st.markdown('<div class="sub-header">Temperature Comparison Tools</div>', unsafe_allow_html=True)
        
        # Year-to-year comparison
        st.markdown("### Compare Years")
        
        col1, col2 = st.columns(2)
        
        with col1:
            year1 = st.slider(
                "Select First Year", 
                min_value=min(historical_df['Year']),
                max_value=current_year + 5, 
                value=current_year - 1
            )
        
        with col2:
            year2 = st.slider(
                "Select Second Year", 
                min_value=min(historical_df['Year']),
                max_value=current_year + 5, 
                value=current_year
            )
        
        # Show year comparison chart
        fig = compare_years(historical_df, forecast_df, year1, year2)
        st.plotly_chart(fig, use_container_width=True)
        
        # Travel recommendations based on temperature
        st.markdown("### Travel Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_temp = st.slider(
                "Minimum Comfortable Temperature (°C)", 
                min_value=0,
                max_value=30, 
                value=18
            )
        
        with col2:
            max_temp = st.slider(
                "Maximum Comfortable Temperature (°C)", 
                min_value=min_temp + 5,
                max_value=40, 
                value=28
            )
        
        # Generate recommendations
        recommendations = get_travel_recommendations(forecast_df, min_temp, max_temp)
        
        if isinstance(recommendations, list):
            st.markdown("#### Best Times to Travel Based on Your Temperature Preferences")
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                <div class="metric-container">
                    <div style="font-weight: bold; font-size: 18px;">{i}. {rec['month_year']}</div>
                    <div style="font-size: 16px;">
                        Predicted temperature: <b>{rec['temperature']:.1f}°C</b> ({rec['status']})
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(recommendations)
        
        # Temperature anomaly map (placeholder for future extension)
        st.markdown("### Temperature Anomaly Forecast")
        
        # Create a heatmap of temperature anomalies
        months = list(calendar.month_abbr)[1:]
        current_month_idx = current_date.month - 1
        
        # Calculate next 12 months
        next_months = months[current_month_idx:] + months[:current_month_idx]
        years = [current_year] * (12 - current_month_idx) + [current_year + 1] * current_month_idx
        
        # Create data for heatmap
        anomaly_data = []
        
        for i, (month_name, year) in enumerate(zip(next_months, years)):
            month_num = months.index(month_name) + 1
            
            # Get forecast for this month
            forecast_temp = forecast_df[
                (forecast_df['Month'] == month_num) & 
                (forecast_df['Year'] == year)
            ]['Temperature'].values
            
            if len(forecast_temp) > 0:
                # Calculate anomaly from historical average
                historical_avg = historical_df[historical_df['Month'] == month_num]['Temperature'].mean()
                anomaly = forecast_temp[0] - historical_avg
                
                anomaly_data.append({
                    'Month': month_name,
                    'Anomaly': anomaly,
                    'Temperature': forecast_temp[0]
                })
        
        if anomaly_data:
            # Create dataframe for heatmap
            anomaly_df = pd.DataFrame(anomaly_data)
            
            # Create a horizontal heatmap
            fig = px.imshow(
                anomaly_df['Anomaly'].values.reshape(1, -1),
                y=['Next 12 Months'],
                x=anomaly_df['Month'],
                color_continuous_scale=px.colors.diverging.RdBu_r,
                color_continuous_midpoint=0,
                labels=dict(color="°C difference from average")
            )
            
            fig.update_layout(
                title='Temperature Anomaly Forecast (Next 12 Months)',
                height=250,
                xaxis=dict(title=''),
                yaxis=dict(title='')
            )
            
            # Add annotations with actual temperatures
            for i, month in enumerate(anomaly_df['Month']):
                fig.add_annotation(
                    x=i,
                    y=0,
                    text=f"{anomaly_df['Temperature'].iloc[i]:.1f}°C",
                    showarrow=False,
                    font=dict(color="black" if abs(anomaly_df['Anomaly'].iloc[i]) < 3 else "white")
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
            <b>How to read this chart:</b> Colors show how much warmer (red) or cooler (blue) each month 
            is expected to be compared to its historical average. Numbers show the actual predicted temperature.
            </div>
            """, unsafe_allow_html=True)
        
    # Tab 4: About
    with tab4:
        st.markdown('<div class="sub-header">About This Application</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### How It Works
        
        This application uses time series forecasting techniques to predict future temperatures based on historical patterns. The model takes into account:
        
        - **Seasonal patterns**: Regular temperature fluctuations throughout the year
        - **Long-term trends**: Gradual temperature changes over many years
        - **Historical anomalies**: Unusual temperature events in the past
        
        The predictions are presented with confidence intervals to indicate the range of possible outcomes. The model uses SARIMA (Seasonal Autoregressive Integrated Moving Average) methodology, which is well-suited for time series with seasonal components.
        
        ### Features
        
        - **Temperature Prediction**: Forecast temperatures for any month up to 5 years in the future
        - **Historical Analysis**: Examine temperature trends and patterns for specific months
        - **Comparison Tools**: Compare temperatures between different years
        - **Travel Recommendations**: Find the best times to travel based on temperature preferences
        - **Temperature Alerts**: Get warnings about potentially extreme temperatures
        
        ### Data Sources
        
        The model is trained on historical temperature data spanning multiple decades. This allows the application to capture both short-term seasonal variations and long-term climate trends.
        
        ### Limitations
        
        - Predictions become less certain the further into the future they extend
        - The model assumes that historical patterns will continue, which may not account for accelerating climate change
        - Extreme weather events and anomalies are difficult to predict accurately
        - Local microclimates and specific weather conditions are not captured by the model
        
        ### Future Enhancements
        
        - Integration with multiple data sources for more robust predictions
        - Addition of other weather variables (precipitation, humidity, etc.)
        - Support for location-specific forecasts
        - Extreme weather event probability estimations
        - Mobile application for on-the-go planning
        """)

if __name__ == "__main__":
    main()