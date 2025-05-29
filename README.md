# 🌡️ Temperature Prediction App

An interactive web application for temperature forecasting and climate analysis built with Streamlit and advanced time series modeling.

## 🚀 Features

### 🔮 **Smart Forecasting**
- Predict monthly temperatures up to **5 years** into the future
- **95% confidence intervals** for reliable uncertainty quantification
- Powered by pre-trained SARIMA models

### 📊 **Interactive Analytics**
- **Historical trend analysis** with seasonal pattern detection
- **Year-to-year comparisons** for climate insights
- **Temperature distribution** analysis for specific months
- **Anomaly detection** and extreme weather alerts

### 🗺️ **Travel Intelligence**
- **Personalized travel recommendations** based on temperature preferences
- **Optimal timing suggestions** for outdoor activities
- **Climate-aware planning** tools

### 📈 **Rich Visualizations**
- Dynamic **time series charts** with Plotly
- **Interactive heatmaps** for temperature patterns
- **Distribution plots** and statistical summaries
- **Confidence interval visualizations**

## 🎯 Use Cases

- **Climate Research**: Analyze long-term temperature trends
- **Travel Planning**: Find optimal travel dates based on weather preferences
- **Agriculture**: Plan seasonal activities with temperature forecasts
- **Event Planning**: Schedule outdoor events with weather insights
- **Education**: Learn about time series analysis and climate patterns

## 🛠️ Installation

### Prerequisites

- **Python 3.8+** 
- **pip** package manager
- **Git** (for cloning the repository)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/temperature-prediction-app.git
   cd temperature-prediction-app
   ```

2. **Set up virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application**
   ```bash
   streamlit run temperature_prediction_app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

### Manual Installation

If `requirements.txt` is not available, install dependencies manually:

```bash
pip install streamlit pandas numpy matplotlib seaborn plotly statsmodels prophet pillow
```

## 📁 Project Structure

```
temperature-prediction-app/
├── 📂 Data/
│   └── surface-air-temperature-monthly-mean.csv    # Historical temperature dataset
├── 📦 best_air_temp_model.pkl                     # Pre-trained SARIMA model
├── 🐍 temperature_prediction_app.py               # Main Streamlit application
├── 📓 AirTempTimeSeries.ipynb                     # Model training notebook
├── 📋 requirements.txt                            # Python dependencies
├── 📖 README.md                                   # This file

```

## 🎮 How to Use

### 1. **Temperature Predictor Tab**
- Select any month and year for temperature forecasting
- View predictions with confidence intervals
- Get automatic alerts for extreme temperatures
- Access practical recommendations based on forecasts

### 2. **Historical Analysis Tab**
- Explore temperature trends over time
- Analyze seasonal patterns and distributions
- View statistical summaries and insights
- Identify long-term climate changes

### 3. **Comparison Tools Tab**
- Compare temperature profiles between different years
- Get personalized travel recommendations
- Find optimal dates based on your temperature preferences
- Analyze year-over-year climate variations

### 4. **About Tab**
- Learn about the app's methodology
- Understand model limitations and assumptions
- Access technical documentation

## 📊 Dataset Information

**Source**: Monthly Mean Air Temperature Dataset (1982-2020)
- **Size**: 462 monthly observations across 38 years
- **Coverage**: Global temperature averages
- **Format**: CSV with date and temperature columns
- **Quality**: Clean dataset with no missing values

**Download**: [AirTempTS.zip]([https://drive.google.com/drive/folders/19CKEcJ7Wb9hnRSMsBrJx6lUfwTWM1WLv?usp=sharing])

## 🤖 Model Details

### **Primary Model: SARIMA**
- **Seasonal Autoregressive Integrated Moving Average**
- Optimized for monthly temperature patterns
- Handles seasonality and long-term trends
- Pre-trained on 38 years of historical data

### **Fallback: Synthetic Model**
- Used when pre-trained model is unavailable
- Parameters: SARIMA(1,1,1)(1,1,1,12)
- Demonstrates app functionality with artificial data

### **Performance Features**
- 95% confidence intervals
- Up to 5-year forecasting horizon
- Seasonal pattern recognition
- Trend analysis capabilities

## ⚠️ Important Limitations

| Limitation | Impact |
|------------|--------|
| **Future Uncertainty** | Predictions become less accurate over longer time horizons |
| **Climate Change** | Model may not capture rapid climate shifts |
| **Extreme Events** | Limited ability to predict unprecedented weather anomalies |
| **Geographic Scope** | General forecasts, not location-specific |
| **Data Dependency** | Synthetic data used when model unavailable |

## 🔧 Troubleshooting

### Common Issues

**🔍 Model Not Found**
```
Solution: Ensure `best_air_temp_model.pkl` is in the project directory
Alternative: App will use synthetic data automatically
```

**📦 Dependency Errors**
```bash
# Update pip and reinstall
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

**🌐 Streamlit Port Issues**
```bash
# Use different port
streamlit run temperature_prediction_app.py --server.port 8502
```

**💾 Dataset Access Problems**
```
Note: App functions with synthetic data if dataset unavailable
For full functionality, download the dataset from provided link
```

## 🚀 Future Roadmap

- [ ] **Multi-location Support**: Location-specific temperature forecasting
- [ ] **Weather Variables**: Add precipitation, humidity, and wind predictions
- [ ] **Mobile App**: React Native or Flutter implementation
- [ ] **API Integration**: Real-time weather data incorporation
- [ ] **Machine Learning**: Explore deep learning models (LSTM, Transformer)
- [ ] **Export Features**: PDF reports and data export functionality

