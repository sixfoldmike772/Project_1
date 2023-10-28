from flask import Flask, render_template, request
import requests
import pandas as pd
from io import StringIO
from sklearn.metrics import mean_absolute_error
from prophet import Prophet

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def stock_prediction():
    prediction = None
    accuracy = None
    error_message = None

    if request.method == "POST":
        symbol = request.form.get("stock_ticker")

        if symbol:
            try:
                api_key = "K827QDFCCCMMN6K1"  # Replace with your actual Alpha Vantage API key
                url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&outputsize=full&apikey={api_key}&datatype=csv'
                r = requests.get(url)

                if r.status_code == 200:
                    df = pd.read_csv(StringIO(r.text))
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values(by='timestamp')
                    columns_to_remove = ['open', 'volume', 'high', 'low']
                    df = df.drop(columns=columns_to_remove)
                    df.rename(columns={'timestamp': 'ds', 'close': 'y'}, inplace=True)
                    df = df.dropna()
                    
                    # Initialize and fit the Prophet model
                    model = Prophet()
                    model.fit(df)

                    # Create a DataFrame with future timestamps for the next day at a per-minute frequency
                    future = pd.DataFrame({'ds': pd.date_range(start=df['ds'].max(), periods=1440, freq='T')})
                    
                    # Make predictions using the model
                    forecast = model.predict(future)

                    # Calculate accuracy based on an acceptable threshold of 2
                    actual_values = df['y']
                    predicted_values = forecast['yhat']
                    threshold = 2  # You can adjust this threshold

                    # Calculate the number of accurate predictions
                    accurate_predictions = sum(abs(actual_values - predicted_values) <= threshold)

                    # Calculate accuracy as the ratio of accurate predictions to the total number of predictions
                    accuracy = (accurate_predictions / len(future)) * 100

                    prediction = round(float(forecast.iloc[-1]["yhat"]), 2)
                    
                else:
                    error_message = f"Failed to fetch data. Status code: {r.status_code}"
                    
            except Exception as e:
                error_message = str(e)

    return render_template("index.html", prediction=prediction, accuracy=accuracy, error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True)
