
# HYBRID STOCK PRICE PREDICTION
# Auto News + FinBERT + LSTM
# Predict Open, High, Low, Close


# STEP 1: Install Required Libraries

!pip install yfinance transformers torch tensorflow scikit-learn pandas numpy matplotlib --quiet


# STEP 2: Import Libraries

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from transformers import pipeline


# STEP 3: Select Stock

ticker = "HDFCBANK.NS"   # Change any stock here

print("Selected Stock:", ticker)

# STEP 4: Download Historical Stock Data

df = yf.download(ticker, period="5y")

# Fix yfinance MultiIndex issue if needed
df.columns = df.columns.get_level_values(0)

print(df.head())
print(df.tail())



# STEP 5: Automatically Fetch Latest News

stock = yf.Ticker(ticker)
news = stock.news

headline = "No latest news found"

for item in news:
    if 'title' in item:
        headline = item['title']
        break

    elif 'content' in item and 'title' in item['content']:
        headline = item['content']['title']
        break

print("Latest News Headline:")
print(headline)



# STEP 6: Load FinBERT Model

print("\nLoading FinBERT Model...")

finbert = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

print("FinBERT Loaded Successfully")




# STEP 7: FinBERT Sentiment Analysis

result = finbert(headline)

print("\nSentiment Result:")
print(result)

label = result[0]['label']

if label == "positive":
    sentiment_score = 1

elif label == "negative":
    sentiment_score = -1

else:
    sentiment_score = 0

print("Sentiment Score:", sentiment_score)


# STEP 8: Prepare Dataset
# OHLC + Sentiment

data = df[['Open', 'High', 'Low', 'Close']].copy()

# Add sentiment score to all rows
data['Sentiment'] = sentiment_score

print("\nFinal Dataset:")
print(data.head())
print(data.tail())



# STEP 9: Check Missing Values

print("\nMissing Values:")
print(data.isnull().sum())



# STEP 10: Scale Data

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)

print("\nScaled Data Shape:", scaled_data.shape)




# STEP 11: Create Training Data

X = []
y = []

for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    
    # Predict only OHLC (first 4 columns)
    y.append(scaled_data[i][:4])

X = np.array(X)
y = np.array(y)

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)

# Expected:
# X -> (samples, 60, 5)
# y -> (samples, 4)




# STEP 12: Build LSTM Model

model = Sequential()

model.add(
    LSTM(
        50,
        return_sequences=True,
        input_shape=(60, 5)
    )
)

model.add(LSTM(50))

model.add(Dense(4))  
# Predict Open, High, Low, Close

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

print("\nModel Summary:")
model.summary()




# STEP 13: Train Model

print("\nTraining Started...\n")

model.fit(
    X,
    y,
    epochs=10,
    batch_size=32
)

print("\nTraining Completed")



# STEP 14: Predict Tomorrow Prices

last_60_days = scaled_data[-60:]

X_input = np.array([last_60_days])

print("\nPrediction Input Shape:", X_input.shape)

prediction = model.predict(X_input)



# STEP 15: Inverse Transform Properly
# Because scaler has 5 columns
# but prediction has only 4

dummy = np.zeros((1, 5))

dummy[0, 0] = prediction[0, 0]   # Open
dummy[0, 1] = prediction[0, 1]   # High
dummy[0, 2] = prediction[0, 2]   # Low
dummy[0, 3] = prediction[0, 3]   # Close
dummy[0, 4] = sentiment_score    # Sentiment

predicted_price = scaler.inverse_transform(dummy)



# STEP 16: Final Predicted Prices

open_price = max(predicted_price[0][0], 0)
high_price = max(predicted_price[0][1], 0)
low_price = max(predicted_price[0][2], 0)
close_price = max(predicted_price[0][3], 0)

print("\n=================================")
print(" TOMORROW STOCK PRICE PREDICTION ")
print("=================================")

print("Tomorrow Opening Price :", open_price)
print("Tomorrow High Price    :", high_price)
print("Tomorrow Low Price     :", low_price)
print("Tomorrow Closing Price :", close_price)




# STEP 17: Final Market Direction

today_close = float(data['Close'].iloc[-1])

print("\nToday's Closing Price  :", today_close)

if close_price > today_close:
    print("Prediction: STOCK MAY GO UP 📈")

elif close_price < today_close:
    print("Prediction: STOCK MAY GO DOWN 📉")

else:
    print("Prediction: STOCK MAY STAY SAME ➖")




# STEP 18: Attractive Visualization
# Today vs Tomorrow Comparison
import numpy as np
import matplotlib.pyplot as plt

labels = ['Open', 'High', 'Low', 'Close']

today_values = [
    float(data['Open'].iloc[-1]),
    float(data['High'].iloc[-1]),
    float(data['Low'].iloc[-1]),
    float(data['Close'].iloc[-1])
]

tomorrow_values = [
    float(open_price),
    float(high_price),
    float(low_price),
    float(close_price)
]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10,6))

bars1 = plt.bar(x - width/2, today_values, width, label='Today')
bars2 = plt.bar(x + width/2, tomorrow_values, width, label='Tomorrow')

# Add value labels
for bar in bars1:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y, round(y,2),
             ha='center', va='bottom')

for bar in bars2:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y, round(y,2),
             ha='center', va='bottom')

plt.xticks(x, labels)
plt.title("Today vs Tomorrow Stock Prediction")
plt.ylabel("Price")
plt.legend()

plt.show()

