import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("Car_Emission.csv")
X = df.drop("co2_emission", axis=1)
y = df["co2_emission"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
print("Model trained successfully and saved as model.pkl")
