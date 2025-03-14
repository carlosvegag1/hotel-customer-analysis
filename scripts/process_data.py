import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

df = pd.read_csv("datosclienteshotel.csv", sep=";")
df.drop(columns=["Unnamed: 0"], inplace=True)
print(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas.")
df.head()