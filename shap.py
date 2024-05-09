import shap
import pandas as pd
import numpy as np
shap.initjs()

customer = pd.read_csv("data/customer_churn.csv")
customer.head()