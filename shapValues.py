import shap
import pandas as pd
import numpy as np
shap.initjs()

customer = pd.read_csv("./customer_churn.csv")
customer.head()


from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X = customer.drop("Churn", axis=1) # Independent variables
y = customer.Churn # Dependent variable

# Split into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train a machine learning model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make prediction on the testing data
y_pred = clf.predict(X_test)

# Classification Report
print(classification_report(y_pred, y_test))

#configurando explicador
explainer = shap.Explainer(clf)
shap_values = explainer.shap_values(X_test)

#exibindo grafico resumido
shap.summary_plot(shap_values, X_test)
shap.dependence_plot("Subscription Length", shap_values[0], X_test,interaction_index="Age")

shap.plots.force(explainer.expected_value[0], shap_values[0][0,:], X_test.iloc[0, :], matplotlib = True)

shap.plots.force(explainer.expected_value[1], shap_values[1][6, :], X_test.iloc[6, :],matplotlib = True)