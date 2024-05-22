import sklearn
import shap
import sklearn.linear_model

#Conjunto de dados de preco de habitacao
X, y = shap.datasets.california(n_points=1000)

#100 intancias para uso de distribuicao em segundo plano
X100 = shap.utils.sample(X, 100)

#modelo linear simples
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)

print("Model coeicients:\n")
for i in range(X.shape[1]):
    print(X.columns[i], "=", model.coef_[i].round(5))
    
    
shap.partial_dependence_plot(
        "MedInc",
        model.predict,
        X100,
        ice=False,
        model_expected_value=True,
        feature_expected_value=True
    )