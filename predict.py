import joblib
import numpy as np

model=joblib.load("models/model.joblib")

print(model)

new_X=np.array([
	[60, 85, 95],
])

print("new_X.shape:",new_X.shape)

prediction=model.predict(new_X)

print("预测类别为：",prediction)

probability=model.predict_proba(new_X)
print("probability:",probability)

prob_class_1=probability[0][1]
print("该病人高危的预测概率为：",prob_class_1)