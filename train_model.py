import numpy as np

X=np.array([
	[66,80,45],
	[58,120,69],
	[77,111,88],
	[50, 90, 98],
])

y=np.array([0,0,1,1])

print("X_shape:",X.shape)
print("y_shape:",y.shape)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe=Pipeline([
	("scaler",StandardScaler()),
	("clf",LogisticRegression())
])

pipe.fit(X,y)

print("训练完成!")

print("打印均值：",pipe.named_steps["scaler"].mean_)
print("打印标准差：",pipe.named_steps["scaler"].scale_)
print("打印参数：",pipe.named_steps["clf"].coef_)
print("打印截距：",pipe.named_steps["clf"].intercept_)

import joblib
joblib.dump(pipe,"models/model.joblib")