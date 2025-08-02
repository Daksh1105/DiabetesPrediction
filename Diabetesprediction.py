import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

scaler = StandardScaler()

#loading the model
loaded_model = pickle.load(open('D:/Python/trained_model.sav', 'rb'))

input_data = (1,106,98,40,237,39.5,0.714,23)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
std_data = scaler.fit_transform(input_data_reshaped)

prediction = loaded_model.predict(std_data)
print(prediction)

if(prediction[0] == 1):
  print("Person is diabetic")
else:
  print("Person is non-diabetic")