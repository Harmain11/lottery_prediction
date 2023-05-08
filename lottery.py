import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
import matplotlib.pyplot as plt


# # Data Preprocessing



df = pd.read_excel('input copy full 2 (copy).xlsx')




df.head()



df.info()




# split the numbers column into separate columns
numbers_df = df['Unnamed: 2'].str.split(expand=True)




numbers_df = numbers_df.astype(int)




numbers_df.describe()



# normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
numbers_df = scaler.fit_transform(numbers_df)



# split the data into training and testing sets
train_size = int(len(numbers_df) * 0.7)
train_data = numbers_df[:train_size, :]
test_data = numbers_df[train_size:, :]




def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)




# create the training and testing datasets
look_back = 5
trainX, trainY = create_dataset(train_data, look_back)
testX, testY = create_dataset(test_data, look_back)


# # Model Training and Testing



# define the model architecture
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(5, 5)))
model.add(LSTM(units=50))
# Adding a first Dropout layer
model.add(Dropout(0.2))
# Adding the first output layer
model.add(Dense(59))
model.add(Dropout(0.2))
model.add(Dense(units=5))

# compile the model
model.compile(optimizer=Adam(learning_rate= 0.00001 ), loss ='mse', metrics=['accuracy'] )
model.summary()




# train the model
history = model.fit(trainX, trainY, validation_data =(testX, testY), epochs=50, batch_size=32)



test_loss, test_acc = model.evaluate(testX, testY)
print('Test accuracy:', test_acc)



history_df = pd.DataFrame(history.history)
history_df[['accuracy', 'val_accuracy']].plot()




history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()


# # Prediction for Next Sequence



# use the model to predict the sequence of numbers
last_numbers = numbers_df[-5:]
last_numbers = np.reshape(last_numbers, (1, 5, 5))
predicted_numbers = model.predict(last_numbers)
predicted_numbers = scaler.inverse_transform(predicted_numbers)
predicted_numbers = [int(i) for i in predicted_numbers[0]]

print('Predicted next Sequence:', predicted_numbers)

