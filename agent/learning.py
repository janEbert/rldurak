from keras.models import Sequential
from keras.layers import Dense, Masking
from keras.optimizers import SGD
from keras.preprocessing.sequences import pad_sequences

model = Sequential()

max_len = 100
pad_sequences(sequence, max_len)
model.add(Masking(input_shape=(1, max_len)))
model.add(Dense(32, activation="relu")) 
model.add(Dense(16, activation="relu"))
model.add(Dense(5))

model.compile(optimizer=SGD(lr=0.1, decay=0.0), loss="mse")

