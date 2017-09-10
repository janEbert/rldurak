from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop, SGD


def create_model():
    model = Sequential()

    model.add(Dense(150, activation='relu', input_shape=(55,))) 
    model.add(Dense(100, activation='relu'))
    attack = Dense(2)
    defend = Dense(4)
    push = Dense(2)
    check = Dense(1)
    wait = Dense(1)
    model.add(Merge([attack, defend, push, check, wait]))

    model.compile(optimizer=RMSprop(lr=0.1), loss='mse')
    # model.compile(optimizer=SGD(lr=0.1, decay=0.0), loss='mse')
    return model