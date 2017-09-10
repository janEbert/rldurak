from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop, SGD


def create_model():
    inputs = Input(shape=(55,))
    x = Dense(150, activation='relu')(inputs)
    x = Dense(100, activation='relu')(x)
    attack = Dense(2)(x)
    defend = Dense(4)(x)
    push = Dense(2)(x)
    check = Dense(1)(x)
    wait = Dense(1)(x)
    outputs = concatenate([attack, defend, push, check, wait])

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=RMSprop(lr=0.1), loss='mse')
    # model.compile(optimizer=SGD(lr=0.1, decay=0.0), loss='mse')
    return model