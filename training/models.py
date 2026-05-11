import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


def transformer_encoder(x, embed_dim=64, num_heads=4, ff_dim=128):

    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim
    )(x, x)

    x = Add()([x, attention_output])
    x = LayerNormalization()(x)

    ffn = Dense(ff_dim, activation='relu')(x)
    ffn = Dense(embed_dim)(ffn)

    x = Add()([x, ffn])
    x = LayerNormalization()(x)

    return x


def build_sequential_model(input_shape=(32,32,3), num_classes=12):

    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)

    x = Reshape((64,64))(x)

    x = transformer_encoder(x)

    x = GlobalAveragePooling1D()(x)

    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    return model


def build_hierarchical_model(input_shape=(32,32,3), num_classes=12):

    inputs = Input(shape=input_shape)

    x = Reshape((256,12))(inputs)

    x = Dense(64)(x)

    x = transformer_encoder(x)

    x = Reshape((16,16,64))(x)

    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)

    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    return model


def build_parallel_model(input_shape=(32,32,3), num_classes=12):

    inputs = Input(shape=input_shape)

    # CNN Branch
    cnn = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    cnn = MaxPooling2D()(cnn)

    cnn = Conv2D(64, (3,3), activation='relu', padding='same')(cnn)
    cnn = MaxPooling2D()(cnn)

    cnn = GlobalAveragePooling2D()(cnn)

    # Transformer Branch
    trans = Reshape((256,12))(inputs)

    trans = Dense(64)(trans)

    trans = transformer_encoder(trans)

    trans = GlobalAveragePooling1D()(trans)

    # Fusion
    fusion = Concatenate()([cnn, trans])

    fusion = Dense(128, activation='relu')(fusion)

    fusion = Dropout(0.4)(fusion)

    outputs = Dense(num_classes, activation='softmax')(fusion)

    model = Model(inputs, outputs)

    return model
