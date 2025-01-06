from .Temporal_Attention import TemporalAttention
import tensorflow as tf
from keras.layers import Input, BatchNormalization, LSTM, SimpleRNN, Dense,Flatten
from keras.models import Sequential

def network_generation(unit_type,
                       input_shape,
                       layers=1, units_per_layer=50,
                       dense_layer=1, units_per_fc_layer=50,
                       output_step=3,
                       normalized=True,
                       attention=True,
                       **kwargs) -> Sequential:
    assert unit_type in ['lstm', 'ann', 'rnn']
    assert layers > 0
    assert units_per_layer > 0
    assert dense_layer > 0
    assert units_per_fc_layer > 0
    assert output_step > 0

    encoder = Sequential()
    encoder.add(Input(input_shape))
    if normalized:
        encoder.add(BatchNormalization())

    if unit_type == 'lstm':
        for i in range(layers):
            encoder.add(LSTM(units=units_per_layer, return_sequences=True))
    if unit_type == 'rnn':
        for i in range(layers):
            encoder.add(SimpleRNN(units=units_per_layer, return_sequences=True))
    if unit_type == 'ann':
        for i in range(layers):
            encoder.add(Dense(units=units_per_layer, activation='relu'))
    if attention:
        encoder.add(TemporalAttention())
    encoder.add(Flatten())
    encoder.add(Dense(units=output_step))
    return encoder


if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))

    with tf.device('GPU:0'):
        network = network_generation(unit_type='lstm', input_shape=[2, 50])
        network.summary()
        network.compile()
