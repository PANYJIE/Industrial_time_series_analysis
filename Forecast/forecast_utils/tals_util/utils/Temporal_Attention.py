import tensorflow as tf
from keras.layers import Layer, Input, Lambda, Dense, Dot, Activation,Reshape
from keras.models import Sequential

class TemporalAttention(Layer):
    def __init__(self, debug=False):
        self.output_transform = None
        self.softmax_normalizer = None
        self.attention_dot = None
        self.input_transformation = None
        self.hidden_state_transformation = None
        self.hidden_state_transform = None
        self.hidden_state = Lambda(lambda x: x[:, -1, :])
        self.debug_flag = debug
        super(TemporalAttention, self).__init__()

    def build(self, input_shape):
        timestep = input_shape[1]
        unit_num = input_shape[-1]
        self.hidden_state_transform = Lambda(lambda x: tf.expand_dims(x, axis=1))
        self.hidden_state_transformation = Dense(input_shape=[1, unit_num], units=unit_num, use_bias=False)
        self.input_transformation = Dense(input_shape=[timestep, unit_num], units=unit_num, activation='ReLU')

        self.attention_dot = Dot(axes=[2, 2], name='temporal_attention_weights')
        self.softmax_normalizer = Activation(activation='softmax', name='softmax_normalizer')

        self.output_transform = Dense(input_shape=[timestep, unit_num], units=unit_num)
        super(TemporalAttention, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        input_sequence = inputs
        hidden_state = self.hidden_state(input_sequence)

        # linear transformation for input sequence and last hidden state
        hidden_state_reshaped = self.hidden_state_transform(hidden_state)
        sequence_transformed = self.input_transformation(input_sequence)

        # calculate the temporal attention weights
        temporal_attention_weights = self.attention_dot([sequence_transformed, hidden_state_reshaped])
        normalized_attention = self.softmax_normalizer(temporal_attention_weights)

        # get the output sequence by weighting
        output_sequences = tf.multiply(input_sequence, normalized_attention)

        output_sequences = self.output_transform(output_sequences)

        return output_sequences

    def get_config(self):
        config = super().get_config()
        config.update({
            'debug': self.debug_flag
        })
        return config


if __name__ == '__main__':
    model = Sequential()
    model.add(Input(shape=(3, 50)))
    model.add(TemporalAttention())
    model.add(Dense(units=5))

    model.summary()
    print(model(tf.random.normal([1000, 3, 50])))
