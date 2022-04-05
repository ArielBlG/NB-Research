from tensorflow.keras import initializers
from tensorflow.keras.layers import InputSpec, Layer
from tensorflow.keras import backend as K


class attentionlayer(Layer):

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(attentionlayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
            # creating layer weights
            # the __call__ method of your layer will automatically run build the first time it is called.
        """
        # The layer will accept inputs with shape 3
        # and raise an appropriate error message otherwise.
        self.input_spec = [InputSpec(ndim=3)]
        # check
        assert len(input_shape) == 3

        # define weights, shape of (?, one fixed vector)
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        # self.trainable_weights = [self.W]
        super(attentionlayer, self).build(input_shape)

    def call(self, x, mask=None):
        """
        """
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result


