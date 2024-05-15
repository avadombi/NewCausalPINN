from Scripts.dependencies.dependencies import initializers, constraints, Layer


class LinearWeights(Layer):
    def __init__(self, **kwargs):
        # the physical parameters of the complete model
        self.w0, self.w1, self.w2, self.w3 = None, None, None, None

        super(LinearWeights, self).__init__(name='linear_weights', **kwargs)

    def weight_adder(self, name=None, v_min=0.0, v_max=1.0):
        # constraints.min_max_norm(min_value=v_min, max_value=v_max)
        w = self.add_weight(name=name, shape=(1,),
                            initializer=initializers.Constant(value=0.5),
                            constraint=constraints.NonNeg(),
                            trainable=True)
        return w

    def build(self, input_shape):
        self.w0 = self.weight_adder()
        self.w1 = self.weight_adder()
        self.w2 = self.weight_adder()
        self.w3 = self.weight_adder()
        super(LinearWeights, self).build(input_shape)

    def call(self, inputs):
        crc, sr1, sr2, sr3 = inputs
        u = self.w0 * crc + self.w1 * sr1 + self.w2 * sr2 + self.w3 * sr3
        return u
