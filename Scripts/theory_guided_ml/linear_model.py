from Scripts.dependencies.dependencies import initializers, constraints, Layer, K


class LinearModel(Layer):
    def __init__(self, units, **kwargs):
        # layers params
        self.units, self.state_size = units, units

        # the physical parameters of the complete model
        self.kv = None

        super(LinearModel, self).__init__(name='linear', **kwargs)

    def weight_adder(self, name=None, v_ini=0.5, v_min=0.0, v_max=1.0, is_trainable=True):
        w = self.add_weight(name=name, shape=(1, self.units),  #
                            initializer=initializers.Constant(value=v_ini),
                            constraint=constraints.min_max_norm(min_value=v_min, max_value=v_max),
                            trainable=is_trainable)
        return w

    def build(self, input_shape):
        self.kv = self.weight_adder('kv', 0.5, 0.0, 1.0)  # correction factor for ep (-)
        super(LinearModel, self).build(input_shape)

    @staticmethod
    def rescaler(kv):
        kv_ = (2.0 - 0.25) * kv + 0.25
        return kv_

    def water_balance(self, vi, ep):
        kv = self.rescaler(self.kv)
        ea = K.maximum(kv * ep, 0.0)
        vi = K.maximum(vi, 0.0)
        r = vi - ea
        return [r, ea]

    def call(self, inputs):
        u = self.units

        # input variables
        vi = inputs[:, :, :u]
        ep = inputs[:, :, u:]

        _r, _ea = self.water_balance(vi, ep)
        return _r

    def compute_output_shape(self, input_shape):
        nx, ny, nz = input_shape[0], input_shape[1], self.units
        return nx, ny, nz

    def get_config(self):
        return {'units': self.units}
