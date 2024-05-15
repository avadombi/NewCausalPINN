from Scripts.dependencies.dependencies import K, initializers, constraints, Layer


class HBVModel(Layer):
    def __init__(self, units, mode='normal', inter_vars='qg', **kwargs):
        # layers params
        self.inter_vars = inter_vars
        self.units, self.state_size, self.mode = units, units, mode

        # the physical parameters of the complete model
        self.kv, self.fc, self.lp, self.cf, self.bt = None, None, None, None, None

        super(HBVModel, self).__init__(name='hbv', **kwargs)

    def weight_adder(self, name=None, v_ini=0.5, v_min=0.0, v_max=1.0, is_trainable=True):
        w = self.add_weight(name=name, shape=(1, self.units),  #
                            initializer=initializers.Constant(value=v_ini),
                            constraint=constraints.min_max_norm(min_value=v_min, max_value=v_max),
                            trainable=is_trainable)
        return w

    def build(self, input_shape):
        self.kv = self.weight_adder('kv', 0.5, 0.0, 1.0)  # correction factor for ep (-)
        self.fc = self.weight_adder('fc', 0.2, 0.0, 1.0)  # field capacity or the maximum soil moisture storage (mm)
        self.lp = self.weight_adder('lp', 0.5, 0.0, 1.0)  # limit above which ea reaches its potential value (-)
        self.cf = self.weight_adder('cf', 0.1, 0.0, 1.0)  # maximum value for Capillary Flow (mm/day)
        self.bt = self.weight_adder('bt', 2.0, 1.0, 6.0)  # param. of power relationship to simulate indirect runoff (-)

        super(HBVModel, self).build(input_shape)

    @staticmethod
    def rescaler(kv, fc, lp, cf, bt):
        kv_ = (2.0 - 0.25) * kv + 0.25
        fc_ = (1000.0 - 50.0) * fc + 50.0
        lp_ = (1.0 - 0.001) * lp + 0.001
        cf_ = (3.0 - 0.01) * cf + 0.01  # 10.0
        bt_ = bt
        return kv_, fc_, lp_, cf_, bt_

    @staticmethod
    def soil_moisture_reservoir(vi, ep, sm, fc, lp, kv, cf, bt):
        # 1. compute the abundant soil water (also referred to as direct runoff, sdr)
        sdr = K.maximum(sm + vi - fc, 0.0)

        # 2. compute the net amount of water that infiltrates into the soil (inet)
        inet = vi - sdr

        # 3. compute the actual evaporation
        tm = lp * fc
        ea = K.switch(condition=K.less(sm - tm, 0.0), then_expression=ep * (sm / tm), else_expression=ep) * kv

        # 4. compute the capillary flow ca = cf * (1.0 - sm / fc)
        ca = K.maximum(cf * (1.0 - sm / fc), 0.0)
        # ca = cf * (1.0 - sm / fc)

        # 5. compute part of the infiltrating water that will run off through the soil layer (seepage).
        sp = inet * K.pow(sm / fc, bt)

        return [inet, sdr, ea, ca, sp]

    @staticmethod
    def constraint_state_vars(sm):
        sm = K.maximum(sm, 1.0)
        return sm

    def step_do(self, step_in, states):
        u, p = self.units, 2 * self.units

        sm = states[0][:, :u]  # Soil moisture reservoir (mm)

        # Load the current input column
        vi = step_in[:, :u]
        ep = step_in[:, u:]

        # rescale parameters
        kv_, fc_, lp_, cf_, bt_ = self.rescaler(self.kv, self.fc, self.lp, self.cf, self.bt)

        # results from soil moisture reservoir
        [_inet, _sdr, _ea, _ca, _sp] = self.soil_moisture_reservoir(vi, ep, sm, fc_, lp_, kv_, cf_, bt_)

        # Water balance equations
        _dsm = _inet + _ca - _ea - _sp

        # next values of state variables
        next_sm = sm + K.clip(_dsm, -1e5, 1e5)

        # constrain values of sm, su and sl to be greater or equal to zero
        next_sm = self.constraint_state_vars(next_sm)

        # concatenate
        step_out = next_sm
        return step_out, [step_out]

    def call(self, inputs):
        u, p = self.units, 2 * self.units

        # define the initial state variables at the beginning
        init_states = [K.zeros((K.shape(inputs)[0], u))]

        # recursively calculate state variables by using RNN
        _, outputs, _ = K.rnn(self.step_do, inputs, init_states)

        sm = outputs[:, :, :u]

        # compute final process variables
        vi = inputs[:, :, :u]
        ep = inputs[:, :, u:]

        # rescale parameters
        kv_, fc_, lp_, cf_, bt_ = self.rescaler(self.kv, self.fc, self.lp, self.cf, self.bt)

        # results from soil moisture reservoir
        [_inet, _sdr, _ea, _ca, _sp] = self.soil_moisture_reservoir(vi, ep, sm, fc_, lp_, kv_, cf_, bt_)

        if self.mode == "normal":
            return _sp  # _sp or _qp, sg, ... (choose the optimal intermediate physical variable)
        elif self.mode == "analysis":
            if self.inter_vars == 'ea':
                return _ea
            elif self.inter_vars == 'ca':
                return _ca
            elif self.inter_vars == 'sp':
                return _sp

    def compute_output_shape(self, input_shape):
        nx, ny, nz = input_shape[0], input_shape[1], self.units
        return nx, ny, nz

    def get_config(self):
        return {'units': self.units, 'mode': self.mode, 'kv': self.kv, 'fc': self.fc, 'lp': self.lp, 'cf': self.cf,
                'bt': self.bt}
