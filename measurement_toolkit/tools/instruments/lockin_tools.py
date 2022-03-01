from qcodes import Parameter


class LockinTimeParameter(Parameter):
    def __init__(self, lockins, name='t_lockin', unit='s', **kwargs):
        super().__init__(name=name, unit=unit, **kwargs)
        self.lockins = lockins

    def get_raw(self):
        time_constants = [lockin.time_constant() for lockin in self.lockins]
        assert len(set(time_constants)) == 1
        return time_constants[0]

    def set_raw(self, time_constant):
        for lockin in self.lockins:
            lockin.time_constant(time_constant)