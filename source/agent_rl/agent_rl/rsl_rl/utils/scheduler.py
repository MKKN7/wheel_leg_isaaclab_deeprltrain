class IdentityScheduler:
    def __init__(self, v, **kwargs):
        self._v = v

    def update(self, step):
        return self._v

    @property
    def value(self):
        return self._v


class LinearScheduler(IdentityScheduler):
    def __init__(self, c0, c1, t0, t1):
        """
        :param c0: starting coefficient
        :param c1: final coefficient
        :param t0: initial step
        :param t1: interval step
        """
        super().__init__(c0)

        self.c0 = c0
        self.c1 = c1
        self.t0 = t0
        self.t1 = t1

    def update(self, step):
        stage = min(max((step - self.t0), 0) / self.t1, 1)
        self._v = stage * (self.c1 - self.c0) + self.c0
        return self._v

    @property
    def value(self):
        return self._v
