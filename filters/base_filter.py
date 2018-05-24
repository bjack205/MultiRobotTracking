

class Filter:
    def __init__(self):
        pass

    def update(self, u, z, model):
        raise NotImplementedError

    def get_params(self):
        """
        :return: dictionary of filter parameters
        """
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def plot(self, logs, description):
        raise NotImplementedError
