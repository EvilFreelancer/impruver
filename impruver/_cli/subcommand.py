class Subcommand:
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def _add_arguments(self):
        pass
