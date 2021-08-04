

class DeadExp(Exception):
    def __init__(self, message="Dead"):
        self.message = message
        super().__init__(self.message)
