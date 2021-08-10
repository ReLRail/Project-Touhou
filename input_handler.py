from keyboard import press, release

class InputHandler:

    def __init__(self):
        self.inputs = []

    def set_input(self, inputs):
        if inputs is not None:
            for i in inputs:
                if i is not None:
                    press(i)

    def release_input(self, inputs):
        if inputs is not None:
            for i in inputs:
                if i is not None:
                    release(i)

    def set(self, inputs):
        for x in inputs:
            if x not in self.inputs:
                press(x)
        for x in self.inputs:
            if x not in inputs:
                release(x)
        self.inputs = inputs

    def clear(self):
        self.set([])

    def close(self):
        self.release_input(self.inputs)



