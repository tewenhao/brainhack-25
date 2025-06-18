class Test():
    def __init__(self):
        pass
    @property
    def name(self):
        return "Margarine"
    @name.setter
    def name(self,value):
        return value

temp = Test()
print(temp.name)
temp.name = "Organ"
print(temp.name)