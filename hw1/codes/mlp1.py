from network import Network
from layers import Relu, Sigmoid, Linear, Gelu

# Your model defintion here
# You should explore different model architecture
model = Network()
model.add(Linear('fc1', 784, 64, 0.01))

# model.add(Relu("relu"))
# model.add(Sigmoid("sigmoid"))
model.add(Gelu("gelu"))

model.add(Linear("fc2", 64, 10, 0.01))
