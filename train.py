import random
from model import LoRAModel

def train():
    model = LoRAModel()
    epochs = 5
    for epoch in range(epochs):
        loss = random.uniform(0.1, 1.0) / (epoch + 1)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

if __name__ == '__main__':
    train()
