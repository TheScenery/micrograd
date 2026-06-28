from micrograd.engine import Value
from micrograd.nn import MLP


def demo():
    model = MLP(3, [4, 4, 1])

    # Example input
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]

    threadhold = 1e-3
    loss = Value(float("inf"))
    max_epochs = 10000
    k = 0
    while loss.data > threadhold and k < max_epochs:  # Training loop
        k += 1
        print(f"Epoch {k}")

        # Forward pass
        outputs = [model(x) for x in xs]
        print("Outputs:", outputs)

        # loss calculation (mean squared error)
        loss = sum((output[0] - y) ** 2 for output, y in zip(outputs, ys)) / len(xs)
        print("Loss:", loss)

        # Backward pass
        model.zero_grad()  # Reset gradients before backward pass
        loss.backward()

        # gradient descent step
        learning_rate = 1.0 - 0.9 * k / max_epochs  # Decaying learning rate
        for p in model.parameters():
            p.data -= learning_rate * p.grad


if __name__ == "__main__":
    demo()
