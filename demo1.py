from micrograd.engine import Value
from micrograd.nn import MLP


def demo():
    model = MLP(3, [4, 4, 1])

    # input
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]

    total_epochs = 1000
    for k in range(total_epochs):  # Training loop
        print(f"Epoch {k + 1}")

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
        learning_rate = 1.0 - 0.9 * k / total_epochs  # Decaying learning rate
        for p in model.parameters():
            p.data -= learning_rate * p.grad


if __name__ == "__main__":
    demo()
