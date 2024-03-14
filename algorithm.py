import matplotlib.pyplot as plt
import numpy as np
import time

def mse(y, y_hat):
    error = 1 / len(y) * np.sum((y - y_hat) ** 2)
    return error

def main():
    N = 0
    while N not in range(1, 11):
        N = input('Number of data points (integer between 1 and 10): ')
        try:
            N = int(N)
        except ValueError:
            N = 0
    
    t = 0
    while t not in range(1, 501):
        t = input('Number of epochs (integer between 1 and 500): ')
        try:
            t = int(t)
        except ValueError:
            t = 0
            
    lr = 0
    while (lr > float('1e-1')) or (lr < float('1e-3')):
        lr = input('Learning rate (float between 1e-3 and 1e-1): ')
        try:
            lr = float(lr)
        except ValueError:
            lr = 0
    
    x = np.random.rand(N,)
    y = np.random.rand(N,)
    
    print(f'\nxs: {x}')
    print(f'\nys: {y}')
    
    w = 0.1
    b = 0.01
    
    for i in range(t):
        start_time = time.time()
        
        prediction = (w * x) + b
        current_cost = mse(y, prediction)
        
        weight_gradient = -(2 / float(N)) * sum(x * (y - prediction))
        bias_gradient = -(2 / float(N)) * sum(y - prediction)
        
        w -= lr * weight_gradient
        b -= lr * bias_gradient
        
        end_time = time.time()
        print(f"Epoch {i + 1}/{t}\n{round((end_time - start_time) * 1000000, 3)}μs - loss: {current_cost}")
    
    final_prediction = (w * x) + b
    
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, marker='D', color='blue')
    plt.plot([min(x), max(x)], [min(prediction), max(prediction)], color='red')
    plt.title('Regression line predicted by gradient descent')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == '__main__':
    main()