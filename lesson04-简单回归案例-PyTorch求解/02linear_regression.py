import numpy as np

# y = w * x + b
def compute_error_for_line_give_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2  # 平方和
        return totalError / float(len(points))

# compute gradient
def step_gradient(b_current, w_current, points, learningRate):
    b_gardient = 0
    w_gardient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gardient += -(2/N) * (y - ((w_current * x) + b_current) )# N for average
        w_gardient += -(2/N) * x * (y - ((w_current) * x) + b_current)
    new_b = b_current - (learningRate * b_gardient)
    new_w = w_current - (learningRate * w_gardient)
    return [new_b, new_w]

#iterate to optimize
def gradient_descent_runner(
        points, starting_b, starting_m,learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return [b,m]

def run():
    points = np.genfromtxt("data.csv",delimiter = ",")# 线函数
    learning_rate = 0.0001
    init_b = 0
    init_m = 0
    num_iterations = 1000
    print("staring gradient descent at b={0}, m={1}, error={2}"
          .format(init_b,init_m,
                  compute_error_for_line_give_points(init_b, init_m, points))
         )
    print("Running......")
    [b,m] = gradient_descent_runner(points, init_b, init_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".
          format(num_iterations, b, m,
                 compute_error_for_line_give_points(b, m, points)) )

if __name__ == "__main__":
    run()




















