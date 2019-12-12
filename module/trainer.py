from collections import deque
import numpy as np

class DelayTrainer:
    def __init__(self, target, delay, outlier_rate):
        self.target = target
        self.delay = delay
        self.outlier_rate = outlier_rate

        self.h_queue = deque()
        self.y_queue = deque()

        self.loss_queue = deque()
        self.loss_max_queue = deque()

    def train(self, x, y):
        self.target.input(x)

        self.h_queue.append(self.target.h)
        self.y_queue.append(y)

        loss =  np.sum(np.square(y - self.target.output()))
        self.loss_queue.append(loss)
        self.loss_max_queue.append(np.max(self.loss_queue))
        
        if len(self.h_queue) > self.delay:
            self.confirm_training()

    def confirm_training(self):
        h = self.h_queue.popleft()
        y = self.y_queue.popleft()

        loss = self.loss_queue.popleft()
        loss_max = self.loss_max_queue.popleft()

        if loss < loss_max * self.outlier_rate:
            self.target.update(h, y)

