import matplotlib.pyplot as plt
import numpy as np
import time

# ref: <https://www.wizard-notes.com/entry/python/matplotlib-realtime-plot-line>
class Monitor():
    def __init__(
        self,
        x_tick,
        length,
        xlabel="step",
        title='',
        label=None,
        color="black",
        marker='.-',
        alpha=1.0,
        ylim=None
    ):
        self.x_tick = x_tick
        self.length = length
        self.color = color
        self.marker = marker
        self.alpha = 1.0
        self.ylim = ylim
        self.label = label
        self.xlabel = xlabel
        self.title = title
        self.init_plot()

    def init_plot(self):
        self.x_vec = np.arange(0, self.length) * self.x_tick \
                     - self.length * self.x_tick
        self.y_vec = np.zeros(self.length)

        plt.ion()
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)

        self.line = ax.plot(self.x_vec, self.y_vec,
                            self.marker, color=self.color,
                            alpha=self.alpha)

        if self.ylim is not None:
            plt.ylim(self.ylim[0], self.ylim[1])
        plt.xlabel(self.xlabel)
        plt.title(self.title)
        plt.grid()
        plt.show()

        self.index = 0
        self.x_data = -self.x_tick
        self.pretime = 0.0
        self.fps = 0.0

    def reset(self):
        self.x_vec = np.arange(0, self.length) * self.x_tick \
                     - self.length * self.x_tick
        self.y_vec = np.zeros(self.length)

    def update_index(self):
        self.index = self.index + 1 if self.index < self.length-1 else 0

    def update_ylim(self, y_data):
        ylim = self.line[0].axes.get_ylim()
        if   y_data < ylim[0]:
            plt.ylim(y_data*1.1, ylim[1])
        elif y_data > ylim[1]:
            plt.ylim(ylim[0], y_data*1.1)

    def update(self, y_data):
        self.x_data += self.x_tick
        self.y_vec[self.index] = y_data

        y_pos = self.index + 1 if self.index < self.length else 0
        tmp_y_vec = np.r_[self.y_vec[y_pos:self.length], self.y_vec[0:y_pos]]
        self.line[0].set_ydata(tmp_y_vec)
        if self.ylim is None:
            self.update_ylim(y_data)

        plt.pause(0.001)
        self.update_index()