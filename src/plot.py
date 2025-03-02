from matplotlib import pyplot as plt

def plot_intervals(intervals):
    """
    Plot a list of intervals.
    """
    for i in intervals:
        plt.plot([i[0], i[1]], [0, 0], 'k-')
    plt.show()
    