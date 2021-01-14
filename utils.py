from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

def save_frames_as_gif(frames, fps=60, filename='gym_animation.gif'):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save('./' + filename, writer='imagemagick', fps=fps)


def get_state_rep_func(maze_size):
    def get_state(observation):
            """
            Converts observation (position in maze) into state.
            """
            if type(observation) is int:
                return observation
            return int(observation[1] * maze_size[1] + observation[0])

    return get_state    


def argmax(values, seed=None):
    """
    Takes in a list of values and returns the index of the item 
    with the highest value. Breaks ties randomly.
    returns: int - the index of the highest value in values
    """
    np.random.seed(seed)
    top_value = float("-inf")
    ties = []
    for i in range(len(values)):
        if values[i] > top_value:
            top_value = values[i]
            ties = []
        if values[i] == top_value:
            ties.append(i)
    return np.random.choice(ties)