import numpy as np
import csv
import matplotlib.pyplot as plt


def draw(row):
    with open('train.csv', 'r') as csv_file:
        is_head_line = True
        i = 0
        for data in csv.reader(csv_file):
            if is_head_line:
                is_head_line = False
                continue
            if i < row:
                i += 1
                continue

            # The first column is the label
            label = data[0]

            # The rest of columns are pixels
            pixels = data[1:]

            # Make those columns into a array of 8-bits pixels
            # This array will be of 1D with length 784
            # The pixel intensity values are integers from 0 to 255
            pixels = np.array(pixels, dtype='uint8')

            # Reshape the array into 28 x 28 array (2-dimensional array)
            pixels = pixels.reshape((28, 28))

            # Plot
            plt.title('Label is {label}'.format(label=label))
            plt.imshow(pixels, cmap='gray')
            plt.show()

            break  # This stops the loop, I just want to see one


if __name__ == '__main__':
    """source from:https://stackoverflow.com/questions/37228371/visualize-mnist-dataset-using-opencv-or-matplotlib-pyplot"""
    draw(3)
