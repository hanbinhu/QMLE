def pltshow():
    import matplotlib.pyplot as plt
    plt.show()

def plot_images(images, size=10):
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(size,size,figsize=(6,6))
    for j in range(size):
        for k in range(size):
            i = np.random.choice(range(len(images)))
            ax[j][k].set_axis_off()
            image = np.squeeze(np.transpose(images[i], (1,2,0)))
            ax[j][k].imshow(image)
