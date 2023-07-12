import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import slic, felzenszwalb, quickshift
from skimage.color import label2rgb


if __name__ == '__main__':
    # Load color image
    image = io.imread("../../import_Almere/seg2.jpg")

    # SLIC segmentation
    slic_segments = slic(image, n_segments=100, compactness=10)
    slic_segmented_image = label2rgb(slic_segments, image, kind='avg')

    # Felzenszwalb segmentation
    felzenszwalb_segments = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
    felzenszwalb_segmented_image = label2rgb(felzenszwalb_segments, image, kind='avg')

    # Quickshift segmentation
    quickshift_segments = quickshift(image, kernel_size=10, max_dist=50, ratio=1)
    quickshift_segmented_image = label2rgb(quickshift_segments, image, kind='avg')

    # Visualize results
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    ax[0, 0].imshow(image)
    ax[0, 0].set_title("Original Image")

    ax[0, 1].imshow(slic_segmented_image)
    ax[0, 1].set_title("SLIC Segmentation")

    ax[1, 0].imshow(felzenszwalb_segmented_image)
    ax[1, 0].set_title("Felzenszwalb Segmentation")

    ax[1, 1].imshow(quickshift_segmented_image)
    ax[1, 1].set_title("Quickshift Segmentation")

    for a in ax.ravel():
        a.axis('off')

    plt.tight_layout()
    plt.show()

    print("hey")