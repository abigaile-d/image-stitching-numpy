# Panorama Stitching from Scratch using Numpy

View in Streamlit: (https://abigaile-d-image-stitching-numpy.streamlit.app/)

This is an image stitching implementation from scratch, without using ready-made packages like OpenCV, skimage, etc. The implementation is done mostly with numpy, except for some minor parts (detailed later).

## Methodology

### Algorithms and Steps

1. Find image keypoints using **Harris Corner Detector**
    - implemented using numpy, with minor use of conv2d and maximum_filter from scipy
2. Extract patches surrounding each selected keypoints
    - implemented using numpy, with minor use of map_coordinates from scipy
3. Match keypoints by using distance metric **Normalized Cross Correlation** and finding highest correlated images
    - implemented using numpy
4. Estimated alignment of images by finding a transformation model using **RANSAC algorithm**
    - both ransac and model fitting are implemented using numpy
5. Align images using the estimated transformation model
    - implemented using cv2.warpPerspective
6. Stitch and blend the two images
    - implemented using numpy

### Technologies

- Python
- Packages: numpy, cv2 and scipy
- Web app: streamlit
- Images: PIL and plotly

## Getting Started

Clone the repository, and run the following commands:

    pip install -r requirements_full.txt
    streamlit run stitch_image.py

*Note: There is a separate requirements.txt and requirements_full.txt as the former lists only the minimum packages needed by streamlit cloud.*
