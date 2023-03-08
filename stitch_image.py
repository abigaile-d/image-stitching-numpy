import streamlit as st
import numpy as np
from cv2 import warpPerspective

from PIL import Image
import plotly.express as px

import utils


st.header("Image Stitching from Scratch")
st.sidebar.write("This is an image stitching implementation that (mostly) doesn't use ready-made functions \
                 i.e. from opencv, sklearn, etc.")
st.sidebar.write("Harris Corner, RANSAC, Homography model estimation and pixel blending are implemented using numpy. \
                 Opencv is only used when transforming the final image (i.e. cv2.warpPerspective).")

target_width = 400

left_col, right_col = st.columns(2)
# file1 = left_col.file_uploader("Upload the 1st image:", type=["png", "jpg", "jpeg"])
# file2 = right_col.file_uploader("Upload the 2nd image:", type=["png", "jpg", "jpeg"])
file1 = None
file2 = None

# display default or uploaded input images
if file1 is None and file2 is None:
    st.markdown("**Sample Input Images:**")
else:
    st.markdown("**Uploaded Images:**")
left_col, right_col = st.columns(2)
if file1 is not None:
    image1 = Image.open(file1)
else:
    image1 = Image.open('input/left_image.jpg')
image1 = image1.resize((target_width, int(target_width / image1.size[0] * image1.size[1])))
left_col.image(image1, caption='Input Image 1')

if file2 is not None:
    image2 = Image.open(file2)
else:
    image2 = Image.open('input/right_image.jpg')
image2 = image2.resize((target_width, int(target_width / image2.size[0] * image2.size[1])))
right_col.image(image2, caption='Input Image 1')

# convert to numpy array in grayscale
im1 = np.array(image1.convert('L'))
im2 = np.array(image2.convert('L'))

# call own implementation of harris detector
str = "Detecting keypoints with Harris corner detector algo..."
with st.spinner(str):
    x1, y1, corners1 = utils.harris_im1(im1, rel_th=0.9)
    x2, y2, corners2 = utils.harris_im2(im2, rel_th=0.9)
    st.caption('[DONE] ' + str)

# extract nxn patches on the neighborhood of each corner pt (x, y)
str = "Extracting patches around the keypoints..."
with st.spinner(str):
    patches1, patches2 = utils.extract_patches(im1, x1, y1, im2, x2, y2)
    st.caption('[DONE] ' + str)

# find distance between all keypoint pairs using normalized cross correlation (ncc)
str = "Computing distance matrix (using normalized cross correlation) between all possible pairs of keypoints..."
with st.spinner(str):
    distmat = utils.compute_ncc(patches1, patches2)
    st.caption('[DONE] ' + str)

# finding best matches = biggest NCC values
str = "Finding best keypoint matches..."
with st.spinner(str):
    # closest patch2 to each patch1
    ncc1 = np.amax(distmat, axis=1)
    ids1 = np.argmax(distmat, axis=1)

    # closest patch1 to each patch2
    ncc2 = np.amax(distmat, axis=0)
    ids2 = np.argmax(distmat, axis=0)

    matches = np.take(ids2, ids1)
    matches = matches[np.equal(matches, np.arange(len(ids1)))]
    pairs = np.vstack((matches, np.take(ids1, matches), np.take(ncc1, matches))).T
    pairs = pairs[pairs[:, 2].argsort()[::-1]]
    st.caption('[DONE] ' + str)

# display images with the top matches, mark top 40 matches
top_n = 40
st.markdown("**Top {} Keypoint Matches:**".format(top_n))
montage = np.concatenate((im1, im2), axis=1)
fig = px.imshow(montage, aspect='equal', color_continuous_scale=px.colors.sequential.Blues)
fig.layout.coloraxis.showscale = False
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)

fig.add_scatter(x=np.take(x1, pairs[:top_n, 0].astype(int)), 
                y=np.take(y1, pairs[:top_n, 0].astype(int)), 
                mode="markers", marker=dict(color='yellow'))
fig.add_scatter(x=np.take(x2, pairs[:top_n, 1].astype(int))+im1.shape[1], 
                y=np.take(y2, pairs[:top_n, 1].astype(int)), 
                mode="markers", marker=dict(color='red'))
fig.layout.update(showlegend=False) 
st.plotly_chart(fig)

# use own implementation of RANSAC to estimate homography / perspective transformation
str = "Finding best homography model using RANSAC..."
with st.spinner(str):
    dst = np.vstack((np.take(x1, pairs[:, 0].astype(int)), np.take(y1, pairs[:, 0].astype(int)), [1] * pairs.shape[0])).T
    src = np.vstack((np.take(x2, pairs[:, 1].astype(int)), np.take(y2, pairs[:, 1].astype(int)), [1] * pairs.shape[0])).T
    H, n = utils.ransac(src, dst)
    st.caption('[DONE] ' + str)
    st.caption('[DONE] Found {} number of inliers'.format(n))

# transform image2
st.markdown("**After Perspective Transformation:**")
im2_new = warpPerspective(np.array(image2), H, (im2.shape[1]*2, im2.shape[0]))
left_col, right_col = st.columns(2)
left_col.image(image1, caption="Original Image 1")
right_col.image(Image.fromarray(im2_new[:, im2.shape[1]:]), caption="Image 2 after Transformation")

# stitch images, with overlapping region get 0.5 from each images
st.markdown("**Stitched Image:**")
im1 = np.array(image1)
im_combined = im2_new.copy()
im_combined[:, :im2.shape[1], :] = im1[:, :, :]
# crop at top right corner
right_border = np.min(np.cumsum(np.sum(im_combined, axis=2)!=0, axis=1)[:, -1])
im_combined = im_combined[:, :right_border, :]

mask = (np.sum(im2_new[:, :im1.shape[1], :], axis=2) != 0) & (np.sum(im1[:, :, :], axis=2) != 0)
im_combined[:, :im2.shape[1], :][mask] = im_combined[:, :im2.shape[1], :][mask] * 0.5 + \
    im2_new[:, :im2.shape[1], :][mask] * 0.5
st.image(Image.fromarray(im_combined), use_column_width='always')

# mixing pixels from 2 images using gradient
st.markdown("**Final Stitched Image:**")
mask_im1 = np.zeros((im_combined.shape[0], right_border), dtype=bool)
mask_im2 = np.zeros((im_combined.shape[0], right_border), dtype=bool)
mask_combined = np.zeros((im_combined.shape[0], right_border))
mask_im1[:, :im1.shape[1]] = (np.sum(im1[:, :, :], axis=2) != 0)
mask_im2 = (np.sum(im2_new[:, :right_border, :], axis=2) != 0)
mask_combined = mask_im1*1 + mask_im2*1 - 1
intersect_i = np.where(mask_combined == 1)

blend = np.zeros((im_combined.shape[0], right_border))
blend[mask_im1] = 1
blend[mask_im2] = 0

i = 0
for r in range(blend.shape[0]):
    if np.sum(mask_combined[r] == 1) == 0:
        continue
    row_filter = np.linspace(1, -1, np.sum(mask_combined[r] == 1))
    row_filter = (row_filter+1) * 0.5
    blend[r, np.where(mask_combined[r] == 1)[0]] = row_filter

blend = blend[:, :im1.shape[1], None]
im_combined[:, :im1.shape[1], :] = im1[:, :im1.shape[1], :] * (blend) + \
    im2_new[:, :im1.shape[1], :] * (1-blend)
st.image(Image.fromarray(im_combined[:, :right_border, :]), use_column_width='always')
