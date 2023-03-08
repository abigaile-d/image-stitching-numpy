import streamlit as st
import numpy as np

from scipy.ndimage.filters import convolve as conv2
from scipy.ndimage import maximum_filter
from scipy.ndimage.interpolation import map_coordinates

# 2d gaussian filter
def gaussian(sigma, N=None):
 
    if N is None:
        N = 2 * np.maximum(4, np.ceil(6*sigma))+1

    k = (N - 1) / 2.
            
    xv, yv = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
    
    # 2d gaussian filter    
    g = 1/(2 * np.pi * sigma**2) * np.exp(-(xv**2 + yv**2) / (2 * sigma ** 2))

    # 1st order derivatives
    gx = -xv / (2 * np.pi * sigma**4) * np.exp(-(xv**2 + yv**2) / (2 * sigma ** 2))
    gy = -yv / (2 * np.pi * sigma**4) * np.exp(-(xv**2 + yv**2) / (2 * sigma**2)) 

    return g, gx, gy


# harris corner function calls, im1 & im2 calls separated for caching
@st.cache_data
def harris_im1(im1, rel_th=0.9):
    return harris(im1, rel_th=rel_th)

@st.cache_data
def harris_im2(im2, rel_th=0.9):
    return harris(im2, rel_th=rel_th)

# harris corner detector algo implementation
@st.cache_data
def harris(im, sigma=1.0, rel_th=0.0001, k=0.04):
    # gaussian filters
    g, _, _ = gaussian(sigma)
    _, gx, gy = gaussian(np.sqrt(0.5))

    # compute partial derivatives
    Ix = conv2(im, -gx, mode='constant')
    Iy = conv2(im, -gy, mode='constant')

    # compute second moment matrix M
    IxIx = conv2(Ix**2, g, mode='constant')
    IyIy = conv2(Iy**2, g, mode='constant')
    IxIy = conv2(Ix*Iy, g, mode='constant')

    # compute corner response function R
    R = ((IxIx * IyIy) - (IxIy**2)) - k * (IxIx+IyIy)**2
    max_corner_val = np.amax(R)

    # find local maxima of corner response function R
    filter = np.ones((3,3))
    filter[1,1] = 0
    local_maxima = maximum_filter(R, footprint=filter, mode='constant')

    # get the points in the local maxima of R
    corners = R > local_maxima

    # filter using threshold
    y, x = np.nonzero((R > (rel_th * max_corner_val)) * corners)
    
    # remove at edges
    edge_n = 10
    idx = np.nonzero((x < edge_n) | (x > im.shape[1] - edge_n - 1) | ( y < edge_n) | (y > im.shape[0] - edge_n - 1))[0]
    x = np.delete(x, idx)
    y = np.delete(y, idx)

    return x, y, corners


# extra patches in the neighborhood
@st.cache_data
def extract_patches(im1, x1, y1, im2, x2, y2, patch_size=21):
    patches1 = np.zeros((x1.shape[0], patch_size, patch_size))
    patches2 = np.zeros((x2.shape[0], patch_size, patch_size))

    k=(patch_size-1)/2.
    xv, yv = np.meshgrid(np.arange(-k, k+1),np.arange(-k, k+1)) 
    for i in range(x1.shape[0]):
        patch = map_coordinates(im1, (yv + y1[i], xv + x1[i]))
        patches1[i,:,:] = patch
    for i in range(x2.shape[0]):
        patch = map_coordinates(im2, (yv + y2[i], xv + x2[i]))
        patches2[i,:,:] = patch
    return patches1, patches2


# compute normalized cross correlation for all patch pairs
@st.cache_data
def compute_ncc(patches1, patches2):
    patch_size = patches1.shape[1]

    # compute patch1 minus the mean
    patches1 = np.resize(patches1, (patches1.shape[0], patch_size*patch_size))
    patches1_norm = np.mean(patches1, axis=1)
    patches1_norm = np.repeat(np.expand_dims(patches1_norm, axis=1), patch_size*patch_size, axis=1)
    patches1_norm = patches1 - patches1_norm

    # compute patch2 minus the mean
    patches2 = np.resize(patches2, (patches2.shape[0], patch_size*patch_size))
    patches2_norm = np.mean(patches2, axis=1)
    patches2_norm = np.repeat(np.expand_dims(patches2_norm, axis=1), patch_size*patch_size, axis=1)
    patches2_norm = patches2 - patches2_norm

    # compute ncc values between all pair combinations
    distmat = np.zeros((patches1.shape[0], patches2.shape[0]))
    for i in range(patches1.shape[0]):
        x = np.sum(patches1_norm[i]**2) * np.sum(patches2_norm**2, axis=1)
        distmat[i, :] = (np.sum(patches1_norm[i] * patches2_norm, axis=1) / 
                        np.sqrt(np.sum(patches1_norm[i]**2) * np.sum(patches2_norm**2, axis=1)))

    return distmat


# estimate a homography transform given minimum # of point pairs
def fit_model(X, Xnew):
    A = np.zeros(X.shape[1] * 3)
    for i in range(X.shape[0]):
        A_row = np.hstack(([0] * X.shape[1], X[i], -Xnew[i,1]*X[i]))
        A = np.vstack((A, A_row))
        A_row = np.hstack((X[i], [0] * X.shape[1], -Xnew[i,0]*X[i]))
        A = np.vstack((A, A_row))
    A = A[1:]
    w, v = np.linalg.eig(np.matmul(A.T, A))
    H = v[:, np.argmin(w)]
    H = np.reshape(H, (3, X.shape[1]))

    return H


# RANSAC algo implementation
def ransac(src, dst, min_samples=4, max_trials=1000, residual_threshold_sigma=3):
    N_min = int(np.log(1-0.99) / np.log(1-(1-0.8)**2))
    N = max_trials
    t = np.sqrt(3.84 * residual_threshold_sigma**2)

    best_num_inliers = 0
    n = 0
    while(n < N_min or (n < N and best_num_inliers < src.shape[0] * 0.5)):
        # draw points randomly, minimum required = 4
        pts_idx = np.random.randint(len(src), size=min_samples)
        pts_src = src[pts_idx]
        pts_dst = dst[pts_idx]

        # fit a line to the 4 pts
        H = fit_model(pts_src, pts_dst)
        src_new = src @ H.T
        src_new = src_new / src_new[:, -1:]

        # compute euclidean distance from each pair of points
        dist = np.linalg.norm(src_new - dst, axis=1)

        # check inliers and save best
        num_inliers = np.sum(dist < t)
        if num_inliers > best_num_inliers: 
            best_num_inliers = num_inliers
            inliers_src = src[dist < t]
            inliers_dst = dst[dist < t]

        n+=1
    
    # refit to all inliers

    H = fit_model(inliers_src, inliers_dst)
    return H, inliers_src.shape[0]
