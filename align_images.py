#!/usr/bin/env python

# Import necessary libraries.
import os, argparse
import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift

# argument parser
def getArgs():

   parser = argparse.ArgumentParser(
    description = '''Demo script showing various image alignment methods
                     including, phase correlation, feature based matching
                     and whole image based optimization.''',
    epilog = '''post bug reports to the github repository''')
    
   parser.add_argument('-im1',
                       '--image_1',
                       help = 'image to reference',
                       required = True)
                       
   parser.add_argument('-im2',
                       '--image_2',
                       help = 'image to match',
                       required = True)

   parser.add_argument('-m',
                       '--mode',
                       help = 'registation mode: translation, ecc or feature',
                       default = 'feature')

   parser.add_argument('-mf',
                       '--max_features',
                       help = 'maximum number of features to consider',
                       default = 5000)

   parser.add_argument('-fr',
                       '--feature_retention',
                       help = 'fraction of features to retain',
                       default = 0.15)

   parser.add_argument('-i',
                       '--iterations',
                       help = 'number of ecc iterations',
                       default = 5000)

   parser.add_argument('-te',
                       '--termination_eps',
                       help = 'ecc termination value',
                       default = 1e-8)

   return parser.parse_args()

# Enhanced Correlation Coefficient (ECC) Maximization
def eccAlign(im1,im2):

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

    # Find size of image1
    sz = im1.shape

    # Define the motion model
    warp_mode = cv2.MOTION_AFFINE

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
     number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    return im2_aligned, warp_matrix

# (ORB) feature based alignment      
def featureAlign(im1, im2):
  
  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(max_features)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
  
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
  
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * feature_retention)
  matches = matches[:numGoodMatches]

  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  #cv2.imwrite("matches.jpg", imMatches)
  
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
  
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
  
  return im1Reg, h      

# FFT phase correlation
def translation(im0, im1):
    
    # Convert images to grayscale
    im0 = cv2.cvtColor(im0,cv2.COLOR_BGR2GRAY)
    im1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    
    shape = im0.shape
    f0 = fft2(im0)
    f1 = fft2(im1)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = np.unravel_index(np.argmax(ir), shape)
    if t0 > shape[0] // 2:
        t0 -= shape[0]
    if t1 > shape[1] // 2:
        t1 -= shape[1]
    return [t0, t1]
      
if __name__ == '__main__':
    
  # parse arguments
  args = getArgs()
    
  # defaults feature values
  max_features = args.max_features
  feature_retention = args.feature_retention
  
  # Specify the ECC number of iterations.
  number_of_iterations = args.iterations

  # Specify the ECC threshold of the increment
  # in the correlation coefficient between two iterations
  termination_eps = args.termination_eps

  # Read the images to be aligned
  im1 =  cv2.imread(args.image_1);
  im2 =  cv2.imread(args.image_2);

  # Switch between alignment modes
  if args.mode == "feature":
   # align and write to disk
   aligned, warp_matrix = featureAlign(im1, im2)
   cv2.imwrite("reg_image.jpg",
    aligned,
    [cv2.IMWRITE_JPEG_QUALITY, 90])
   print(warp_matrix)
  elif args.mode == "ecc":
   aligned, warp_matrix = eccAlign(im1, im2)
   cv2.imwrite("reg_image.jpg",
    aligned,
    [cv2.IMWRITE_JPEG_QUALITY, 90])
   print(warp_matrix)
  else:
   warp_matrix = translation(im1, im2)
   print(warp_matrix)
