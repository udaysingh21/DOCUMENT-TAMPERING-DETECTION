import cv2
import numpy as np
import os

'''
Yes, in the provided function, ela_scale is essentially the ELA (Error Level Analysis) image. 
It represents the differences between the original grayscale image and the ELA image, 
highlighting regions where there are discrepancies in pixel values due to compression or editing.

So, ela_scale or ela_image represents the ELA image, 
which can be further analyzed or used for various purposes like detecting edited regions in an image.

Yes, a resave quality of 60 is generally considered low and should be sufficient for creating tampered images with noticeable compression artifacts. 
This level of compression is likely to introduce significant distortions, making it easier to detect any tampering or changes in the image.
'''

def generate_ela_batch(original_image, resave_quality=60):
    # Ensure original_image is a numpy array
    original_image = np.array(original_image)

    # Convert original image to grayscale
    original_image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply a blur to the grayscale image
    blurred_gray = cv2.GaussianBlur(original_image_gray, (5, 5), 0)

    # Calculate the pixel-wise differences between the two images
    ela_image = cv2.absdiff(blurred_gray, original_image_gray)

    # Calculate the ELA scale
    ela_scale = np.abs(original_image_gray - ela_image)

    # Normalize the ELA scale to 0-255 range
    ela_scale = ((ela_scale - np.min(ela_scale)) / (np.max(ela_scale) - np.min(ela_scale))) * 255

    # Convert to uint8
    ela_scale = ela_scale.astype(np.uint8)

    return ela_scale

def preprocessing(img1, img2):
    # Convert both images to numpy arrays
    img1 = np.array(img1)
    img2 = np.array(img2)

    print(img1.shape), print(img2.shape)

    # Convert images to grayscale if they have more than one channel
    if len(img1.shape) > 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) > 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Resize the images to the same dimensions
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    print(img1.shape), print(img2.shape)
    print(type(img1)), print(type(img2))

    return img1, img2


import cv2
import numpy as np

class MSE:
    def __init__(self):
        pass

    def convert_to_gray(self, img):
        image_np = np.array(img)

        # Convert color image to grayscale using OpenCV
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        return gray_image
            
    def calculate_mse(self, img1, img2):
        # img1_gray = self.convert_to_gray(img1)
        # img2_gray = self.convert_to_gray(img2)  # Assuming img2 is already in grayscale
        
        err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
        err = err // (img1.shape[0] * img2.shape[1])
        return err



import cv2
import numpy as np

class MSE_LAB:
    def __init__(self):
        pass
    
    def convert_to_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def calculate_mse(self, img1, img2):
        # img1_gray = self.convert_to_gray(img1)
        
        mse = np.mean((img1 - img2)**2)
        return mse



import numpy as np
import cv2

class MSE_A:
    def __init__(self):
        pass

    def convert_to_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def mse_a(self, au_image, ela_image):
        # au_gray = self.convert_to_gray(au_image)
        ela_image_float = ela_image.astype(np.float32)

        # Calculate the squared difference of magnitudes
        diff = np.abs(np.abs(au_image) - np.abs(ela_image_float))**2

        # Calculate the mean squared error
        mse = np.mean(diff)

        return mse



import cv2
import numpy as np

class MSE_P:
    def __init__(self):
        pass
    
    def preprocess_images(self, img1, img2):
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        return img1_gray, img2

    def mse_p(self, img1, img2):
        # img1_gray, img2 = self.preprocess_images(img1, img2)

        # Convert images to complex numbers by applying FFT
        img1_fft = np.fft.fft2(img1)
        img2_fft = np.fft.fft2(img2)

        # Calculate the difference of phases
        diff = np.angle(img1_fft) - np.angle(img2_fft)

        # Calculate the mean squared error
        mse = np.mean(diff**2)

        return mse




import cv2
import numpy as np

class Median_M:
    # class Variable
    block_size = 8

    def __init__(self, au_image, ela_image):
        self.au_image = au_image
        self.ela_image = ela_image
        # self.au_gray = self.convert_to_gray(au_image)

    def convert_to_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def preprocess_images(self, img1, img2):
        max_dim = max(img1.shape[0], img2.shape[0], img1.shape[1], img2.shape[1])
        resized_img1 = cv2.resize(img1, (max_dim, max_dim))
        resized_img2 = cv2.resize(img2, (max_dim, max_dim))
        return resized_img1, resized_img2

    def calculate_Jm(self):
        # self.au_gray, self.ela_image = self.preprocess_images(self.au_gray, self.ela_image)
        
        height, width = self.au_image.shape[:2]
        num_blocks_height = height // Median_M.block_size
        num_blocks_width = width // Median_M.block_size
        Jm_values = []

        for i in range(num_blocks_height):
            for j in range(num_blocks_width):
                start_row = i * Median_M.block_size
                end_row = start_row + Median_M.block_size
                start_col = j * Median_M.block_size
                end_col = start_col + Median_M.block_size

                original_block = self.au_image[start_row:end_row, start_col:end_col]
                modified_block = self.ela_image[start_row:end_row, start_col:end_col]

                distortion = np.abs(modified_block - original_block)
                median_distortion = np.median(distortion)
                Jm_values.append(median_distortion)

        return np.median(Jm_values)



import cv2
import numpy as np

class MEDIAN_P:
    # class Variable
    block_size = 8

    def __init__(self, au_image, ela_image):
        self.au_image = au_image
        self.ela_image = ela_image
        # self.au_gray = self.convert_to_gray(au_image)

    def convert_to_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def preprocess_images(self, img1, img2):
        max_dim = max(img1.shape[0], img2.shape[0], img1.shape[1], img2.shape[1])
        resized_img1 = cv2.resize(img1, (max_dim, max_dim))
        resized_img2 = cv2.resize(img2, (max_dim, max_dim))
        return resized_img1, resized_img2

    def calculate_Jphi(self):
        # self.au_gray, self.ela_image = self.preprocess_images(self.au_gray, self.ela_image)
        
        height, width = self.au_image.shape[:2]
        num_blocks_height = height // MEDIAN_P.block_size
        num_blocks_width = width // MEDIAN_P.block_size
        Jphi_values = []

        for i in range(num_blocks_height):
            for j in range(num_blocks_width):
                start_row = i * MEDIAN_P.block_size
                end_row = start_row + MEDIAN_P.block_size
                start_col = j * MEDIAN_P.block_size
                end_col = start_col + MEDIAN_P.block_size

                block_image = self.au_image[start_row:end_row, start_col:end_col]
                block_modified_image = self.ela_image[start_row:end_row, start_col:end_col]

                distortion = np.angle(np.fft.fft2(block_modified_image) - np.fft.fft2(block_image))
                median_distortion = np.median(distortion)

                Jphi_values.append(median_distortion)

        return np.median(Jphi_values)


import cv2
import numpy as np

class HVS1:
    def __init__(self, img1, img2):
        self.img1, self.img2 = img1, img2

    def preprocessed_images(self, img1, img2):
        # U: function to simulate HVS effect, takes an image array and returns a transformed array
        # grayscale: boolean flag to convert images to grayscale (default: False)

        # Convert images to grayscale if needed
        if len(img1.shape) == 3:  # Convert img1 to grayscale if it's not
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1

        img2_gray = img2  # img2 is already grayscale

        # Resize images to the maximum width and height
        max_dim = max(img1_gray.shape[0], img2_gray.shape[0], img1_gray.shape[1], img2_gray.shape[1])
        resized_img1 = cv2.resize(img1_gray, (max_dim, max_dim))
        resized_img2 = cv2.resize(img2_gray, (max_dim, max_dim))

        return resized_img1, resized_img2

    def U(self, x):
        # U function that applies contrast stretching and x is input image array
        min_val = np.min(x)
        max_val = np.max(x)
        
        # Check if the range of values is non-zero
        if max_val != min_val:
            stretched = (x - min_val) / (max_val - min_val)
        else:
            stretched = x

        return stretched

    def calculate_h1(self):
        K = len(self.img1)  # number of distorted images or channels
        N = self.img1.shape[0] * self.img1.shape[1]  # number of pixels in each image

        h1_sum = 0.0  # initialize sum of absolute differences

        for k in range(K):
            # preprocess original and distorted images with U function
            original_u = self.U(self.img1[k])
            distorted_u = self.U(self.img2[k])

            # calculate sum of absolute differences between U(Ck(i,j)) and U(Ĉk(i,j))
            diff_sum = np.sum(np.abs(original_u - distorted_u))

            # calculate average absolute difference and add to H1 sum
            h1_sum += diff_sum / (N * (np.sum(original_u) / N))

        # calculate final H1 value as average over K distorted images
        H1 = h1_sum / K

        return H1

import numpy as np
from PIL import Image
from skimage.transform import resize

class HVS2:
    def __init__(self, img1, img2):
        self.img1, self.img2 = img1,img2

    def preprocessed_images(self, img1, img2):
        max_height = max(img1.shape[0], img2.shape[0])
        max_width = max(img1.shape[1], img2.shape[1])

        resized_img1 = resize(img1, (max_height, max_width))
        resized_img2 = resize(img2, (max_height, max_width))

        return resized_img1, resized_img2

    def U(self, image):
        gamma = 1.5  # Example gamma value

        transformed_image = np.power(image, gamma)
        return transformed_image

    def calculate_h2(self):
        U_distorted = self.U(self.img2)
        U_original = self.U(self.img1)

        # Ensure that both images have the same number of channels
        if len(U_distorted.shape) == 3 and len(U_original.shape) == 2:
            U_original = np.stack([U_original] * 3, axis=-1)
        elif len(U_original.shape) == 3 and len(U_distorted.shape) == 2:
            U_distorted = np.stack([U_distorted] * 3, axis=-1)

        squared_diff = np.square(U_distorted - U_original)
        sum_squared_diff = np.sum(squared_diff)

        squared_original = np.square(U_distorted)
        sum_squared_original = np.sum(squared_original)

        H2 = sum_squared_diff / sum_squared_original

        return H2

import numpy as np
from PIL import Image
from skimage.transform import resize

class HVS3:
    def __init__(self, img1, img2):
        self.img1, self.img2 = img1,img2

    def preprocessed_images(self, img1, img2):
        max_height = max(img1.shape[0], img2.shape[0])
        max_width = max(img1.shape[1], img2.shape[1])

        resized_img1 = resize(img1, (max_height, max_width))
        resized_img2 = resize(img2, (max_height, max_width))

        return resized_img1, resized_img2

    def U(self, image):
        gamma = 1.5  # Example gamma value

        transformed_image = np.power(image, gamma)
        return transformed_image

    def calculate_h3(self):
        U_distorted = self.U(self.img2)
        U_original = self.U(self.img1)

        # Ensure that both images have the same number of channels
        if len(U_distorted.shape) == 3 and len(U_original.shape) == 2:
            U_original = np.stack([U_original] * 3, axis=-1)
        elif len(U_original.shape) == 3 and len(U_distorted.shape) == 2:
            U_distorted = np.stack([U_distorted] * 3, axis=-1)

        squared_diff = np.square(U_distorted - U_original)
        sum_squared_diff = np.sum(squared_diff)

        squared_original = np.square(U_distorted)
        sum_squared_original = np.sum(squared_original)

        H3 = sum_squared_diff / sum_squared_original

        return H3

    

from skimage.metrics import structural_similarity as ssim

class SSIM:
    def __init__(self, image1, image2):
        self.img1, self.img2 = image1, image2
        # self.preprocess()

    def preprocess(self):
        # Convert images to grayscale if needed
        if len(self.img1.shape) == 3:  # Convert img1 to grayscale if it's not
            self.img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)

        if len(self.img2.shape) == 3:  # Convert img2 to grayscale if it's not
            self.img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)

        # Resize images to the minimum shape
        min_height = min(self.img1.shape[0], self.img2.shape[0])
        min_width = min(self.img1.shape[1], self.img2.shape[1])
        
        self.img1 = cv2.resize(self.img1, (min_width, min_height))
        self.img2 = cv2.resize(self.img2, (min_width, min_height))

    def calculate_ssim(self):
        # Compute the SSIM score between the images
        score = ssim(self.img1, self.img2, multichannel=False)  # Grayscale images, so multichannel=False
        return score


import cv2
import numpy as np

class NCC:
    def __init__(self, image1, image2):
        self.img1, self.img2 = image1, image2
        # self.preprocess()

    def preprocess(self):
        # Convert images to grayscale if needed
        if len(self.img1.shape) == 3:  # Convert img1 to grayscale if it's not
            self.img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)

        if len(self.img2.shape) == 3:  # Convert img2 to grayscale if it's not
            self.img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)

        # Resize images to the same dimensions
        self.img1 = cv2.resize(self.img1, (self.img2.shape[1], self.img2.shape[0]), interpolation=cv2.INTER_LINEAR)

    def ncc(self):
        # Convert images to float32
        image1 = self.img1.astype(np.float32)
        image2 = self.img2.astype(np.float32)

        # Calculate mean and standard deviation of each image
        mean1 = np.mean(image1)
        mean2 = np.mean(image2)
        std1 = np.std(image1)
        std2 = np.std(image2)

        # Subtract mean and divide by standard deviation for each image
        image1_norm = (image1 - mean1) / std1
        image2_norm = (image2 - mean2) / std2

        # Calculate NCC
        ncc_value = np.sum(image1_norm * image2_norm) / (image1.size * std1 * std2)

        return ncc_value


import cv2
from skimage.metrics import mean_squared_error, structural_similarity

class IF:
    def __init__(self, image1, image2):
        self.img1, self.img2 = image1,image2

    def preprocess_images(self, image1, image2):
        # Convert to grayscale if needed
        if len(image1.shape) == 3:
            image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        else:
            image1_gray = image1

        # Resize images to have the same dimensions
        max_height = max(image1_gray.shape[0], image2.shape[0])
        max_width = max(image1_gray.shape[1], image2.shape[1])

        img1_resized = cv2.resize(image1_gray, (max_width, max_height))
        img2_resized = cv2.resize(image2, (max_width, max_height))

        return img1_resized, img2_resized

    def image_fidelity(self):
        mse = mean_squared_error(self.img1, self.img2)
        ssim = structural_similarity(self.img1, self.img2, win_size=7, multichannel=False)

        # Return the AVERAGE of MSE and SSIM values
        # Lower MSE and higher SSIM values generally indicate better image fidelity and a closer resemblance between the images.
        return (mse + ssim) / 2


import cv2

class Histogram:
    def __init__(self, image1, image2):
        self.img1, self.img2 = image1,image2

    def preprocess_images(self, image1, image2):
        # Convert to grayscale if needed
        if len(image1.shape) == 3:
            image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        else:
            image1_gray = image1

        image2_gray = image2  # ela_image is already grayscale

        return image1_gray, image2_gray

    def calculate_similarity(self):
        # Calculate histograms
        original_hist = cv2.calcHist([self.img1], [0], None, [256], [0, 256])
        tampered_hist = cv2.calcHist([self.img2], [0], None, [256], [0, 256])

        # Normalize histograms
        original_hist = cv2.normalize(original_hist, original_hist).flatten()
        tampered_hist = cv2.normalize(tampered_hist, tampered_hist).flatten()

        # Compare histograms using histogram intersection metric
        intersection = cv2.compareHist(original_hist, tampered_hist, cv2.HISTCMP_INTERSECT)

        return intersection


def generate_data(input_image,ela_image):
    
    data={}

    mse_calculator = MSE()
    mse_value = mse_calculator.calculate_mse(input_image, ela_image)
    data['mse']=[mse_value]
    
    mselab_calculator = MSE_LAB()
    mse_value = mselab_calculator.calculate_mse(input_image, ela_image)
    data['mse_lab']=[mse_value]
    
    mse_mag_calculator = MSE_A()
    mse_a_value = mse_mag_calculator.mse_a(input_image, ela_image)
    data['mse_magnitude']=[mse_a_value]
    
    mse_phase_calculator = MSE_P()
    mse_p_value = mse_a_value = mse_phase_calculator.mse_p(input_image, ela_image)
    data['mse_phase']=[mse_p_value]
    
    
    mse_m_calculator = Median_M(input_image, ela_image)
    med_m_value = mse_m_calculator.calculate_Jm()
    data['med_magnitude']=[med_m_value]
    
    mse_p_calculator = MEDIAN_P(input_image, ela_image)
    med_p_value = mse_p_calculator.calculate_Jphi()
    data['med_phase']=[med_p_value]
    
    hvs1_calculator = HVS1(input_image, ela_image)
    h1_value = hvs1_calculator.calculate_h1()
    data['hvs1']=[h1_value]
    
    h2_calculator = HVS2(input_image, ela_image)
    h2_value = h2_calculator.calculate_h2()
    data['hvs2']=[h2_value]
    
    h3_calculator = HVS3(input_image, ela_image)
    h3_value = h3_calculator.calculate_h3()
    data['hvs3']=[h3_value]
    
    ssim_calculator = SSIM(input_image, ela_image)
    ssim_value = ssim_calculator.calculate_ssim()
    data["ssim"]=[ssim_value]
    
    ncc_calculator = NCC(input_image, ela_image)
    ncc_value = ncc_calculator.ncc()
    data['ncc']=[ncc_value]
    
    if_calculator = IF(input_image, ela_image)
    if_value = if_calculator.image_fidelity()
    data['if_value']=[if_value]
    
    histogram_calculator = Histogram(input_image, ela_image)
    histogram_value = histogram_calculator.calculate_similarity()
    data['histogram']=[histogram_value]

    return data



