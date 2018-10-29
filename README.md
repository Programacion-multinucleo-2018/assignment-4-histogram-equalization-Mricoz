# Assignment 4: Histogram Equalization

Assignment No 4 for the multi-core programming course. Implement histogram equalization for a gray scale image in CPU and GPU. The result of applying the algorithm to an image with low contrast can be seen in Figure 1:

![Figure 1](Images/histogram_equalization.png)
<br/>Figure 1: Expected Result.

The programs have to do the following:

1. Using Opencv, load and image and convert it to grayscale.
2. Calculate de histogram of the image.
3. Calculate the normalized sum of the histogram.
4. Create an output image based on the normalized histogram.
5. Display both the input and output images.

Test your code with the different images that are included in the *Images* folder. Include the average calculation time for both the CPU and GPU versions, as well as the speedup obtained, in the Readme.

Rubric:

1. Image is loaded correctly.
2. The histogram is calculated correctly using atomic operations.
3. The normalized histogram is correctly calculated.
4. The output image is correctly calculated.
5. For the GPU version, used shared memory where necessary.
6. Both images are displayed at the end.
7. Calculation times and speedup obtained are incuded in the Readme.

**Grade: 100**

# Results in (ms):

| CPU        | GPU       |
| -----------|:---------:|
| 242.878052 | 0.032582  |
| 242.273829 | 0.077432  |
| 243.023315 | 0.039863  |
| 243.099411 | 0.041444  |
| 242.895081 | 0.041020  |
| 242.861252 | 0.032526  |
| 242.880905 | 0.047476  |
| 243.390350 | 0.039982  |
| 242.826920 | 0.052203  |
| 243.164764 | 0.034996  |
| 242.802673 | 0.034918  |
| 242.906250 | 0.051203  |
| 242.951477 | 0.051213  |
| 243.025391 | 0.051405  |
| 243.049149 | 0.041161  |
| 242.803375 | 0.038845  |
| 242.925522 | 0.039390  |
| 242.892639 | 0.030915  |
| 243.055557 | 0.051035  |
| 242.952835 | 0.049149  |

### Average CPU: 242.93293735
### Average GPU: 0.0439379

## Speedup: 5529.00655
