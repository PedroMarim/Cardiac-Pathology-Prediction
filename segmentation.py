import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib

from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage.exposure import match_histograms
from skimage.exposure import equalize_adapthist
from scipy.ndimage import binary_fill_holes, label
from sklearn.linear_model import LinearRegression
from skimage import morphology as morpho


def compute_seed(data_ED, data_ES, hough_radii_range=(15, 50)):
    """
    Compute initial seed point from the same slice in ED and ES 3D images.
    Args:
        data_ED: 3D NumPy array [height, width, slices] for End Diastole.
        data_ES: 3D NumPy array [height, width, slices] for End Systole.
        hough_radii_range: range of radii to test in Hough circle detection.
    Returns:
        slice_idx: index of the used slice.
        seed_point: (x, y) coordinates of the detected center.
        radius: radius of the detected LV cavity.
    """
    slice_idx = data_ED.shape[2] // 2
    slice_ED = data_ED[:, :, slice_idx]
    slice_ES = data_ES[:, :, slice_idx]

    # Difference image
    diff_image = np.abs(slice_ED - slice_ES)
    diff_image = (diff_image - diff_image.min()) / (diff_image.max() - diff_image.min())

    # Edge detection
    edges = canny(diff_image, sigma=2)

    # Hough Transform
    hough_radii = np.arange(hough_radii_range[0], hough_radii_range[1], 1)
    hough_res = hough_circle(edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    if len(cx) > 0:
        seed_point = (cx[0], cy[0])  # (x, y)
        return slice_idx, seed_point, radii[0]
    else:
        return slice_idx, None, None


def plot_heart_slice_with_seed(data_ED, slice_idx, seed_point, radius):
    """
    Plota a fatia do coração com o ponto semente e o círculo detectado.
    """

    img_slice = data_ED[:, :, slice_idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_slice, cmap='gray')
    ax.set_title(f"Slice {slice_idx} with Seed Point")

    if seed_point is not None:
        cx, cy = int(seed_point[0]), int(seed_point[1])
        rr, cc = circle_perimeter(cy, cx, int(radius), shape=img_slice.shape)
        img_overlay = img_slice.copy()
        img_overlay[rr, cc] = img_overlay.max()  # destaca o círculo
        ax.plot(cx, cy, 'ro', label='Seed Point')
        ax.imshow(img_overlay, cmap='gray')

    ax.legend()
    plt.axis('off')
    plt.show()

    return


def compute_seed_hist_matching(data_ED, data_ES, hough_radii_range=(15, 50), reference_image=None):
    """
    Compute initial seed point from the same slice in ED and ES 3D images.
    Optionally apply histogram transfer using a reference image.
    
    Args:
        data_ED: 3D NumPy array [height, width, slices] for End Diastole.
        data_ES: 3D NumPy array [height, width, slices] for End Systole.
        hough_radii_range: range of radii to test in Hough circle detection.
        reference_image: 2D NumPy array to match histograms against (optional).
    
    Returns:
        slice_idx: index of the used slice.
        seed_point: (x, y) coordinates of the detected center.
        radius: radius of the detected LV cavity.
    """
    slice_idx = data_ED.shape[2] // 2
    slice_ED = data_ED[:, :, slice_idx]
    slice_ES = data_ES[:, :, slice_idx]

    # Apply histogram matching if reference is provided
    if reference_image is not None:
        slice_ED = match_histograms(slice_ED, reference_image)
        slice_ES = match_histograms(slice_ES, reference_image)

    # Difference image
    diff_image = np.abs(slice_ED - slice_ES)
    diff_image = (diff_image - diff_image.min()) / (diff_image.max() - diff_image.min())

    # Edge detection
    edges = canny(diff_image, sigma=2)

    # Hough Transform
    hough_radii = np.arange(hough_radii_range[0], hough_radii_range[1], 2)
    hough_res = hough_circle(edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    if len(cx) > 0:
        seed_point = (cx[0], cy[0])  # (x, y)
        return slice_idx, seed_point, radii[0]
    else:
        return slice_idx, None, None
    

def compute_seed_equalization(data_ED, data_ES, hough_radii_range=(15, 50), case_id=None):
    slice_idx = data_ED.shape[2] // 2
    slice_ED = data_ED[:, :, slice_idx]
    slice_ES = data_ES[:, :, slice_idx]

    apply_eq = True

    if apply_eq:
        # Normalizar para [0, 1] antes de aplicar equalize_adapthist
        slice_ED = (slice_ED - slice_ED.min()) / (slice_ED.max() - slice_ED.min())
        slice_ES = (slice_ES - slice_ES.min()) / (slice_ES.max() - slice_ES.min())

        slice_ED = equalize_adapthist(slice_ED, clip_limit=0.03)
        slice_ES = equalize_adapthist(slice_ES, clip_limit=0.03)
        

    # Difference image
    diff_image = np.abs(slice_ED - slice_ES)
    diff_image = (diff_image - diff_image.min()) / (diff_image.max() - diff_image.min())

    # Edge detection
    edges = canny(diff_image, sigma=2)

    # Hough Transform
    hough_radii = np.arange(hough_radii_range[0], hough_radii_range[1], 2)
    hough_res = hough_circle(edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    if len(cx) > 0:
        seed_point = (cx[0], cy[0])
        return slice_idx, seed_point, radii[0]
    else:
        return slice_idx, None, None
    

def compute_seeds(data_ED, data_ES, hough_radii_range=(13, 50), num_peaks=3):
    """
    Compute up to `num_peaks` candidate seed circles from the same slice in ED and ES 3D images.
    
    Returns:
        slice_idx: index of the used slice.
        circles: list of (x, y, radius) tuples for the top Hough circles.
    """
    slice_idx = data_ED.shape[2] // 2
    slice_ED = data_ED[:, :, slice_idx]
    slice_ES = data_ES[:, :, slice_idx]

    # Difference image
    diff_image = np.abs(slice_ED - slice_ES)
    diff_image = (diff_image - diff_image.min()) / (diff_image.max() - diff_image.min())

    # Edge detection
    edges = canny(diff_image, sigma=2)

    # Hough Transform
    hough_radii = np.arange(hough_radii_range[0], hough_radii_range[1], 2)
    hough_res = hough_circle(edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=num_peaks)

    # Retornar os círculos encontrados
    circles = []
    for x, y, r in zip(cx, cy, radii):
        circles.append((x, y, r))

    return slice_idx, circles


def plot_heart_slice_with_circles(data_ED, slice_idx, circles):
    """
    Plota a fatia do coração com múltiplos círculos detectados.
    Cada círculo é representado por (x, y, radius).
    """
    import matplotlib.pyplot as plt
    from skimage.draw import circle_perimeter

    img_slice = data_ED[:, :, slice_idx]
    img_overlay = img_slice.copy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_overlay, cmap='gray')
    ax.set_title(f"Slice {slice_idx} with Detected Circles")

    # Cores diferentes para cada círculo
    colors = ['r', 'g', 'b', 'y', 'c', 'm']

    for i, (x, y, r) in enumerate(circles):
        cx, cy = int(x), int(y)
        rr, cc = circle_perimeter(cy, cx, int(r), shape=img_slice.shape)
        img_overlay[rr, cc] = img_overlay.max()  # destaca borda
        ax.plot(cx, cy, marker='o', color=colors[i % len(colors)], label=f'Seed {i+1}')

    ax.imshow(img_overlay, cmap='gray')
    ax.legend()
    ax.axis('off')
    plt.show()

    return


def select_lowest_circle(circles):
    if not circles:
        return None
    return max(circles, key=lambda c: c[1]) 


def compute_seeds_radii_adapt(data_ED, data_ES, hough_radii_range=(13, 50), num_peaks=3):
    """
    Compute up to `num_peaks` candidate seed circles from the same slice in ED and ES 3D images.
    Uses CLAHE adaptively if intensity stats are abnormal.
    
    Returns:
        slice_idx: index of the used slice.
        circles: list of (x, y, radius) tuples for the top Hough circles.
    """
    slice_idx = data_ED.shape[2] // 2
    slice_ED = data_ED[:, :, slice_idx]
    slice_ES = data_ES[:, :, slice_idx]

    # Difference image
    diff_image = np.abs(slice_ED - slice_ES)
    diff_image = (diff_image - diff_image.min()) / (diff_image.max() - diff_image.min())

    h, w = diff_image.shape
    min_dim = min(h, w)
    min_radius = max(13, int(0.07 * h))
    max_radius = 50
    hough_radii = np.arange(min_radius, max_radius, 1)

    # Edge detection
    edges = canny(diff_image, sigma=2)

    # Hough Transform
    #hough_radii = np.arange(hough_radii_range[0], hough_radii_range[1], 2)
    hough_res = hough_circle(edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=num_peaks)

    # Retornar os círculos encontrados
    circles = []
    for x, y, r in zip(cx, cy, radii):
        circles.append((x, y, r))

    return slice_idx, circles


def region_growing_from_seed(image, seed, max_diff=10, print_=True):

    h, w = image.shape
    visited = np.zeros_like(image, dtype=bool)
    mask = np.zeros_like(image, dtype=bool)

    x0, y0 = int(seed[0]), int(seed[1])
    region_values = [image[y0, x0]]
    threshold = max_diff

    # Seed value smoothed by a 3x3 gaussian kernel
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]) / 16.0
    image[y0, x0] = np.sum(image[y0-1:y0+2, x0-1:x0+2] * kernel)

    stack = [(x0, y0)]
    visited[y0, x0] = True
    mask[y0, x0] = True

    step_counter = 0
    exploded = False

    while stack:
        x, y = stack.pop()
        current_mean = np.mean(region_values)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                xn, yn = x + dx, y + dy
                if 0 <= xn < w and 0 <= yn < h and not visited[yn, xn]:
                    intensity = image[yn, xn]
                    if abs(intensity - current_mean) <= threshold:
                        visited[yn, xn] = True
                        mask[yn, xn] = True
                        region_values.append(intensity)
                        stack.append((xn, yn))

        step_counter += 1
        if step_counter % 100 == 0:
            region_size = np.sum(mask)
            if region_size > 3000:
                if print_:
                    print("Exploded! Region too large")
                exploded = True
                break
    

    # mask = binary_fill_holes(mask)
    # labeled_mask, _ = label(mask)
    # labels, counts = np.unique(labeled_mask, return_counts=True)

    # if len(labels) > 1:
    #     # Ignora o fundo (label 0) e pega o maior entre os demais
    #     valid_labels = labels[1:]
    #     valid_counts = counts[1:]
    #     largest_label = valid_labels[np.argmax(valid_counts)]
    #     final_mask = labeled_mask == largest_label
    # elif len(labels) == 1 and labels[0] != 0:
    #     # Apenas uma região válida (sem fundo)
    #     final_mask = labeled_mask == labels[0]
    # else:
    #     # Nada segmentado de verdade, usar a máscara bruta
    #     final_mask = mask

    final_mask = mask

    mean_intensity = image[final_mask].mean() if final_mask.any() else 0
    std_intensity = image[final_mask].std() if final_mask.any() else 0

    return final_mask, mean_intensity, std_intensity, exploded

def compensate_coil_sensitivity(image, mask):
    """
    Step 3: Fit a planar surface to the intensities of the full-blood region (mask)
    and subtract this bias from the entire image to compensate for coil sensitivity.

    Args:
        image: 2D numpy array (original MR slice)
        mask: 2D boolean numpy array (region of full-blood from step 2)

    Returns:
        corrected_image: 2D numpy array with coil bias corrected
        plane: fitted plane as a 2D array (for visualization/debug if needed)
    """
    y_idx, x_idx = np.where(mask)
    intensities = image[y_idx, x_idx]

    # Features: x, y coordinates
    X = np.column_stack((x_idx, y_idx))  # shape (n_samples, 2)

    # Fit a plane: z = a*x + b*y + c
    model = LinearRegression().fit(X, intensities)
    a, b = model.coef_
    c = model.intercept_

    # Generate the fitted plane over the whole image
    height, width = image.shape
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    plane = a * xx + b * yy + c

    # Subtract the fitted plane (bias field) from the image
    corrected_image = image - plane

    return corrected_image, plane

def estimate_myocardium_mean(image, seed, max_diff_list, plot=False):

    volumes = [-100000, -20000]
    masks = []
    means = []

    for i, max_diff in enumerate(max_diff_list):
        mask, _, _, leaked = region_growing_from_seed(image, seed, max_diff=max_diff, print_=plot)
        volumes.append(np.sum(mask))
        masks.append(mask)
        if plot:
            print(f"Mask {i+1} volume: {volumes[-1]}")
            plt.imshow(mask, cmap='gray')
            plt.show()
            #print(image[mask].mean())
        means.append(image[mask].mean() if mask.any() else 0)
        if (volumes[i+2] - volumes[i+1]) > 2 * (volumes[i+1] - volumes[i]):
            leaked = True
            if plot:
                print("Leaked!")
            break
        if leaked:
            if plot:
                print("Area too large, breaking")
            break

    if len(masks) < 2:
        print("Not enough masks, returning None")
        return None, None, None, None, None, masks
    
    lv_mask = masks[-2] 
    
    lv_mean = image[lv_mask].mean() if lv_mask.any() else 0
    lv_std = image[lv_mask].std() if lv_mask.any() else 0

    dilated_mask = morpho.binary_dilation(lv_mask, morpho.disk(4))

    dif_mask = dilated_mask & ~lv_mask

    # plt.imshow(dif_mask, cmap='gray')
    # plt.show()

    myoc_mean = image[dif_mask].mean() if dif_mask.any() else 0
    myoc_std = image[dif_mask].std() if dif_mask.any() else 0

    #return lv_mask, lv_mean, lv_std, myoc_mean, myoc_std, masks

    return masks[-2], lv_mean, lv_std, myoc_mean, myoc_std, masks


def normalize_image(image):
    """
    Normalize the image to the range [0, 255].
    """
    image = image - image.min()
    image = image / image.max()
    image = (image * 255)
    return image


def abs_threshold_region_growing(image, seed, abs_threshold):

    h, w = image.shape
    visited = np.zeros_like(image, dtype=bool)
    mask = np.zeros_like(image, dtype=bool)

    x0, y0 = int(seed[0]), int(seed[1])

    stack = [(x0, y0)]
    visited[y0, x0] = True
    mask[y0, x0] = True

    while stack:
        x, y = stack.pop()
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                xn, yn = x + dx, y + dy
                if 0 <= xn < w and 0 <= yn < h and not visited[yn, xn]:
                    intensity = image[yn, xn]
                    if intensity >= abs_threshold:
                        visited[yn, xn] = True
                        mask[yn, xn] = True
                        stack.append((xn, yn))


    # Pós-processamento: preencher buracos e pegar maior componente conexa
    mask = binary_fill_holes(mask)
    labeled_mask, _ = label(mask)
    labels, counts = np.unique(labeled_mask, return_counts=True)

    if len(labels) > 1:
        # Ignora o fundo (label 0) e pega o maior entre os demais
        valid_labels = labels[1:]
        valid_counts = counts[1:]
        largest_label = valid_labels[np.argmax(valid_counts)]
        final_mask = labeled_mask == largest_label
    elif len(labels) == 1 and labels[0] != 0:
        # Apenas uma região válida (sem fundo)
        final_mask = labeled_mask == labels[0]
    else:
        # Nada segmentado de verdade, usar a máscara bruta
        final_mask = mask

    mean_intensity = image[final_mask].mean() if final_mask.any() else 0
    std_intensity = image[final_mask].std() if final_mask.any() else 0

    return final_mask

