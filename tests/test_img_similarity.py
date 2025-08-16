import numpy as np
from pilar.utils.image_processor import ImageProcessor


def img_similarity_original(img1, img2):
    data1 = img1.flatten()
    data2 = img2.flatten()
    return np.sum(np.isclose(data1, data2, atol=50)) / len(data1)


def test_img_similarity_equivalence():
    for _ in range(5):
        img1 = np.random.randint(0, 256, (60, 80), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (60, 80), dtype=np.uint8)
        expected = img_similarity_original(img1, img2)
        actual = ImageProcessor.img_similarity(img1, img2)
        assert np.isclose(actual, expected)
