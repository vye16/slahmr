"""
Modified code from https://github.com/nwojke/deep_sort
"""

import numpy as np

class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.
    """

    def __init__(self, detection_data):
        self.detection_data           = detection_data
        self.tlwh                     = np.asarray(detection_data['bbox'], dtype=np.float64)

        image_size                    = self.detection_data['size']
        img_height, img_width         = float(image_size[0]), float(image_size[1])
        new_image_size                = max(img_height, img_width)
        delta_w                       = new_image_size - img_width
        delta_h                       = new_image_size - img_height
        top, _, left, _               = delta_h//2, delta_h-(delta_h//2), delta_w//2, delta_w-(delta_w//2)
        self.xy                       = [((self.tlwh[0]+self.tlwh[2])/2.0+left)/new_image_size, ((self.tlwh[1]+self.tlwh[3])/2.0+top)/new_image_size]
        self.detection_data['xy']     = self.xy
        self.detection_data['scale']  = float(max(self.detection_data['scale']))/new_image_size
        
    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret      = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret      = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2]  /= ret[3]
        return ret
