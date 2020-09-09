import numpy as np
import os
import numpy as np
import imp
import torch 
import torchvision.ops
import torch.nn as nn
import torch.nn.functional
import PIL.Image

import time
import os
import sys
from math import ceil
import torch
import numpy as np

import vipy.image
import vipy.object

#import sys
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'detection'))
#from config import cfg

from pycollector.model.face.faster_rcnn import FasterRCNN, FasterRCNN_MMDNN

DIM_THRESH = 15
CONF_THRESH = 0.5
NMS_THRESH = 0.15
FUSION_THRESH = 0.60
VERBOSE = False

def log_info(s):
    if VERBOSE:
        print(s)


class FaceRCNN(object):
    "Wrapper for PyTorch RCNN detector"
    def __init__(self, model_path=None, gpu_index=None, conf_threshold=None, rotate_flags=None,
                 rotate_thresh=None, fusion_thresh=None, test_scales=800, max_size=1300, as_scene=False):
        
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/detection/resnet-101_faster_rcnn_ohem_iter_20000.pth')
            if not os.path.exists(model_path):
                d = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
                raise ValueError('[pycollector.detection]: FaceRCNN detection models not downloaded; Run "cd %s; ./download_models.sh"' % d)
    
        # This logs the contents of the detector_params dict, along with the other values that we passed.
        #log_info(f"Params=[{', '.join((chr(34) + k + chr(34) + ': ' + str(v)) for k, v in detector_params)}], threshold=[{conf_threshold}], "
        #         "rotate=[{rotate_flags}], rotate_thresh=[{rotate_thresh}], fusion_thresh=[{fusion_thresh}]")
        log_info(f"model=[{model_path}], gpu=[{gpu_index}], threshold=[{conf_threshold}], "
                 "rotate=[{rotate_flags}], rotate_thresh=[{rotate_thresh}], fusion_thresh=[{fusion_thresh}]")

        # Originally stored in config.py, hardcoded defaults here
        self.cfg = {'TRAIN': {'SCALES': [1024], 'MAX_SIZE': 1024, 'IMS_PER_BATCH': 1, 'BATCH_SIZE': 64, 'FG_FRACTION': 0.4, 'FG_THRESH': 0.5, \
                              'BG_THRESH_HI': 0.5, 'BG_THRESH_LO': -0.1, 'USE_FLIPPED': True, 'BBOX_REG': True, 'BBOX_THRESH': 0.5, \
                              'SNAPSHOT_ITERS': 5000, 'SNAPSHOT_INFIX': '', 'USE_PREFETCH': False, 'BBOX_NORMALIZE_TARGETS': True, \
                              'BBOX_INSIDE_WEIGHTS': [1.0, 1.0, 1.0, 1.0], 'BBOX_NORMALIZE_TARGETS_PRECOMPUTED': False, 'BBOX_NORMALIZE_MEANS': \
                              [0.0, 0.0, 0.0, 0.0], 'BBOX_NORMALIZE_STDS': [0.1, 0.1, 0.2, 0.2], 'PROPOSAL_METHOD': 'selective_search', 'ASPECT_GROUPING': \
                              True, 'HAS_RPN': False, 'RPN_POSITIVE_OVERLAP': 0.7, 'RPN_NEGATIVE_OVERLAP': 0.3, 'RPN_CLOBBER_POSITIVES': False, \
                              'RPN_FG_FRACTION': 0.5, 'RPN_BATCHSIZE': 256, 'RPN_NMS_THRESH': 0.7, 'RPN_PRE_NMS_TOP_N': 12000, 'RPN_POST_NMS_TOP_N': \
                              2000, 'RPN_MIN_SIZE': 3, 'RPN_BBOX_INSIDE_WEIGHTS': [1.0, 1.0, 1.0, 1.0], 'RPN_POSITIVE_WEIGHT': -1.0}, \
                    'TEST': {'SCALES': [800], 'MAX_SIZE': 1300, 'NMS': 0.3, 'SVM': False, 'BBOX_REG': True, 'HAS_RPN': True, 'PROPOSAL_METHOD': \
                             'selective_search', 'RPN_NMS_THRESH': 0.7, 'RPN_PRE_NMS_TOP_N': 6000, 'RPN_POST_NMS_TOP_N': 300, 'RPN_MIN_SIZE': 3}, \
                    'DEDUP_BOXES': 0.0625, 'PIXEL_MEANS': np.array([[[102.9801, 115.9465, 122.7717]]]), 'RNG_SEED': 3, 'EPS': 1e-14, \
                    'ROOT_DIR': None, 'DATA_DIR': None, 'MODELS_DIR': None, 'MATLAB': 'matlab', 'EXP_DIR': 'default', 'USE_GPU_NMS': True, 'GPU_ID': 0}

        # Now do any setup required by the parameters that the framework
        # itself won't handle.
        # import pdb; pdb.set_trace()
        if gpu_index is not None and gpu_index >= 0:
            dev = torch.device(gpu_index)
            self.cfg['GPU_ID'] = gpu_index
        else:
            dev = torch.device("cpu")

        self.cfg['TEST']['HAS_RPN'] = True  # Use RPN for proposals
        self.cfg['TEST']['SCALES'] = (test_scales,)
        self.cfg['TEST']['MAX_SIZE'] = max_size
        #self.net = FasterRCNN_MMDNN(model_path, dev)  # model_path is directory
        self.net = FasterRCNN(dev)
        self.net.load_state_dict(torch.load(model_path))
        if conf_threshold is None:
            self.conf_threshold = CONF_THRESH
        else:
            self.conf_threshold = conf_threshold
        if rotate_flags is None:
            self.rotate_flags = 0
        else:
            self.rotate_flags = rotate_flags
        if rotate_thresh is None:
            self.rotate_thresh = conf_threshold
        else:
            self.rotate_thresh = rotate_thresh
        if fusion_thresh is None:
            self.fusion_thresh = FUSION_THRESH
        else:
            self.fusion_thresh = fusion_thresh
        self.as_scene = as_scene
        log_info('Init success; threshold {}'.format(self.conf_threshold))


    def __call__(self, img, padding=0, min_face_size=DIM_THRESH):
        """Return list of [[x,y,w,h,conf],...] face detection"""
        return self.detect(img, padding=padding, min_face_size=min_face_size)


    def dets_to_scene(img, dets):
        """Convert detections returned from this object to a vipy.image.Scene object"""
        return vipy.image.Scene(array=img, colorspace='rgb', objects=[vipy.object.Detection('face', xmin=bb[0], ymin=bb[1], width=bb[2], height=bb[3], confidence=bb[4]) for bb in dets])

        
    def detect(self, image, padding=0, min_face_size=DIM_THRESH):
        "Run detection on a numpy image, with specified padding and min size"

        # Input must be a np.array(), have a method image.numpy() or is convertible as np.array(image), otherwise error
        if 'numpy' not in str(type(image)) and hasattr(image, 'numpy'):
            image = image.numpy()
        else:
            try:
                image = np.array(image)
            except:
                raise ValueError('Input must be a numpy array')
        
        
        start_time = time.time()
        width = image.shape[1]
        height = image.shape[0]
        # These values will get updated for resizing and padding, so we'll have good numbers
        # for un-rotating bounding boxes where needed
        detect_width = width
        detect_height = height
        color_space = 1 if image.ndim > 2 else 0

        log_info('w/h/cs: %d/%d/%d' %(width, height, color_space))

        img = np.array(image)

        if padding > 0:
            perc = padding / 100.
            padding = int(ceil(min(width, height) * perc))

            # mean bgr padding
            bgr_mean = np.mean(img, axis=(0, 1))
            detect_width = width + padding * 2
            detect_height = height + padding * 2
            pad_im = np.zeros((detect_height, detect_width, 3), dtype=np.uint8)
            pad_im[:, :, ...] = bgr_mean
            pad_im[padding:padding + height, padding:padding + width, ...] = img
            img = pad_im
            log_info('mean padded to w/h: %d/%d' % (img.shape[1], img.shape[0]))
            # cv2.imwrite('debug.png', im)

        if width <= 16 or height <= 16:
            img = np.array(PIL.Image.fromarray(img).resize( (32, 32), PIL.Image.BILINEAR))
            width = img.shape[1]
            height = img.shape[0]

        rotation_angles = []
        if (self.rotate_flags & 1) != 0:
            rotation_angles.append(90)
        if (self.rotate_flags & 2) != 0:
            rotation_angles.append(-90)
        if (self.rotate_flags & 4) != 0:
            rotation_angles.append(180)
        current_rotation = 0

        # parallel arrays: one is list of boxes, per rotation; other is list of scores
        det_lists = []
        box_proposals = None
        im_rotated = img
        while True:
            scores, boxes = self.im_detect(self.net, im_rotated, box_proposals)

            # Threshold on score and apply NMS
            cls_ind = 1
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]

            # Each row of dets is Left, Top, Right, Bottom, score
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            orig_dets = dets.shape
            #keep = nms(dets, NMS_THRESH, force_cpu=False)
            keep = self.net._nms(dets, NMS_THRESH)  # JEBYRNE
            dets = dets[keep, :]
            new_dets = dets.shape
            log_info('Before NMS: {}; after: {}'.format(orig_dets, new_dets))

            # If we just ran the detector on a rotated image, use the rotation threshold
            if current_rotation != 0:
                keep = np.where(dets[:, 4] > self.rotate_thresh)
            else:
                keep = np.where(dets[:, 4] > self.conf_threshold)
            # print 'After filter for rotation {}: keep = {}'.format(current_rotation, keep)
            dets = dets[keep]

            # This is converting the max coords to width and height. The coordinates haven't been
            # unrotated yet--save a bit of energy by thresholding and such first.
            dets[:, 2] = dets[:, 2] - dets[:, 0] + 1
            dets[:, 3] = dets[:, 3] - dets[:, 1] + 1
            if current_rotation != 0:
                # Now unrotate
                # Rotated coordinates are x_rot, y_rot, Wr, Hr
                # Unrotated, X, Y, W, H
                # for +90, width and height swap, top right becomes top left
                #   W = Hr, H = Wr, X = y_rot, Y = (rotated image width) - (x_rot + Wr)
                # for -90, width and height swap, bottom left becomes top left
                #   W = Hr, H = Wr, X = (rotated image height) - (y_rot + Hr), Y = x_rot
                # for 180, width and height same, bottom right becomes top left
                #   W = Wr, H = Hr, X = image width - (x_rot + Wr), Y = image height - (y_rot + Hr)
                if current_rotation == 90:
                    for det in dets:
                        x_rot = det[0]
                        y_rot = det[1]
                        det[0] = y_rot
                        # Image was rotated, so width and height swapped
                        det[1] = detect_height - (x_rot + det[2])
                        det[2], det[3] = det[3], det[2]
                elif current_rotation == -90:
                    for det in dets:
                        x_rot = det[0]
                        y_rot = det[1]
                        # Image was rotated, so width and height swapped
                        det[0] = detect_width - (y_rot + det[3])
                        det[1] = x_rot
                        det[2], det[3] = det[3], det[2]
                elif current_rotation == 180:
                    for det in dets:
                        x_rot = det[0]
                        y_rot = det[1]
                        det[0] = detect_width - (x_rot + det[2])
                        det[1] = detect_height - (y_rot + det[3])

            if padding > 0:
                # Adjust to original coordinates
                dets[:, 0] -= padding
                dets[:, 1] -= padding

                keep = np.where(np.bitwise_and(dets[:, 2] > min_face_size,
                                               dets[:, 3] > min_face_size))
                dets = dets[keep]
            else:
                keep = np.where(np.bitwise_and(dets[:, 2] > min_face_size,
                                               dets[:, 3] > min_face_size))
                dets = dets[keep]
            det_lists.append(dets)
            # Exit the list if we've done all the rotations we need
            if len(rotation_angles) == 0:
                break
            current_rotation = rotation_angles[0]
            rotation_angles = rotation_angles[1:]
            log_info('Rotating to %d' % current_rotation)
            if current_rotation == 90:
                im_rotated = img.transpose(1,0,2)
                im_rotated = np.flipud(im_rotated)
            elif current_rotation == -90:
                im_rotated = img.transpose(1,0,2)
                im_rotated = np.fliplr(im_rotated)
            else:
                # Must be 180
                im_rotated = np.fliplr(np.flipud(img))

                # Now have 1, 3 (0, 90, -90), or 4 (0, 90, -90, 180) elements of det_lists.
        if len(det_lists) > 1:
            return self.select_from_rotated(det_lists, start_time)
        else:
            dets = det_lists[0]
            log_info('Found %d faces' % dets.shape[0])
            log_info('===elapsed %.6f===' % ((time.time() - start_time) * 1000))
            return dets if not self.as_scene else self.dets_to_scene(img, dets)  # [[x,y,w,h,conf], ...]


    def select_from_rotated(self, det_lists, start_time):
        "Given that we tried rotating the image, select the best rotation to use"
        dets = det_lists[0]
        original_dets = dets.shape[0]
        i = 0
        for rot_dets in det_lists[1:]:
            i = i + 1
            log_info('Processing rotated detections from slot %d' % (i))
            # Now iterate over the rows, 1/detection
            for rot_det in rot_dets:
                rot_xmin = rot_det[0]
                rot_ymin = rot_det[1]
                rot_xmax = rot_xmin + rot_det[2]
                rot_ymax = rot_ymin + rot_det[3]
                rot_area = rot_det[2] * rot_det[3]
                matched = False
                best_iou = 0.0
                for det in dets:
                    xmin = det[0]
                    ymin = det[1]
                    xmax = xmin + det[2]
                    ymax = ymin + det[3]
                    intersection_width = min(xmax, rot_xmax) - max(xmin, rot_xmin)
                    intersection_height = min(ymax, rot_ymax) - max(ymin, rot_ymin)
                    if intersection_width > 0 and intersection_height > 0:
                        intersection_area = intersection_width * intersection_height
                        union_area = rot_area + det[2] * det[3] - intersection_area
                        iou = intersection_area / union_area
                        if iou > best_iou:
                            best_iou = iou
                        if iou > self.fusion_thresh:
                            matched = True
                            if rot_det[4] > det[4]:
                                # Rotated detection was better
                                det[0] = rot_det[0]
                                det[1] = rot_det[1]
                                det[2] = rot_det[2]
                                det[3] = rot_det[3]
                                det[4] = rot_det[4]
                            break
                if not matched:
                    # Add this guy, since he had no matches
                    dets = np.vstack((dets, rot_det))
        log_info('Found %d face%s (orig %d)' %
                 (dets.shape[0], '' if dets.shape[0] == 0 else 's', original_dets))
        log_info('===elapsed %.6f===' % ((time.time() - start_time) * 1000))
        return dets

    
    def _get_image_blob(self, im):
        """Converts an image into a network input.

        Arguments:
        im (ndarray): a color image in BGR order

        Returns:
        blob (ndarray): a torch Tensor holding the image. Some transposition might have to occur, because
        we need N, 3, 800, 1205 (say), while the image itself is likely 800, 1205, 3. N is the number of images
        to process (if len(TEST.SCALES) > 1, then it won't be 1).
        im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
        """
        im_orig = im.astype(np.float32, copy=True)
        #im_orig -= self.cfg.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        
        processed_ims = []
        im_scale_factors = []

        for target_size in self.cfg['TEST']['SCALES']:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > self.cfg['TEST']['MAX_SIZE']:
                im_scale = float(self.cfg['TEST']['MAX_SIZE']) / float(im_size_max)

            im = np.array(PIL.Image.fromarray(np.uint8(im_orig)).resize((int(np.round(im_scale*im_orig.shape[1])), int(np.round(im_scale*im_orig.shape[0]))), PIL.Image.BILINEAR))
            im = im.astype(np.float32, copy=True)        
            im -= self.cfg['PIXEL_MEANS']

            #im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            #                interpolation=cv2.INTER_LINEAR)
            #log_info('Add %s from %s' % (im.shape, im_orig.shape))
            im_scale_factors.append(im_scale)
            # We need number of channels first, then height, then width
            im_transpose = im.transpose(2, 0, 1)
            processed_ims.append(im)

        # Create a tensor to hold the input images. Typically this will be
        # 1, 3, ..., 
        #blob = torch.Tensor(im_list_to_blob(processed_ims))
        blob = torch.Tensor(np.array(processed_ims).transpose([0,3,1,2]))  # JEBYRNE
        return blob, np.array(im_scale_factors)

    def _get_rois_blob(self, im_rois, im_scale_factors):
        """Converts RoIs into network inputs.

        Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob
        
        Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
        """
        rois, levels = self._project_im_rois(im_rois, im_scale_factors)
        rois_blob = np.hstack((levels, rois))
        return rois_blob.astype(np.float32, copy=False)

    def _project_im_rois(self, im_rois, scales):
        """Project image RoIs into the image pyramid built by _get_image_blob.
        
        Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob
        
        Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
        """
        im_rois = im_rois.astype(np.float, copy=False)
        
        if len(scales) > 1:
            widths = im_rois[:, 2] - im_rois[:, 0] + 1
            heights = im_rois[:, 3] - im_rois[:, 1] + 1

            areas = widths * heights
            scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
            diff_areas = np.abs(scaled_areas - 224 * 224)
            levels = diff_areas.argmin(axis=1)[:, np.newaxis]
        else:
            levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

            rois = im_rois * scales[levels]

        return rois, levels

    def im_detect(self, net, im, boxes=None):
        """Detect object classes in an image given object proposals.
        
        Arguments:
        net (pytorch): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order, as (H, W, C)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)
        
        Returns:
        scores (ndarray): R x K array of object class scores (K includes
        background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
        """
        im_blob, im_scales = self._get_image_blob(im)

        im_info = torch.Tensor(np.array([[im_blob.shape[2], im_blob.shape[3], im_scales[0]]], dtype=np.float32))

        # We think these are already the right shape?
        # # Now ready to supply inputs to network.
        # # reshape network inputs
        # net.blobs['data'].reshape(*(blobs['data'].shape))
        # if self.cfg.TEST.HAS_RPN:
        #     net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
        # else:
        #     net.blobs['rois'].reshape(*(blobs['rois'].shape))
        
        # do forward
        # Returns are all on CPU
        (rois, bbox_pred, cls_prob, cls_score) = net(im_blob, im_info)
        del im_blob
        del im_info
        # gc.collect(2)
        # torch.cuda.empty_cache()

        if self.cfg['TEST']['HAS_RPN']:
            assert len(im_scales) == 1, "Only single-image batch implemented"
            rois = rois.detach().numpy()
            # unscale back to raw image space
            boxes = rois[:, 1:5] / im_scales[0]

        if self.cfg['TEST']['SVM']:
            # use the raw scores before softmax under the assumption they
            # were trained as linear SVMs
            scores = cls_score.detach().numpy()
        else:
            # use softmax estimated probabilities
            scores = cls_prob.detach().numpy()

        if self.cfg['TEST']['BBOX_REG']:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.detach().numpy()
            pred_boxes = self.bbox_transform_inv(boxes, box_deltas)
            pred_boxes = self.clip_boxes(pred_boxes, im.shape)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        if self.cfg['DEDUP_BOXES'] > 0 and not self.cfg['TEST']['HAS_RPN']:
            # Map scores and predictions back to the original set of boxes
            raise ValueError('unsupported configuration option')
            #scores = scores[inv_index, :]
            #pred_boxes = pred_boxes[inv_index, :]

        del rois
        del bbox_pred
        del cls_prob
        del cls_score

        return scores, pred_boxes


    def bbox_transform(self, ex_rois, gt_rois):
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
        
        gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
        gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
        gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights
        
        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = np.log(gt_widths / ex_widths)
        targets_dh = np.log(gt_heights / ex_heights)
        
        targets = np.vstack(
            (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
        return targets

    def bbox_transform_inv(self, boxes, deltas):
        if boxes is None or boxes.shape[0] == 0:
            return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

        boxes = boxes.astype(deltas.dtype, copy=False)

        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        
        dx = deltas[:, 0::4]
        dy = deltas[:, 1::4]
        dw = deltas[:, 2::4]
        dh = deltas[:, 3::4]
        
        pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
        pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        pred_w = np.exp(dw) * widths[:, np.newaxis]
        pred_h = np.exp(dh) * heights[:, np.newaxis]
        
        pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        # y2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes

    def clip_boxes(self, boxes, im_shape):
        """
        Clip boxes to image boundaries.
        """
        
        # x1 >= 0
        boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
        # y1 >= 0
        boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
        # x2 < im_shape[1]
        boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
        # y2 < im_shape[0]
        boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
        return boxes
