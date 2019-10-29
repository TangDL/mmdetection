import os.path as osp
import warnings
import cv2

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from ..registry import PIPELINES


@PIPELINES.register_module
class LoadImageFromFile(object):

    def __init__(self, to_float32=False):
        self.to_float32 = to_float32

    def __call__(self, results):
        filename = osp.join(results['img_prefix'],
                            results['img_info']['filename'])
        img = mmcv.imread(filename)                            # 将已经读取到的瑕疵图片存放到这里

        # my second attemp to send 2 images into net
        #         template_root_path = "/data1/bupi_data/round2/all_template/"
        #         template_filename = template_root_path + 'template_' + filename.split('/')[-1].split('_')[0] + '.jpg'
        #         template_img = mmcv.imread(template_filename)             # 将读取到的模板图片存放到这里
        #         # 需要将瑕疵图片和模板图片的尺寸对应上，这里选择将模板图片resize到瑕疵图片的大小。注意：读入的模板图片是原图尺寸
        #         h, w, _ = img.shape                                       # 得到瑕疵图片的高，宽
        #         template_img = cv2.resize(template_img, (w, h))
        #
        #         print("template_img", template_img.shape, "img:", img.shape)
        #         # img, template_img = img.transpose(2,0,1), template_img(2,0,1)   # 先将图片矩阵的维度进行变换，方便后面连接
        #         # img = np.concatenate([img, template_img])                # 将两张图片拼接成6*905*1024的矩阵
        #         # img = img.transpose(1, 2, 0)  # 最后将图片矩阵变换为905*1024*6。
        #         img = np.dstack((img, template_img))
        #
        #         # 以上便完成了两张图片的读取与传入
        #
        #         print(img.shape, template_img.shape)
        #         # 将模板图片和瑕疵图片合并，合并成6通道的
        #         print(filename, "1111111111111111111111111111111111111111111111")

        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename

        # end--------------------------------------------------------------------------------
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(
            self.to_float32)


@PIPELINES.register_module
class LoadAnnotations(object):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 skip_img_without_anno=True):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.skip_img_without_anno = skip_img_without_anno

    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes']
        if len(results['gt_bboxes']) == 0 and self.skip_img_without_anno:
            file_path = osp.join(results['img_prefix'],
                                 results['img_info']['filename'])
            warnings.warn(
                'Skip the image "{}" that has no valid gt bbox'.format(
                    file_path))
            return None
        results['gt_bboxes_ignore'] = ann_info.get('bboxes_ignore', None)
        results['bbox_fields'].extend(['gt_bboxes', 'gt_bboxes_ignore'])
        return results

    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['labels']
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _load_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        results['gt_semantic_seg'] = mmcv.imread(
            osp.join(results['seg_prefix'], results['ann_info']['seg_map']),
            flag='unchanged').squeeze()
        return results

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={}, with_label={}, with_mask={},'
                     ' with_seg={})').format(self.with_bbox, self.with_label,
                                             self.with_mask, self.with_seg)
        return repr_str


@PIPELINES.register_module
class LoadProposals(object):

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        proposals = results['proposals']
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                'but found {}'.format(proposals.shape))
        proposals = proposals[:, :4]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([0, 0, 0, 0], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(num_max_proposals={})'.format(
            self.num_max_proposals)
