import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES
from einops import rearrange
from mmdet3d.datasets.pipelines import LoadAnnotations3D
import os

@PIPELINES.register_module()
class LoadMultiViewImageFromFilesInCeph(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged', file_client_args=dict(backend='disk'), img_root=''):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = mmcv.FileClient(**self.file_client_args)
        self.img_root = img_root

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (list of str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        images_multiView = []
        filename = results['img_filename']
        for img_path in filename:
            img_path = os.path.join(self.img_root, img_path)
            if self.file_client_args['backend'] == 'petrel':
                img_bytes = self.file_client.get(img_path)
                img = mmcv.imfrombytes(img_bytes)
            elif self.file_client_args['backend'] == 'disk':
                img = mmcv.imread(img_path, self.color_type)
            images_multiView.append(img)
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            #[mmcv.imread(name, self.color_type) for name in filename], axis=-1)
            images_multiView, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations3D_E2E(LoadAnnotations3D):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """
    def __init__(self,
                 with_future_anns=False,
                 with_ins_inds_3d=False,
                 ins_inds_add_1=False,  # NOTE: make ins_inds start from 1, not 0
                 **kwargs):
        super().__init__(**kwargs)
        self.with_future_anns = with_future_anns
        self.with_ins_inds_3d = with_ins_inds_3d

        self.ins_inds_add_1 = ins_inds_add_1
    
    def _load_future_anns(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """

        gt_bboxes_3d = []
        gt_labels_3d = []
        gt_inds_3d = []
        # gt_valid_flags = []
        gt_vis_tokens  = []

        for ann_info in results['occ_future_ann_infos']:
            if ann_info is not None:
                gt_bboxes_3d.append(ann_info['gt_bboxes_3d'])
                gt_labels_3d.append(ann_info['gt_labels_3d'])
                
                ann_gt_inds = ann_info['gt_inds']
                if self.ins_inds_add_1:
                    ann_gt_inds += 1
                    # NOTE: sdc query is changed from -10 -> -9
                gt_inds_3d.append(ann_gt_inds)

                # gt_valid_flags.append(ann_info['gt_valid_flag'])
                gt_vis_tokens.append(ann_info['gt_vis_tokens'])
            else:
                # invalid frame
                gt_bboxes_3d.append(None)
                gt_labels_3d.append(None)
                gt_inds_3d.append(None)
                # gt_valid_flags.append(None)
                gt_vis_tokens.append(None)

        results['future_gt_bboxes_3d'] = gt_bboxes_3d
        # results['future_bbox3d_fields'].append('gt_bboxes_3d')  # Field is used for augmentations, not needed here
        results['future_gt_labels_3d'] = gt_labels_3d
        results['future_gt_inds'] = gt_inds_3d
        # results['future_gt_valid_flag'] = gt_valid_flags
        results['future_gt_vis_tokens'] = gt_vis_tokens

        return results 
  
    def _load_ins_inds_3d(self, results):
        ann_gt_inds = results['ann_info']['gt_inds'].copy() # TODO: note here

        # NOTE: Avoid gt_inds generated twice
        results['ann_info'].pop('gt_inds')
        
        if self.ins_inds_add_1:
            ann_gt_inds += 1
        results['gt_inds'] = ann_gt_inds
        return results

    def __call__(self, results):
        results = super().__call__(results)
        
        if self.with_future_anns:
            results = self._load_future_anns(results)
        if self.with_ins_inds_3d:
            results = self._load_ins_inds_3d(results)
        
        # Generate ann for plan
        if 'occ_future_ann_infos_for_plan' in results.keys():
            results = self._load_future_anns_plan(results)
        
        return results

    def __repr__(self):
        repr_str = super().__repr__()
        indent_str = '    '
        repr_str += f'{indent_str}with_future_anns={self.with_future_anns}, '
        repr_str += f'{indent_str}with_ins_inds_3d={self.with_ins_inds_3d}, '
        return repr_str


@PIPELINES.register_module()
class LoadCachedBEV:
    def __init__(self, cache_dir="", cache_format="npy", point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)):
        self.cache_dir = cache_dir
        self.cache_format = cache_format
        self.pixel_per_meter = 1/0.16 # 1 pixel is 0.16 meters
        self.original_point_cloud_range=(-69.12, -69.12, -2, 69.12, 69.12, 6)
        self.point_cloud_range = point_cloud_range

    def __call__(self, results):
        """Transform BEV features from ego to lidar coordinate system and crop to point cloud range.
        
        Args:
            results (dict): Contains sample information including sample_idx
            
        Returns:
            dict: Updated results with transformed and cropped BEV features
        """
        sample_token: str = results['sample_idx']
        cache_path = os.path.join(self.cache_dir, f"{sample_token}.{self.cache_format}")
        bev = np.load(cache_path)  # shape: (num_channels, height, width)
        
        original_shape = bev.shape
        # Rotate BEV 90 degrees counterclockwise to align with lidar coordinate system
        bev = np.rot90(bev, k=1, axes=(1, 2))
        
        # Calculate lidar offset translation (1m forward)
        forward_offset_pixels = int(1.0 * self.pixel_per_meter)
        
        # Create padded array for translation
        padded_bev = np.zeros((
            original_shape[0],  # channels
            original_shape[1] + forward_offset_pixels * 2,  # height
            original_shape[2] + forward_offset_pixels * 2,  # width
        ))
        
        # Place rotated BEV in padded array with offset
        padded_bev[
            :,
            forward_offset_pixels : forward_offset_pixels + bev.shape[1],
            forward_offset_pixels : forward_offset_pixels + bev.shape[2],
        ] = bev
        
        # Crop to original size, centered at lidar position
        translated_bev = padded_bev[
            :,
            forward_offset_pixels : forward_offset_pixels + original_shape[1],
            forward_offset_pixels : forward_offset_pixels + original_shape[2],
        ]
        
        # Calculate dimensions for cropping
        original_width_meters = (
            self.original_point_cloud_range[3] - self.original_point_cloud_range[0]
        )
        original_height_meters = (
            self.original_point_cloud_range[4] - self.original_point_cloud_range[1]
        )
        
        # Calculate crop indices in pixels
        left_crop = int(
            ((self.point_cloud_range[0] - self.original_point_cloud_range[0]) 
             / original_width_meters * translated_bev.shape[2])
        )
        right_crop = int(
            ((self.original_point_cloud_range[3] - self.point_cloud_range[3]) 
             / original_width_meters * translated_bev.shape[2])
        )
        bottom_crop = int(
            ((self.point_cloud_range[1] - self.original_point_cloud_range[1]) 
             / original_height_meters * translated_bev.shape[1])
        )
        top_crop = int(
            ((self.original_point_cloud_range[4] - self.point_cloud_range[4]) 
             / original_height_meters * translated_bev.shape[1])
        )
        
        # Perform final crop
        cropped_bev = translated_bev[
            :,
            bottom_crop : -top_crop if top_crop > 0 else None,
            left_crop : -right_crop if right_crop > 0 else None,
        ]
        
        results["bev_embed"] = cropped_bev
        return results