## BIBR
A Bi-temporal Image Boundary Refinement (BIBR) module for Remote Sensing Image Change Detection.

## Introduction
Overview of the proposed BIBR module. Given the bi-temporal images and the coarse change map from the CD network, we first extract contours from the bi-temporal images and match them with the contours in the change map. The boundary blocks of these contours are extracted via a BBE and overlap filter, followed by filtering boundary blocks located at the land-cover object center with an inside filter. These prefiltered boundary blocks are fed into the BDN, and the re-delineation predictions are filtered with a distribution filter. Finally, the predictions from both are combined to obtain the final BR result.
<p align="center">
<img src="BIBR.png" width="100%" alt="framework"/>
</p>

For more details, please refer to our [paper](https://ieeexplore.ieee.org/document/10965597).

## Prepare Boundary Block dataset [optional]
If you want to regenerate the block dataset, please download our [pretrained weights](https://drive.google.com/drive/folders/1pmHfvk_doWFFpoBk4SB4cHHIRDkddHfX?usp=sharing) <for Contour Match (CM)> and place them in the path of the corresponding dataset.

For example, preparing boundary block dataset for WHU-CD 40%:
* `Google Drive/BIBR/exp/whu/pretrained/best.pth` -> `BIBR/exp/whu/pretrained/best.pth`

Replace the path in the code and run the script below.
```
python prepare_dataset_whu.py
```

**Alternatively, we have already segmented the boundary block datasets for WHU-CD, LEVIR-CD, and DSIN-CD and can download them [here](https://pan.baidu.com/s/1VXFkd15YvW7by02vkW3I6w?pwd=w7nb).**


## Train Boundary Delineation Network (BDN)
Prepare the corresponding proportion of semi-supervised boundary block training and testing datasets, and use [UniMatch](https://github.com/LiheYoung/UniMatch) to train the Boundary Delineation Network (BDN).
```
python unimatch_deeplabv3plus_whu.py 2>&1 | tee train_whu40_0.02_16.log
```

## Acknowledgement

This project is based on [BPR](https://github.com/chenhang98/BPR/tree/main) and [UniMatch](https://github.com/LiheYoung/UniMatch). Thank you very much for their outstanding work.

## Citation

If you find this project useful in your research, please consider citing:

```
@article{su2025semi,
  title={Semi-Supervised Change Detection With Boundary Refinement Teacher},
  author={Su, You and Song, Yonghong and Wu, Xiaomeng and Hu, Hao and Chen, Jingqi and Wen, Zehan},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025},
  publisher={IEEE}
}
```
