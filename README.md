# M3Act: Learning from Synthetic Human Group Activities

**[CVPR 2024](https://cvpr.thecvf.com/virtual/2024/poster/29759) | Official Repository**


[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://cjerry1243.github.io/M3Act/)
[![Paper](https://img.shields.io/badge/Paper-Link-blue)](https://openaccess.thecvf.com/content/CVPR2024/html/Chang_Learning_from_Synthetic_Human_Group_Activities_CVPR_2024_paper.html)
[![arXiv](https://img.shields.io/badge/arXiv-2306.16772-red)](https://arxiv.org/abs/2306.16772)

This repository contains a Unity project with the core modules and assets for our synthetic data generator, M3Act. 
We also release the 3D group activity dataset, M3Act3D, as well as the essential tools for data processing, visualization, and evaluation of the dataset.

## Introduction
**TLDR**. M3Act is a synthetic data generator with multi-view multi-group multi-person atomic human actions and group activities.
![Teaser](assets/Teaser.png)
M3Act is designed to support multi-person and multi-group research. It features multiple semantic groups and produces highly diverse and photorealistic videos with a rich set of annotations suitable for human-centered tasks including multi-person tracking, group activity recognition, and controllable human group activity generation. Please refer to our project page and paper for more details.

## Synthetic Data Generator

For setting up the Unity environment and generating data with M3Act, please refer to the [m3act-unity-generator](https://github.com/danruili/m3act-unity-generator) repository.


## 3D Group Activity Generation

Please refer to [gag](./gag/) folder for more details.


## Citation

If you find our work useful, please cite the following works.

```BibTeX
@inproceedings{chang2024learning,
  title={Learning from Synthetic Human Group Activities},
  author={Chang, Che-Jui and Li, Danrui and Patel, Deep and Goel, Parth and Zhou, Honglu and Moon, Seonghyeon and Sohn, Samuel S and Yoon, Sejong and Pavlovic, Vladimir and Kapadia, Mubbasir},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21922--21932},
  year={2024}
}
```

```BibTeX
@article{chang2024equivalency,
  title={On the Equivalency, Substitutability, and Flexibility of Synthetic Data},
  author={Chang, Che-Jui and Li, Danrui and Moon, Seonghyeon and Kapadia, Mubbasir},
  journal={arXiv preprint arXiv:2403.16244},
  year={2024}
}
```

## License
This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
See the [LICENSE](LICENSE) file for more details.

