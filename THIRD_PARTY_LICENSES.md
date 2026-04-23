# Third-Party Licenses

Reconstruction Zone uses the following open-source libraries:

| Package | License | Use |
|---------|---------|-----|
| [PyTorch](https://pytorch.org/) | BSD-3-Clause | Tensor computation, GPU acceleration |
| [torchvision](https://pytorch.org/) | BSD-3-Clause | Image transforms, model utilities |
| [NumPy](https://numpy.org/) | BSD-3-Clause | Array computation |
| [OpenCV](https://opencv.org/) | Apache-2.0 | Image processing |
| [Pillow](https://pillow.readthedocs.io/) | HPND (MIT-like) | Image I/O |
| [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) | MIT | GUI framework |
| [tqdm](https://tqdm.github.io/) | MIT + MPL-2.0 | Progress bars |
| [PyYAML](https://pyyaml.org/) | MIT | YAML parsing |
| [py360convert](https://github.com/sunset1995/py360convert) | MIT | Equirectangular projection |
| [SciPy](https://scipy.org/) | BSD-3-Clause | Scientific computing |
| [RF-DETR](https://github.com/roboflow/rf-detr) | Apache-2.0 | Transformer object detection |
| [supervision](https://github.com/roboflow/supervision) | MIT | Detection result processing |
| [Transformers](https://huggingface.co/docs/transformers) | Apache-2.0 | Model loading (HuggingFace) |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | Apache-2.0 | Model downloads |
| [SAM 3](https://github.com/facebookresearch/sam3) | Apache-2.0 | Text-prompted segmentation |

### GitHub release only (not included in commercial builds)

| Package | License | Use |
|---------|---------|-----|
| [ultralytics](https://github.com/ultralytics/ultralytics) | **AGPL-3.0** | YOLO26 object detection |

The AGPL-3.0 license requires that any distributed software incorporating ultralytics
must also be distributed under AGPL-3.0 or a compatible copyleft license. The GitHub
release of Reconstruction Zone is distributed under GPL-3.0, which is compatible.
The commercial (Gumroad) release excludes ultralytics entirely.
