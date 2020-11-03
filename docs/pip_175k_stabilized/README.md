# People in Public - 175k - Stabilized

[![Collector visualization](https://rawcdn.githack.com/visym/collector/5381e805cac4195ddebe25389b0e40368536a710/docs/pip_175k_stabilized/pip_175k_stabilized.webp)](https://rawcdn.githack.com/visym/collector/5381e805cac4195ddebe25389b0e40368536a710/docs/pip_175k_stabilized/pip_175k_stabilized.webp)

# Overview

The People in Public dataset is a consented large scale video dataset of people doing things in public places.  Our team has pioneered the use of a 
custom designed mobile app that combines video collection, activity labeling and bounding box annotation into a single step.  Our goal is to 
make collecting annotated video datasets as easily and cheaply as recording a video.  Currently, we are collecting a dataset of the MEVA 
classes (http://mevadata.org).  This package provides a release of this dataset, containing 184,379 annotated activity instances collected by 
over 150 subjects in 44 countries around the world. 

This dataset contains 184,379 stabilized video clips of 68 classes of activities performed by people in public places.  The activity labels are subsets of the 37 activities in the [Multiview Extended Video with Activities (MEVA)](https://mevadata.org) dataset and is consistent with the [Activities in Extended Video (ActEV)](https://actev.nist.gov/) challenge.  

[Background stabilization](https://github.com/visym/vipy/blob/bc20f6f32492badd181faa0ccf7b0029f1f63fee/vipy/video.py#L2084-L2087) was performed using an affine coarse to fine optical-flow method, followed by [actor bounding box stabilization](https://github.com/visym/collector/blob/adc5486c7f88291b77f9a707a78763c2b5958406/pycollector/detection.py#L177-L236).  Stabilizations exhibit low artifacts for small motions in the region near the center of the actor box.  All stabilizations can be filtered using the attribute "v.getattribute('stabilize')" on the stabilization residual as desired.  Remaining stabilization artifacts are due to non-planar scene structure, rolling shutter distortion, and sub-pixel optical flow correspondence errors. 

# Download

* [pip_175k_stabilized_0.tar.gz (12.4 GB)](https://dl.dropboxusercontent.com/s/j8p4gxeyjit3z1z/pip_175k_stabilized_0.tar.gz)&nbsp;&nbsp;MD5:1b66b03173dab65318454bf77b898b52&nbsp;&nbsp;&nbsp;&nbsp;
* pip_175k_stabilized_{1-9}.tar.gz (XXX GB) uploading ...

# Quickstart

See [pip-175k](https://visym.github.io/collector/pip_175k/).

To extract the smallest video crop containing the stabilized track for a vipy.video.Scene() object v:

```python
import vipy
v.crop(v.trackbox(dilate=1.0).maxsquare()).saveas('/path/to/out.mp4')
v.getattribute('stabilize')   # returns a stabilization residual (bigger is worse)
```
# Best Practices for Training

[Notebook demo](https://htmlpreview.github.io/?https://github.com/visym/collector/blob/master/docs/pip_175k/best_practices.html) showing best practices for using the PIP-175k dataset for training.

# License

Creative Commons Attribution 4.0 International [(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

Every subject in this dataset has consented to their personally identifable information to be shared publicly for the purpose of advancing computer vision research.  Non-consented subjects have their faces blurred out.  

# Acknowledgement

Supported by the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior/ Interior Business Center (DOI/IBC) contract number D17PC00344. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA, DOI/IBC, or the U.S. Government.

# Contact

Visym Labs <a href="mailto:info@visym.com">&lt;info@visym.com&gt;</a>

