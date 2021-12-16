# People in Public - 175k - Stabilized

![](pip_175k_stabilized.webp)

# Overview

The People in Public dataset is a consented large scale video dataset of people doing things in public places.  Our team has pioneered the use of a 
custom designed mobile app that combines video collection, activity labeling and bounding box annotation into a single step.  Our goal is to 
make collecting annotated video datasets as easily and cheaply as recording a video.  Currently, we are collecting a dataset of the MEVA 
classes (http://mevadata.org).  This package provides a release of this dataset, containing 215,250 annotated activity instances collected by 
over 150 subjects in 44 countries around the world. 

This dataset contains 215,250 stabilized video clips of 66 classes of activities performed by people in public places.  The activity labels are subsets of the 37 activities in the [Multiview Extended Video with Activities (MEVA)](https://mevadata.org) dataset and is consistent with the [Activities in Extended Video (ActEV)](https://actev.nist.gov/) challenge.  

[Background stabilization](https://github.com/visym/vipy/blob/bc20f6f32492badd181faa0ccf7b0029f1f63fee/vipy/flow.py#L307-L328) was performed using an affine coarse to fine optical-flow method, followed by [actor bounding box stabilization](https://github.com/visym/collector/blob/adc5486c7f88291b77f9a707a78763c2b5958406/pycollector/detection.py#L177-L236).  Stabilization is designed to minimize distortion for small motions in the region near the center of the actor box.  Remaining stabilization artifacts are due to non-planar scene structure, rolling shutter distortion, and sub-pixel optical flow correspondence errors.  Stabilization artifacts manifest as a subtly shifting background relative to the actor which may affect optical flow based methods.  All stabilizations can be filtered using the provided stabilization residual which measures the quality of the stabilization.  

## Download

* [pip_370k_stabilized.tar.gz (226 GB)](https://github.com/visym/collector/tree/master/docs/pip_370k_stabilized#download) 
     * This release contains the pip-175k-stabilized as a subset and should be used for new development

<!--
Legacy Downloads:
* [pip_175k_stabilized.tar.gz (117.10 GB)](https://dl.dropboxusercontent.com/s/mutvbxnj9nid8yd/pip_175k_stabilized.tar.gz)&nbsp;&nbsp;MD5:3c3c70cfe888e101822a9dc0dc0ff96e&nbsp;&nbsp;Updated:08Mar21
     * New training should use the full release in [People in Public - 370k - stabilized](../pip_370k_stabilized/README.md).  
     * This training set is preserved for legacy training.
-->

## Quickstart

See [pip-175k](https://visym.github.io/collector/pip_175k/).

To extract the smallest square video crop containing the stabilized track for a vipy.video.Scene() object v:

```python
import vipy
v = vipy.util.load('/path/to/stabilized.json')[0]   # load videos and take one
vs = v.crop(v.trackbox(dilate=1.0).maxsquare()).resize(224,224).saveas('/path/to/out.mp4')
vs.getattribute('stabilize')   # returns a stabilization residual (bigger is worse)
```

## Best Practices 

[Notebook demo](https://htmlpreview.github.io/?https://github.com/visym/collector/blob/master/docs/pip_175k/best_practices.html)&nbsp;[[html]](https://htmlpreview.github.io/?https://github.com/visym/collector/blob/master/docs/pip_175k/best_practices.html)[[ipynb]](https://github.com/visym/collector/blob/master/docs/pip_175k/best_practices.ipynb) showing best practices for using the PIP-175k dataset for training.

## Errata

* The classes hand_interacts_with_person_shakehands and person_shakes_hand are duplicated and should be merged
* The classes hand_interacts_with_person_holdhands and person_holds_hand are duplicated and should be merged
* The classes person_abandons_bag and person_abandons_package are near duplicates and should be merged
* The classes person_steals_object and person_steals_object_from_person are duplicated and should be merged

# License

Creative Commons Attribution 4.0 International [(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

Every subject in this dataset has consented to their personally identifable information to be shared publicly for the purpose of advancing computer vision research.  Non-consented subjects have their faces blurred out.  

# Acknowledgement

Supported by the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior/ Interior Business Center (DOI/IBC) contract number D17PC00344. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA, DOI/IBC, or the U.S. Government.

# Contact

Visym Labs <a href="mailto:info@visym.com">&lt;info@visym.com&gt;</a>

