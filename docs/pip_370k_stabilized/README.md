# People in Public - 370k - Stabilized

![](../pip_175k_stabilized/pip_175k_stabilized.webp)

# Overview

The People in Public dataset is a consented large scale video dataset of people doing things in public places.  Our team has pioneered the use of a 
custom designed mobile app that combines video collection, activity labeling and bounding box annotation into a single step.  Our goal is to 
make collecting annotated video datasets as easily and cheaply as recording a video.  Currently, we are collecting a dataset of the MEVA 
classes (http://mevadata.org).  This dataset contains 405,781 background stabilized video clips of 63 classes of activities collected by over 150 subjects in 44 countries around the world. 


## Download

* [pip_370k.tar.gz (226 GB)](https://dl.dropboxusercontent.com/s/fai9ontpmx4xv9i/pip_370k.tar.gz)&nbsp;&nbsp;MD5:2cf844fbc78fde1c125aa250e99db19f&nbsp;&nbsp;Last Updated 16Sep21
    * This dataset contains 405,781 background stabilized video clips of 63 classes of activities.
    * This release uses the [no-meva-pad](../pip_250k_stabilized/README.md) annotation style.
    * This is the recommended dataset for download, as it is the union of pip-170k-stabilized, pip-250k-stabilized and pip-d370k-stabilized, which fixes known errata.   


<!--
Legacy Downloads:
* [pip_d370k_stabilized.tar.bz2 (59.7 GB)](https://dl.dropboxusercontent.com/s/vxjik8a01lp6uif/pip_d370k_stabilized.tar.bz2)&nbsp;&nbsp;MD5:7f705d6291dfa333000e40779b595d4f&nbsp;&nbsp;Last Updated: 04Apr21
    * An incremental release which augments [pip_250k](https://github.com/visym/collector/tree/master/docs/pip_250k_stabilized)
    * This dataset contains 95990 stabilized video clips of 34 classes of activities performed by people in public places.  
* [pip_d370k_stabilized_objects.tar.gz (709 MB)](https://dl.dropboxusercontent.com/s/ip3w9fmt8d26h94/pip_d370k_stabilized_objects.tar.gz)&nbsp;&nbsp;MD5:5e13f783ceec1378800d0e5de81f3257&nbsp;&nbsp;&nbsp;&nbsp;Last Updated: 06May21
    * An incremental release which augments [pip_250k](https://github.com/visym/collector/tree/master/docs/pip_250k_stabilized) that includes secondary vehicle and people track annotations for 40856 of 95990 instances in pip_d370k that contain secondary objects.
    * Contains 38546 instances with both vehicle and person tracks, 1245 instances with bicycle and person tracks, 1065 instances with person and friend
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
 
# Errata

* A small number of videos exhibit a face detector false alarm which looks like a large pixelated circle which lasts a single frame.  This is the in-app face blurring incorrectly redacting the background.  You can filter these videos by removing videos v with 

```python
videolist = [v for v in videolist if not v.getattribute('blurred faces') > 0]

```

* The metadata for each video in PIP-370k contains unique IDs that identify the collector who recorded the video and the subject in the video. A portion of the PIP-370k collection included a bug that caused subject IDs to be randomly generated. We recommend using the collector ID as the identifier of the subject. Collectors and subjects were required to work in pairs for this collection, so the collector ID uniquely identifies when the collector is behind the camera and their subject is in front of the camera. This means that collector_id for this collection will uniquely identify the subject in the pixels.  This can be accessed using the following for a video object v:

```python
v.metadata()['collector_id']
```

# Frequently Asked Questions

* Are there repeated instances in this release?  For this release, we asked collectors to perform the same activity multiple times in a row per collection, but to perform the activity slightly differently each time.  This introduces a form of on-demand dataset augmentation performed by the collector.  You may identify these collections with a filename structure "VIDEOID_INSTANCEID.mp4" where the video ID identifies the collected video, and each instance ID is an integer that identifies the order of the collected activity in the collected video.  
* Are there missing activities?  This is an incremental release, and should be combined with pip-250k for a complete release.
* What is "person_walks"?  This is a background activity class.  We asked collectors to walk around and act like they were waiting for a bus or the subway, to provide a background class.  The collection names in the video metadata for these activities are "Slowly walk around in an area like you are waiting for a bus" and "Walk back and forth".    
* Do these videos include the MEVA padding?  No, these videos are collected using the temporal annotations from the collectors directly.  This is due to the fact that many of the activities are collected back to back in a single collection, which may violate the two second padding requirement for MEVA annotations.  If the MEVA annotations are needed, they can be applied as follows to a list of video objects (videolist):

```python
padded_videolist = pycollector.dataset.asmeva(videolist)
```
* How is this related to pip-175k and pip-250k?  This dataset is a superset of pip-175k and pip-250k.
* How is background stabilization performed?  [Background stabilization](https://github.com/visym/vipy/blob/bc20f6f32492badd181faa0ccf7b0029f1f63fee/vipy/flow.py#L307-L328) was performed using an affine coarse to fine optical-flow method, followed by [actor bounding box stabilization](https://github.com/visym/collector/blob/adc5486c7f88291b77f9a707a78763c2b5958406/pycollector/detection.py#L177-L236).  Stabilization is designed to minimize distortion for small motions in the region near the center of the actor box.  Remaining stabilization artifacts are due to non-planar scene structure, rolling shutter distortion, and sub-pixel optical flow correspondence errors.  Stabilization artifacts manifest as a subtly shifting background relative to the actor which may affect optical flow based methods.  All stabilizations can be filtered using the provided stabilization residual which measures the quality of the stabilization.  


# License

Creative Commons Attribution 4.0 International [(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

Every subject in this dataset has consented to their personally identifable information to be shared publicly for the purpose of advancing computer vision research.  Non-consented subjects have their faces blurred out.  

# Acknowledgement

Supported by the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior/ Interior Business Center (DOI/IBC) contract number D17PC00344. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA, DOI/IBC, or the U.S. Government.

# Contact

Visym Labs <a href="mailto:info@visym.com">&lt;info@visym.com&gt;</a>

