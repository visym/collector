<a href="https://github.com/visym/collector" class="github-corner" aria-label="View source on GitHub"><svg width="80" height="80" viewBox="0 0 250 250" style="fill:#151513; color:#fff; position: absolute; top: 0; border: 0; right: 0;" aria-hidden="true"><path d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"></path><path d="M128.3,109.0 C113.8,99.7 119.0,89.6 119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3 123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9 134.4,103.2" fill="currentColor" style="transform-origin: 130px 106px;" class="octo-arm"></path><path d="M115.0,115.0 C114.9,115.1 118.7,116.5 119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0 127.5,74.4 143.8,58.0 C148.5,53.4 154.0,51.2 159.7,51.0 C160.3,49.4 163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6 187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2 216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4 203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1 C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z" fill="currentColor" class="octo-body"></path></svg></a>

# People in Public - 175k

[![Collector visualization](https://i3.ytimg.com/vi/HjNa7_T-Xkc/maxresdefault.jpg)](https://youtu.be/HjNa7_T-Xkc)

# Overview

The People in Public dataset is a consented large scale video dataset of people doing things in public places.  Our team has pioneered the use of a 
custom designed mobile app that combines video collection, activity labeling and bounding box annotation into a single step.  Our goal is to 
make collecting annotated video datasets as easily and cheaply as recording a video.  Currently, we are collecting a dataset of the MEVA 
classes (http://mevadata.org).  This package provides a release of this dataset, containing 184,379 annotated activity instances collected by 
over 150 subjects in 44 countries around the world. 

# Download

This dataset contains 184,379 video clips of 68 classes of activities performed by people in public places.  The activity labels are subsets of the 37 activities in the [Multiview Extended Video with Activities (MEVA)](https://mevadata.org) dataset and is consistent with the [Activities in Extended Video (ActEV)](https://actev.nist.gov/) challenge.  
* [pip_175k.tar.gz (55.3GB)](https://dl.dropboxusercontent.com/s/xwiacwo9y5uci9v/pip_175k.tar.gz)&nbsp;&nbsp;MD5:9e49f8608ba0170dfaa1ed558351f0df&nbsp;&nbsp;&nbsp;&nbsp;
* Visualization of [Training set random sample (87MB)](https://rawcdn.githack.com/visym/collector/5b051c625ef458417a16ed48d5a0693ef59fd9ff/docs/pip_175k/trainset_small.html),&nbsp;[full validation set (1.1GB)](https://dl.dropboxusercontent.com/s/8fp77nvxeywrq7f/pip_175k_valset.html)


# Quickstart

## Release summary

This release was curated to export PIP-175k with additional context, to:

* Extract only instances that have been positively rated by the review team
* Stabilize the bounding box to the primary actor
* Dilate the bounding box for each primary actor performing the activity by a factor of 2.0, to provide context 
* Set the bounding box to maximum square
* Crop the actor tubelet in each frame, with zero padding
* Resize the tubelet so that the maximum dimension is 512x512
* Add the MEVA-specific temporal padding

## Installation

Follow the installation instructions for [vipy](https://github.com/visym/vipy)

Unpack pip_175k.tar.gz in /path/to/, then:

```python
import vipy
cd /path/to/pip_175k
pip = vipy.util.load('valset.pkl')
```

## Visualize

```python
v = pip[0]  # first video 
v.play()   # display unannotated video
v.show()   # generate video annotations and play when complete
v.fastshow()   # generate streaming video annotations
v.quicklook().show()   # display video summary image
v[0].savefig().saveas('out.png')  # save annotated first frame of first video, and save to a PNG
v.tracks()  # tracks ID and tracks in this video
v.activities()  # activity ID and activities in this video
v_doors = [v for v in pip if 'door' in v.category()]  # only videos with door categories
categories = set([v.category() for v in pip])  # set of pip categories
d_pip2meva = vipy.util.load('categories_pip_to_meva.pkl')  # category mapping
d_category_to_counts = vipy.util.countby(pip, lambda v: v.category())
```

## Toolchain Exports

```python
v.csv('/path/to/out.csv')  # export annotations for this video as flat CSV file (with header)
pipcsv = [v.csv() for v in pip]  # export all annotations for this dataset as list of tuples
v.json()  # export this annotated video as json 
v.torch()   # export frames as torch tensor
v.numpy()  # export frames as numpy array
labels = [(labels, im) for (labels, im) in v.labeled_frames()]  # framewise activity labels for multi-label loss
v.mindim(256).randomcrop( (224,224) ).torch(startframe='random', length=64)   # change the minimum dimension of the video to (and scale annotations), take random square center crop 
    			      					     		  # and export as a torch tensor of size 1x64x224x224 starting from a random start frame. 
mp4file = v.filename()  # absolute path the the MP4 video file                                                      
mp4file_resized = v.resize(cols=256).saveas('resized.mp4').filename() # absolute path the resized MP4 video file                                                      
```

If you are training with this dataset, we recommend [following this demo to generate framewise activity labels and tensors](https://github.com/visym/vipy/blob/master/demo/training.ipynb) and use the [best practices](https://htmlpreview.github.io/?https://github.com/visym/collector/blob/master/docs/pip_175k/best_practices.html).  

Alternatively, contact us and we can work with you to export a dataset to your specifications that can be imported directly by your toolchain.


# PIP Collection Notes

* PIP has separate vehicle activity classes for car and motorcycle
* PIP has two separate purchasing classes for person_purchaes_with_machine and person_purchase_with_cashier.  The cashier class has been removed temporarily.
* PIP has three separate hand interaction classes for highfive, handshake and holding_hands
* PIP stealing.  We are collecting "person takes object while person is not looking".  
* PIP person_transfers_object.  We are collecting "person hands object to person" and "person hands object to person in car".  
* PIP videos are limited to maximum of 30 seconds
* PIP currently contains only the primary actor, and does not yet include the additional required objects
* PIP is designed for training activity classification using actor centered tubelet or cuboid activity proposals
* PIP is reviewed by at least two human reviewer for labeling accuracy.  
* PIP does not enforce MEVA excluded objects:  Phones, Pens/Pencils/Markers, Individual Sheets of Paper, Money, Hat, Gloves, Apple (or similarly sized food items).  We leave the choice of prop up to the collectors
* PIP is exported from the raw uploaded original video by creating an actor centered tublet, clipping each activity, dilating by 2x, cropping around the actor, setting to maxsquare, resizing to 512x512 and encoding to H.264.

* Moving camera.  Our cameras are hand-held, which means that the background is not stabilized.  We provide optical flow based stabilization tools and a [stabilized version of this dataset](https://github.com/visym/collector/tree/master/docs/pip_175k_stabilized).   You can run your own stabilization using:

```python
v.stabilize().show()
```

* Disjoint activities.  The MEVA annotation definitions can result in [disjoint activites that overlap](https://github.com/visym/vipy/tree/master/vipy/dataset) (e.g. opening and closing simultaneously), as shown in the [MEVA visualization](https://www.dropbox.com/s/benzhkmzqrggj5j/meva_kf1_annotations_07may20.html?dl=0).  As a result, we do not enforce disjoint activities in this release.    

* Temporal padding.  We have added the [MEVA annotation style](https://gitlab.kitware.com/meva/meva-data-repo/blob/master/documents/MEVA-Annotation-Definitions.pdf) temporal padding requirements as follows:
 
    * Pad one second before, zero seconds after: set('person_opens_facility_door', 'person_closes_facility_door', 'person_opens_car_door', 'person_closes_car_door', 'person_opens_car_trunk', 'person_opens_motorcycle_trunk', 'person_closes_car_trunk', 'person_closes_motorcycle_trunk',
'car_stops', 'motorcycle_stops', 'person_interacts_with_laptop')        
    * Pad one second before, one second after, up to maximum of two seconds:  set(['person_enters_scene_through_structure'])
    * person_exits_scene_through_structure:  Pad one second before person_opens_facility_door label (if door collection), and ends with enough padding to make this minimum two seconds     
    * person_enters_vehicle: Starts one second before person_opens_vehicle_door activity label and ends at the end of person_closes_vehicle_door activity, split motorcycles into separate class
    * person_exits_vehicle:  Starts one second before person_opens_vehicle_door, and ends at person_exits_vehicle with enough padding to make this minimum two seconds, split motorcycles into separate class
    * person_unloads_vehicle:  No padding before label start (the definition states one second of padding before cargo starts to move, but our label starts after the trunk is open, so there is a lag from opening to touching the cargo which we assume is at least 1sec, ends at the end of person_closes_trunk.
    * Equal padding to minimum of five seconds:  set('person_talks_to_person', 'person_reads_document')
    * person_texting_on_phone:  Equal padding to minimum of two seconds
    * Pad one second before, one second after:  set('car_turns_left', 'motorcycle_turns_left', 'car_turns_right', 'motorcycle_turns_right', 'person_transfers_object_to_person', 'person_transfers_object_to_vehicle','person_sets_down_object', 'hand_interacts_with_person_handshake', 'hand_interacts_with_person_highfive', 'hand_interacts_with_person_holdhands', 'person_embraces_person', 'person_purchases',
'vehicle_picks_up_person','vehicle_drops_off_person')
    * pad zero second before, one second after:  set('vehicle_makes_u_turn', 'person_picks_up_object')
    * person_abandons_package:  two seconds before, two seconds after

This temporal padding may result in negative start times for some activities.

# Best Practices for Training

[Notebook demo](https://htmlpreview.github.io/?https://github.com/visym/collector/blob/master/docs/pip_175k/best_practices.html)&nbsp;[[html]](https://htmlpreview.github.io/?https://github.com/visym/collector/blob/master/docs/pip_175k/best_practices.html)[[ipynb]](https://github.com/visym/collector/blob/master/docs/pip_175k/best_practices.ipynb) showing best practices for using the PIP-175k dataset for training.

# License

Creative Commons Attribution 4.0 International [(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

Every subject in this dataset has consented to their personally identifable information to be shared publicly for the purpose of advancing computer vision research.  Non-consented subjects have their faces blurred out.  

# Acknowledgement

Supported by the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior/ Interior Business Center (DOI/IBC) contract number D17PC00344. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA, DOI/IBC, or the U.S. Government.

# Contact

Visym Labs <a href="mailto:info@visym.com">&lt;info@visym.com&gt;</a>

