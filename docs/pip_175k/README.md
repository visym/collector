# People in Public - 175K

# Overview

The People in Public dataset is a consented large scale video dataset of people doing things in public places.  Our team has pioneered the use of a 
custom designed mobile app that combines video collection, activity labeling and bounding box annotation into a single step.  Our goal is to 
make collecting annotated video datasets as easily and cheaply as recording a video.  Currently, we are collecting a dataset of the MEVA 
classes (http://mevadata.org).  This package provides a release of this dataset, containing 184,404K annotated activity instances collected by 
over 150 subjects in 44 countries around the world. 

# Quickstart

## Release summary

This release was curated to export PIP-175K with additional context, to:

-Extract only instances that have been positively rated by the review team
-Stabilize the bounding box to the primary actor
-Dilate the bounding box for each primary actor performing the activity by a factor of 2.0, to provide context 
-Set the bounding box to maximum square
-Crop the actor tubelet in each frame, with zero padding
-Resize the tubelet so that the maximum dimension is 512x512
-Add the MEVA-specific temporal padding

## Installation

Follow the installation instructions for vipy-1.8.26

https://github.com/visym/vipy

Then:

```python
unzip pip175k.zip -d /path/to/your/folder
import vipy
cd /path/to/your/folder
pip = vipy.util.load('valset.pkl')
```

## Visualize

```python
v = pip[0]  # first video 
v.show()   # display annotated video
v.play()   # display unannotated video
v.quicklook().show()   # display video summary image
v[0].savefig().saveas('out.png')  # save annotated first frame of first video, convert rgba to rgb colorspace, and save to a PNG
v.tracks()  # tracks ID and tracks in this video
v.activities()  # activity ID and activities in this video
v_doors = [v for v in pip if 'door' in v.category()]  # only videos with door categories
categories = set([v.category() for v in pip])  # set of pip categories
d_pip2meva = vipy.util.load('categories_pip_to_meva.pkl')  # category mapping
d_category_to_counts = {k:len(v) for (k,v) in vipy.util.groupbyasdict(pip, lambda v: v.category()).items()}
```

## Toolchain Exports

```python
v.csv('/path/to/out.csv')  # export annotations as flat CSV
v.dict()  # export this object as python dictionary
v.torch()   # export frames as torch tensor
v.numpy()  # export frames as numpy array
labels = [(labels, im) for (labels, im) in v.labeled_frames()]  # framewise activity labels for multi-label loss
v.mindim(256).randomcrop( (224,224) ).torch(startframe='random', length=64)   # change the minimum dimension of the video to (and scale annotations), take random square center crop 
    			      					     		  # and export as a torch tensor of size 1x64x224x224 starting from a random start frame. 
```

If you are training with this dataset, we recommend following this demo to generate framewise activity labels and tensors:

https://github.com/visym/vipy/blob/master/demo/training.ipynb

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
* PIP is exported from the raw uploaded original video by creating an actor centered tublet, clipping each activity, cropping around the actor, setting to maxsquare, resizing to 256x256 and encoding to H.264.

* Moving camera.  Our cameras are hand-held, which means that the background is not stabilized.  We provide stabilization tools runnable as:

```python
from vipy.flow import Flow
v_stabilized = Flow().stabilize(v.mindim(256))
v_stabilized.show()
```

* Temporal padding.  We have added the MEVA annotation style temporal padding requirements as follows:
 
    * Reference:  https://gitlab.kitware.com/meva/meva-data-repo/blob/master/documents/MEVA-Annotation-Definitions.pdf
    * Pad one second before, zero seconds after: set(['person_opens_facility_door', 'person_closes_facility_door', 'person_opens_car_door', 'person_closes_car_door', 
                                                      'person_opens_car_trunk', 'person_opens_motorcycle_trunk', 'person_closes_car_trunk', 'person_closes_motorcycle_trunk',
                                                      'car_stops', 'motorcycle_stops', 'person_interacts_with_laptop'])        

    * pad one second before, one second after, up to maximum of two seconds:  set(['person_enters_scene_through_structure'])
    * person_exits_scene_through_structure:  Pad one second before person_opens_facility_door label (if door collection), and ends with enough padding to make this minimum two seconds     
    * person_enters_vehicle: Starts one second before person_opens_vehicle_door activity label and ends at the end of person_closes_vehicle_door activity, split motorcycles into separate class
    * person_exits_vehicle:  Starts one second before person_opens_vehicle_door, and ends at person_exits_vehicle with enough padding to make this minimum two seconds, split motorcycles into separate class
    * person_unloads_vehicle:  No padding before label start (the definition states one second of padding before cargo starts to move, but our label starts after the trunk is open, 
    * so there is a lag from opening to touching the cargo which we assume is at least 1sec), ends at the end of person_closes_trunk.
    * equal padding to minimum of five seconds:  set(['person_talks_to_person', 'person_reads_document'])
    * person_texting_on_phone:  Equal padding to minimum of two seconds
    * pad one second before, one second after:  set(['car_turns_left', 'motorcycle_turns_left', 'car_turns_right', 'motorcycle_turns_right', 'person_transfers_object_to_person', 'person_transfers_object_to_vehicle',
                                                     'person_sets_down_object', 'hand_interacts_with_person_handshake', 'hand_interacts_with_person_highfive', 'hand_interacts_with_person_holdhands', 'person_embraces_person', 'person_purchases',
                                                     'vehicle_picks_up_person','vehicle_drops_off_person'])
    * pad zero second before, one second after:  set(['vehicle_makes_u_turn', 'person_picks_up_object'])
    * person_abandons_package:  two seconds before, two seconds after


# License

Creative commons Attribution 4.0 International (CC BY 4.0)
https://creativecommons.org/licenses/by/4.0/


# Contact

Visym Labs <info@visym.com>

