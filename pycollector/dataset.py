import os
import vipy
import numpy as np
import pycollector.detection
from vipy.globals import print
from vipy.util import findpkl, toextension, filepath, filebase, jsonlist
import random


def tocsv(pklfile):
    pass


def isdataset(indir):
    """Does the provided path contain a collector dataset?"""
    assert os.path.isdir(indir) and len(jsonlist(indir))>0


def disjoint_activities(V, activitylist):    
    assert all([isinstance(v, vipy.video.Video) for v in V])
    assert isinstance(activitylist, list) and len(activitylist)>0 and len(activitylist[0]) == 2
    for (after, before) in activitylist:
        V = [v.activitymap(lambda a: a.disjoint([sa for sa in v.activitylist() if sa.category() == after]) if a.category() == before else a) for v in V]  
    V = [v.activityfilter(lambda a: len(a)>0) for v in V]  # some activities may be zero length after disjoint
    return V


def resize_dataset(indir, outdir, dilate=1.2, maxdim=256, maxsquare=True):
    """Convert redistributable dataset by resizing the activity tube.
    
       * Indir: /path/to/src that contains trainset.pkl
       * outdir: should be /mypath/src

    """

    (indir, outdir) = (os.path.normpath(indir), os.path.normpath(outdir))
    print('[people_in_public]: converting %s -> %s' % (indir, outdir))
    pklfiles = [f for f in findpkl(indir) if filebase(f) in set(['trainset','valset','testset'])]
    assert len(pklfiles) > 0
    for pklfile in pklfiles:
        b = vipy.batch.Batch(vipy.util.load(pklfile), 24)
        dataset = b.filter(lambda v: v.hastracks()).map(lambda v: v.activitytube(dilate=dilate, maxdim=maxdim).saveas(v.filename().replace(indir, outdir)).print()).result()
        vipy.util.distsave(dataset, datapath=indir, outfile=pklfile.replace(indir, outdir))
    return outdir


def boundingbox_refinement(V, ngpu, batchsize):
    # Proposals:  Improve collector proposal for each video with an optimal object proposal.  This will result in filtering away a small number of hard positives.
    print('[prepare_pip]: betterbox %d videos' % (len(V)))
    model = pycollector.detection.VideoProposalRefinement(batchsize=batchsize)  # =8 (0046) =20 (0053)
    B = vipy.batch.Batch(V, ngpu=ngpu)
    V = B.scattermap(lambda net,v: net(v, proposalconf=5E-2, proposaliou=0.8, miniou=0.2, dt=3, mincover=0.8, byclass=True, shapeiou=0.7, smoothing='spline', splinefactor=None, strict=True), model).result()  
    V = [v.activityfilter(lambda a: any([a.hastrack(t) and len(t)>5 and t.during(a.startframe(), a.endframe()) for t in v.tracklist()])) for v in V]  # get rid of activities without tracks greater than dt
    B.shutdown()  # garbage collect GPU resources
    return V


def split_dataset(A, trainfraction=0.7, testfraction=0.1, valfraction=0.2, seed=42, verbose=True):
    assert all([isinstance(a, vipy.video.Video) for a in A]), "Invalid input"
    
    np.random.seed(seed)
    d = vipy.util.groupbyasdict(A, lambda a: a.category())
    (trainset, testset, valset) = ([],[],[])
    for (c,v) in d.items():
        np.random.shuffle(v)
        if len(v) < 3:
            print('[collector-backend.dataset]: skipping category "%s" with too few examples' % c)
            continue

        (testlist, vallist, trainlist) = vipy.util.dividelist(v, (testfraction, valfraction, trainfraction))
        testset.extend(testlist)
        valset.extend(vallist)  
        trainset.extend(trainlist)

    if verbose:
        print('[pycollector.dataset]: trainset=%d (%1.1f)' % (len(trainset), trainfraction))
        print('[pycollector.dataset]: valset=%d (%1.1f)' % (len(valset), valfraction))
        print('[pycollector.dataset]: testset=%d (%1.1f)' % (len(testset), testfraction))
    return (trainset, testset, valset)


def asmeva(V):
    """Convert a list of collector.dataset.Video() to MEVA annotation style"""
    assert all([isinstance(v, vipy.video.Scene) for v in V])
    
    # MEVA annotations assumptions:  https://docs.google.com/spreadsheets/d/19I3C5Zb6RHS0QC30nFT_m0ymArzjvlPLfb5SSRQYLUQ/edit#gid=0
    # Pad one second before, zero seconds after
    before1after0 = set(['person_opens_facility_door', 'person_closes_facility_door', 'person_opens_car_door', 'person_closes_car_door', 
                         'person_opens_car_trunk', 'person_opens_motorcycle_trunk', 'person_closes_car_trunk', 'person_closes_motorcycle_trunk',
                         'car_stops', 'motorcycle_stops', 'person_interacts_with_laptop'])        
    
    V = [v.activitymap(lambda a: a.temporalpad( (v.framerate()*1.0, 0) ) if a.category() in before1after0 else a) for v in V]
    
    # pad one second before, one second after, up to maximum of two seconds
    before1after1max2 = set(['person_enters_scene_through_structure'])
    V = [v.activitymap(lambda a: a.temporalpad(max(0, (v.framerate()*1.0)-len(a))) if a.category() in before1after1max2 else a) for v in V]
    
    # person_exits_scene_through_structure:  Pad one second before person_opens_facility_door label (if door collection), and ends with enough padding to make this minimum two seconds
    V = [v.activitymap(lambda a: (a.startframe(np.min([sa.startframe() for sa in v.activities().values() if sa.isneighbor(a) and sa.category() == 'person_opens_facility_door'] + [a.startframe()]))
                                  .temporalpad( (0, 0) )  # padding is rolled into person_opens_*
                                  if a.category() == 'person_exits_scene_through_structure' else a)) for v in V]        
    
    # person_enters_vehicle: Starts one second before person_opens_vehicle_door activity label and ends at the end of person_closes_vehicle_door activity, split motorcycles into separate class
    V = [v.activitymap(lambda a: (a.startframe(np.min([sa.startframe() for sa in v.activities().values() if sa.isneighbor(a) and sa.category() == 'person_opens_car_door'] + [a.startframe()]))
                                  .endframe(np.max([sa.endframe() for sa in v.activities().values() if sa.isneighbor(a) and sa.category() == 'person_closes_car_door'] + [a.endframe()]))
                                  .temporalpad( (0, 0) )  # padding is rolled into person_opens_*
                                  if a.category() == 'person_enters_car' else a)) for v in V]        
    
    # person_exits_vehicle:  Starts one second before person_opens_vehicle_door, and ends at person_exits_vehicle with enough padding to make this minimum two seconds, split motorcycles into separate class
    V = [v.activitymap(lambda a: (a.startframe(np.min([sa.startframe() for sa in v.activities().values() if sa.isneighbor(a) and sa.category() == 'person_opens_car_door'] + [a.startframe()]))
                                  .endframe(np.max([sa.endframe() for sa in v.activities().values() if sa.isneighbor(a) and sa.category() == 'person_closes_car_door'] + [a.endframe()]))
                                  .temporalpad( (0, 0) )  # padding is rolled into person_opens_*
                                  if a.category() == 'person_exits_car' else a)) for v in V]        
    
    # person_unloads_vehicle:  No padding before label start (the definition states one second of padding before cargo starts to move, but our label starts after the trunk is open, 
    # so there is a lag from opening to touching the cargo which we assume is at least 1sec), ends at the end of person_closes_trunk.
    V = [v.activitymap(lambda a: (a.endframe(np.max([sa.endframe() for sa in v.activities().values() if sa.isneighbor(a) and sa.category() == 'person_closes_car_trunk'] + [a.endframe()]))
                                  if a.category() == 'person_unloads_car' else a)) for v in V]
    V = [v.activitymap(lambda a: (a.endframe(np.max([sa.endframe() for sa in v.activities().values() if sa.isneighbor(a) and sa.category() == 'person_closes_motorcycle_trunk'] + [a.endframe()]))
                                  if a.category() == 'person_unloads_motorcycle' else a)) for v in V]
    
    # person_talks_to_person:  Equal padding to minimum of five seconds
    equal5 = set(['person_talks_to_person', 'person_reads_document'])
    V = [v.activitymap(lambda a: a.temporalpad(max(0, (v.framerate()*2.5)-len(a))) if a.category() in equal5 else a) for v in V]
    
    # person_texting_on_phone:  Equal padding to minimum of two seconds
    V = [v.activitymap(lambda a: a.temporalpad(max(0, (v.framerate()*1.0)-len(a))) if a.category() == 'person_texting_on_phone' else a) for v in V]
    
    # Pad one second before, one second after
    before1after1 = set(['car_turns_left', 'motorcycle_turns_left', 'car_turns_right', 'motorcycle_turns_right', 'person_transfers_object_to_person', 'person_transfers_object_to_vehicle',
                         'person_sets_down_object', 'hand_interacts_with_person_handshake', 'hand_interacts_with_person_highfive', 'hand_interacts_with_person_holdhands', 'person_embraces_person', 'person_purchases',
                         'car_picks_up_person', 'car_drops_off_person', 'motorcycle_drops_off_person', 'motorcycle_picks_up_person'])
    V = [v.activitymap(lambda a: a.temporalpad(v.framerate()*1.0) if a.category() in before1after1 else a) for v in V]
    
    # Pad zero second before, one second after
    before0after1 = set(['car_makes_u_turn', 'motorcycle_makes_u_turn', 'person_picks_up_object'])  
    V = [v.activitymap(lambda a: a.temporalpad( (0, v.framerate()*1.0) ) if a.category() in before0after1 else a) for v in V]
    
    # person_abandons_package:  two seconds before, two seconds after
    V = [v.activitymap(lambda a: a.temporalpad(v.framerate()*2.0) if a.category() == 'person_abandons_package' else a) for v in V]
    return V



def tohtml(pklfile, outfile, mindim=512, datapath=None, title='Visualization'):
    """Generate a standalone HTML file containing quicklooks for each annotated activity in dataset, along with some helpful provenance information for where the annotation came from"""
    vipy.globals.max_workers(24)
    dataset = vipy.util.load(pklfile) if datapath is None else vipy.util.distload(pklfile, datapath)
    dataset = dataset if not isinstance(dataset, tuple) else dataset[0]  # backwards compatible
    quicklist = vipy.batch.Batch(dataset).map(lambda v: (v.load().quicklook(), v.flush().print()))
    quicklooks = [imq for (imq, v) in quicklist]  # for HTML display purposes
    provenance = [{'clip':str(v), 'activities':str(';'.join([str(a) for a in v.activitylist()])), 'category':v.category()} for (imq, v) in quicklist]
    (quicklooks, provenance) = zip(*sorted([(q,p) for (q,p) in zip(quicklooks, provenance)], key=lambda x: x[1]['category']))  # sorted in category order
    return vipy.visualize.tohtml(quicklooks, provenance, title='%s: %s' % (title, pklfile), outfile=outfile, mindim=mindim)


def activitymontage(pklfile, gridrows=30, gridcols=50, mindim=64):
    """30x50 activity montage, each 64x64 elements using the output of prepare_dataset"""
    (vidlist, d_category) = vipy.util.load(pklfile)
    actlist = [v.mindim(mindim) for v in vidlist]
    np.random.seed(42); random.shuffle(actlist)
    actlist = actlist[0:gridrows*gridcols]
    return vipy.visualize.videomontage(actlist, mindim, mindim, gridrows=gridrows, gridcols=gridcols).saveas(toextension(pklfile, 'mp4')).filename()


def activitymontage_bycategory(pklfile, gridcols=49, mindim=64):
    """num_categoryes x gridcols activity montage, each row is a category"""
    np.random.seed(42)
    (vidlist, d_category) = vipy.util.load(pklfile)

    actlist = []
    for k in sorted(d_category.keys()):
        actlist_k = [v for v in vidlist if v.category() == k]
        random.shuffle(actlist_k)
        assert len(actlist_k) >= gridcols
        actlist.extend(actlist_k[0:gridcols])
        outfile = os.path.join(filepath(pklfile), '%s_%s.mp4' % (filebase(pklfile), k))
        print(vipy.visualize.videomontage(actlist_k[0:15], 256, 256, gridrows=3, gridcols=5).saveas(outfile).filename())

    outfile = os.path.join(filepath(pklfile), '%s_%d_bycategory.mp4' % (filebase(pklfile), gridcols))
    print('[pycollector..dataset.activitymontage_bycategory]: rows=%s' % str(sorted(d_category.keys())))
    return vipy.visualize.videomontage(actlist, mindim, mindim, gridrows=len(d_category), gridcols=gridcols).saveas(outfile).filename()
