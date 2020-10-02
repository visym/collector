import vipy
import pycollector.detection


def test_face():
    im = vipy.image.Image(url='https://upload.wikimedia.org/wikipedia/en/d/d6/Friends_season_one_cast.jpg')
    d = pycollector.detection.FaceDetector()
    ims = d(im)
    assert len(ims.objects()) == 6
    print('[test_detection]: face detector passed')

    
def test_object():
    im = vipy.image.vehicles()
    d = pycollector.detection.Detector()
    ims = d(im)
    assert len(ims.objects()) == 50
    print('[test_detection]: object detector passed')
    
    
def _test_proposal():
    videoids = ['20200522_0150473561208997437622670',
                '00EBA2AD-8D45-4888-AA28-A71FAAA1ED88-1000-0000005BA4C1929E',
                'F3EAB9E5-3DCF-41AD-90C0-626E56E367A9-1000-00000064E17CA24F',
                'DD4F0E6E-A222-4FE0-8180-378B57A9FA76-2093-0000019498261CC0',
                '20200528_1411011390426986',
                '20200528_141345-2035479356',
                '20200520_2017063615715963202368863',
                '477AE69F-153D-48F7-8CEA-CE54688DE942-8899-0000068727516F05',
                'DCE8A056-8EAF-4268-8624-5FA4EB42B416-4214-00000259540C3B4C',
                '20200521_180041290232562',
                '20200526_183503604561539675548030',
                '9F7BEDDF-4317-4CCF-A17B-D0CD84BE7D29-14557-000009A78377B03B',
                '82DD0A37-30CC-4A74-B1F9-028FDF567983-289-0000000E8DC9A679',
                'E4B58A6B-F79A-4E11-83C3-924BEDA69D3A-320-000000225CC6B4E8',
                '20200503_1101496154439725041568313',
                '20200505_1608442273346368432213366',
                '9802EE07-1C5F-467E-AF19-3569A6AF9440-1763-00000147FBD3138A',
                '20200423_1104206253700009606748525',
                '07826D93-E5C4-41DB-A8BC-4D3203E64F91-862-0000009D976888CB',
                '24FD34F3-AC56-4528-8770-D6A0A30A3358-4367-000002F2F18C0C5F',
                '6A8698F4-31C4-43E2-B061-55FF4E250615-264-0000000DE8AE00DB',
                '20200525_1830282647560902514919243',
                '20200525_1748021531575967472321212',
                '20200525_1658548039717461873489470',
                '133BA88D-A828-4397-81BD-6EEB9393F20B-710-0000005AEDD91457']
    

    #import shutil    
    #shutil.rmtree(remkdir('test_proposal'))
    #vipy.globals.gpuindex(0)
    #for videoid in videoids:
    #    collectorproposal_vs_objectproposal(Video(videoid), dt=3).annotate().saveas(os.path.join(remkdir('test_proposal'), '%s.mp4' % videoid))
