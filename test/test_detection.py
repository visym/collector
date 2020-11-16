import os
import vipy
from vipy.util import remkdir
import pycollector.detection
import pytest 

@pytest.mark.skip(reason="Skip testing for now")
def test_face():
    im = vipy.image.Image(url='https://upload.wikimedia.org/wikipedia/en/d/d6/Friends_season_one_cast.jpg')
    d = pycollector.detection.FaceDetector()
    ims = d(im)
    assert len(ims.objects()) == 6
    print('[test_detection]: face detector passed')

    
@pytest.mark.skip(reason="Skip testing for now")
def test_object():
    im = vipy.image.vehicles()
    d = pycollector.detection.ObjectDetector()
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

    from pycollector.admin.video import Video  # admin only
    from pycollector.admin.globals import backend, api
    api('v1')
    for videoid in videoids:
        pycollector.detection._collectorproposal_vs_objectproposal(Video(videoid, ignoreerrors=True), dt=3).annotate().saveas(os.path.join(remkdir('test_proposal'), '%s.mp4' % videoid))


def _test_actor_association():
    from pycollector.admin.video import Video  # admin only
    from pycollector.admin.globals import backend

    videoid = ['16E4872D-94BA-4764-B47E-65DAFBA5A055',
               'A3A05DEF-4E8B-4650-B6D2-71BF43AD18D8',
               '2329CC6E-0C3C-4131-82CC-C0D97E014D28',
               '06F2C244-13B1-4432-BACE-50E7B7DB3031',
               '8B51B8F4-8563-4EE6-B07F-40FA5CBFC08F',
               '2A0E7D65-2516-4339-85C4-58E9723620E4',
               '4CEFF31A-EDB7-4965-AEAF-1011F0B7F6FD',
               '2153BF43-7400-404C-90F8-E4DC01FB1CD7',
               'AB1F0FEC-0792-4D63-B134-8FCBA770A4BA',
               'E8F54604-E424-4288-ABCE-3E250F02A606',
               'AFCACDBF-4D3D-47B7-8C26-9CC20A30D676',
               'F2AFB94C-24A1-4E8F-A9C7-9116724EE2B9',
               '2144C42F-F4FF-4026-A694-611A260A0A25',
               '155942E9-8737-4198-9D81-1CEC7B57FE65',
               '0A51105F-EBDB-4B6E-8E8E-339BDFCB0B90',
               'FF827732-7879-4A71-8AB5-C08955ACCA04',
               'F36203CA-60BC-40AD-9401-34A184C16D83',
               '5C778B7F-3A51-420A-8BF4-90340D2F8245',
               '40D34A1A-1452-46C4-8349-8DF6E839C488',
               'B5C773D4-A4FD-42DE-A835-D71C0A95BF82',
               '2462FAD0-978A-44FE-B76C-280CBDF28947',
               'BF50607F-A760-4A61-A59C-D7995D9CCC13',
               'A5CF88A6-50DD-4172-B584-2F449D3CB924',
               '962DD31D-16DE-41C6-B50B-D3A3C51E0376',
               '0BD99FE0-72B4-49C1-88BF-A6B1CA4962F9',
               '1D2B142B-C326-4220-88BE-CC8995AF36E9',
               'A44F1067-3407-4641-8082-FB97DAB690B0',
               '30DA050E-0A0E-436F-9B6C-4D75D6AD0733',
               'EBA0ECD8-FF0D-4243-85DF-817F9FB26CDD',
               '390B542F-DF37-40AD-A5D0-A62D64F3D686',
               'C173B159-63AD-4A72-A703-1759FF5D7BCA']
             
    V = [Video(id) for id in videoid]    

    V = vipy.util.scpload('scp://ma01-5200-0052:/tmp/57e1d8821c0df241.pkl')
    
    C = backend().collections()
    P = pycollector.detection.ActorAssociation()

    for v in V:
        for o in C.collection(v.project(), v.collection()).secondary_objects():
            target = o if o != 'Friend' else 'Person'
            vp = P(v.stabilize().savetmp(), target, dt=10)
            vp.crop(vp.trackbox().dilate(1.2)).annotate().saveas(os.path.join(remkdir('test_actor_association'), '%s.mp4' % v.videoid()))
            print(vp)

