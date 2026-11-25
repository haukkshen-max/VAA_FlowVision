"""Offline regression tests. Does NOT claim YOLO real-world accuracy."""
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pandas as pd
import pytest

from src.fixed_camera import (
    AnalysisConfig, Zone, assign_zone, build_partition_zones,
    probe_video, process_video, summarize, validate_zones, zone_overlaps,
)


class FakeYOLO:
    """Predicts predetermined bounding boxes for pipeline and metrics testing."""
    def __init__(self, batches, tracked=False):
        self.batches = batches
        self.index = 0
        self.tracked = tracked

    def predict(self, img, **kwargs):
        assert kwargs['classes']==[0]
        assert img.ndim==3
        batch = self.batches[min(self.index,len(self.batches)-1)]
        self.index+=1
        arr = np.array(batch,dtype=float).reshape(-1,4)
        boxes = SimpleNamespace(xyxy=arr,conf=np.ones(len(arr))*.91,id=None)
        return [SimpleNamespace(boxes=boxes)]

    def track(self,img,**kw):
        if not self.tracked:
            return self.predict(img,**kw)
        batch=self.batches[min(self.index,len(self.batches)-1)]
        self.index+=1
        arr=np.array(batch,dtype=float).reshape(-1,4)
        boxes=SimpleNamespace(xyxy=arr,conf=np.ones(len(arr))*.9,id=np.arange(len(arr)))
        return [SimpleNamespace(boxes=boxes)]


@pytest.fixture
def test_video(tmp_path):
    path = tmp_path/'one_fixed_camera.avi'
    writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10.0,(300,180))
    assert writer.isOpened()
    for k in range(40):
        frame=np.full((180,300,3),232,dtype=np.uint8)
        cv2.putText(frame,f'frame {k}',(20,90),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),2)
        writer.write(frame)
    writer.release()
    return path


def box(x):
    return [x-8,60,x+8,140]


def test_partition_and_assign():
    zones=build_partition_zones(['Trái','Giữa','Phải'],[10,12,15],boundaries=[1/3,2/3])
    assert assign_zone(20,80,300,180,zones)==0
    assert assign_zone(150,80,300,180,zones)==1
    assert assign_zone(280,80,300,180,zones)==2
    assert assign_zone(300,180,300,180,zones)==2
    assert zone_overlaps(zones)==[]
    assert zone_overlaps([Zone('A',0,0,.7,1),Zone('B',.5,0,1,1)])==[('A','B')]
    with pytest.raises(ValueError):
        validate_zones([Zone('trung',0,0,.2,1),Zone('trung',.2,0,.4,1)])
    with pytest.raises(ValueError):
        validate_zones([Zone('Khu',.7,0,.3,1)])


def test_pipeline_full_one_camera(test_video,tmp_path):
    zones=build_partition_zones(['Khu A','Khu B','Khu C'],[1,1,1])
    # fps 10, 1fps sample -> indexes 0,10,20,30
    batches=[
        [box(50),box(140),box(150)],          # A1, B2
        [box(160),box(250)],                  # B1, C1
        [box(120),box(140),box(170)],         # B3
        [box(110),box(165)],                  # B2
    ]
    messages=[]
    out=process_video(test_video,zones,AnalysisConfig(sample_fps=1.,max_duration_sec=4.,image_size=480),
                      model=FakeYOLO(batches),progress=lambda p,txt:messages.append((p,txt)),
                      output_root=tmp_path/'sessions')
    assert out.info.fps==10
    assert out.summary['camera_count']==1
    assert out.summary['sampled_frames']==4
    assert out.summary['total_detection_events']==10
    assert out.summary['mean_people_per_frame']==2.5
    assert out.summary['peak_people_in_frame']==3
    assert out.summary['busiest_zone_by_average']=='Khu B'
    assert out.summary['busiest_zone_mean']==2.
    assert out.summary['estimated_track_ids'] is None
    assert list(out.samples['total'])==[3,2,3,2]
    assert list(out.samples['Khu B'])==[2,1,3,2]
    assert out.samples['outside_zones'].sum()==0
    assert len(out.previews)==4
    assert all(p.exists() for p in out.previews)
    assert out.zip_path.exists()
    assert (out.session_dir/'bao_cao_demo.html').exists()
    assert (out.session_dir/'so_nguoi_theo_khung.csv').exists()
    html=(out.session_dir/'bao_cao_demo.html').read_text(encoding='utf-8')
    assert 'Khu B' in html and '2.50' in html
    assert 'không phải tổng khách duy nhất' in html
    import zipfile
    with zipfile.ZipFile(out.zip_path) as z:
        names=set(z.namelist())
    assert 'ket_qua.json' in names and 'bao_cao_demo.html' in names
    assert 'one_fixed_camera.avi' not in names  # original video never packed
    assert messages[-1][0]==1.


def test_app_analysis_renders_heatmap_and_reports(test_video, monkeypatch):
    """Exercise the analyze button and render all tabs using real pipeline output."""
    import json
    from streamlit.testing.v1 import AppTest

    # Make B busiest so the sorted zone table differs from configured A/B/C order.
    model = FakeYOLO([[box(50), box(140), box(150)]] * 4)
    monkeypatch.setattr('src.fixed_camera.load_model', lambda path: model)
    monkeypatch.setattr('src.fixed_camera.RUNTIME_DIR', test_video.parent / 'app_runs')
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'))
    app.run()
    app.sidebar.radio[0].set_value('Đường dẫn video trên máy').run()
    app.sidebar.text_input[0].set_value(str(test_video)).run()
    app.button[0].click().run(timeout=30)

    assert not app.exception, [exc.message for exc in app.exception]
    assert not app.error
    assert app.success
    result = app.session_state['analysis_result']
    assert result.zone_summary['zone'].tolist() == ['Khu B', 'Khu A', 'Khu C']
    charts = app.get('plotly_chart')
    assert len(charts) == 3
    heatmap = json.loads(charts[2].proto.spec)['data'][0]
    assert heatmap['type'] == 'heatmap'
    assert heatmap['y'] == ['Khu A', 'Khu B', 'Khu C']
    assert heatmap['z'] == [[1.0], [2.0], [0.0]]
    assert len(app.get('download_button')) == 4
    assert result.zip_path.is_file()

    # Streamlit keeps analysis in session state when the page reruns.
    app.run()
    assert not app.exception
    assert len(app.get('plotly_chart')) == 3
    assert len(app.get('download_button')) == 4


def test_roi_outside_and_tracking(test_video,tmp_path):
    zones=build_partition_zones(['A','B'],[5,5])
    # crop x=[150..300], detected box center in cropped coords=35 -> global x=185 -> B
    batches=[[box(35)] for _ in range(4)]
    out=process_video(test_video,zones,AnalysisConfig(sample_fps=1,max_duration_sec=4,
             roi=(.5,0.,1.,1.),use_tracking=True),model=FakeYOLO(batches,tracked=True),output_root=tmp_path/'runs')
    assert out.summary['estimated_track_ids']==1
    assert out.samples['B'].sum()==4
    assert out.samples['A'].sum()==0
    assert out.summary['total_detection_events']==4
    # tracking IDs are intentionally described as *not* unique people
    assert 'not a validated count' in out.summary['tracked_id_warning']


def test_unassigned_counts_and_start_video(test_video,tmp_path):
    zones=[Zone('Only',0,0,.2,1,10)]
    model=FakeYOLO([[box(260)] for _ in range(2)])
    out=process_video(test_video,zones,AnalysisConfig(sample_fps=1,start_sec=2,max_duration_sec=2),
                      model=model,output_root=tmp_path/'runs')
    assert list(out.samples['video_frame'])==[20,30]
    assert out.summary['mean_people_per_frame']==1
    assert out.summary['mean_unassigned_per_frame']==1
    assert out.samples['Only'].sum()==0
    assert out.zone_summary.iloc[0]['mean_people']==0
    assert out.summary['busiest_minute_total']==0


def test_empty_frame_is_valid(test_video,tmp_path):
    z=build_partition_zones(['A','B'],[12,12])
    out=process_video(test_video,z,AnalysisConfig(sample_fps=1,max_duration_sec=2),
                      model=FakeYOLO([[],[]]),output_root=tmp_path/'empty')
    assert out.summary['sampled_frames']==2
    assert out.summary['peak_people_in_frame']==0
    assert out.summary['total_detection_events']==0
    assert out.zone_summary['share_detected_pct'].sum()==0


def test_30_minute_math_without_fabricating_results():
    # 1fps for 30min = 1800 frame positions, *not* 1800 individuals.
    fps=25
    total_frames=30*60*fps
    step=round(fps/1.)
    assert (total_frames+step-1)//step==1800


def test_full_30_minute_timeline_with_mock_model(tmp_path):
    """Real 30-minute VIDEO TIMELINE, mocked detections (not YOLO accuracy)."""
    path=tmp_path/'30min_sample.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),1.,(300,180))
    assert writer.isOpened()
    blank=np.full((180,300,3),200,dtype=np.uint8)
    for _ in range(1800):
        writer.write(blank)
    writer.release()
    batches=[[box(50)] if t<1200 else [box(250)] for t in range(1800)]
    zones=build_partition_zones(['A','B'],[15,15])
    out=process_video(path,zones,AnalysisConfig(sample_fps=1.,max_duration_sec=30*60),
        model=FakeYOLO(batches),output_root=tmp_path/'reports')
    assert out.summary['sampled_frames']==1800
    assert out.summary['analyzed_duration_sec']==1800
    assert out.summary['peak_people_in_frame']==1
    assert out.summary['total_detection_events']==1800
    assert len(out.minute_summary)==30
    assert out.summary['busiest_zone_by_average']=='A'
    assert list(out.samples['A'].iloc[[0,1199,1200,1799]])==[1,1,0,0]
    assert len(out.previews)==5


def test_optional_video_export(test_video,tmp_path):
    zones=build_partition_zones(['A','B'],[4,4])
    model=FakeYOLO([[box(50)]]*2)
    out=process_video(test_video,zones,AnalysisConfig(max_duration_sec=2,sample_fps=1.,create_annotated_video=True),
                      model=model,output_root=tmp_path/'exports')
    assert out.annotated_video is not None
    assert out.annotated_video.exists()
    assert out.annotated_video.stat().st_size>0
    video=cv2.VideoCapture(str(out.annotated_video))
    assert video.isOpened()
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT))==2
    video.release()
