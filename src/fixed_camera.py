"""Fixed single-camera person occupancy analysis.

Counts are sampled-frame *occupancy estimates*, not unique visitors.
Each detection is assigned to at most one user-defined rectangular zone by
bottom-center of the box; unassigned detections remain in the total.
"""
from __future__ import annotations

import base64
import html
import os
import subprocess
import io
import json
import math
import shutil
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / 'assets' / 'model' / 'yolo11n.pt'
RUNTIME_DIR = PROJECT_DIR / 'data' / 'runtime'
COLORS = [(39, 103, 228), (41, 161, 120), (221, 144, 48), (140, 80, 200)] # OpenCV BGR


@dataclass(frozen=True)
class Zone:
    name: str
    x1: float
    y1: float
    x2: float
    y2: float
    capacity: int = 15


@dataclass(frozen=True)
class AnalysisConfig:
    confidence: float = 0.25
    image_size: int = 640
    sample_fps: float = 1.0
    start_sec: float = 0.0
    max_duration_sec: float = 1800.0
    device: str = 'auto'
    use_tracking: bool = False
    create_annotated_video: bool = False
    roi: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float
    fps_from_metadata: bool


@dataclass
class AnalysisResult:
    session_dir: Path
    info: VideoInfo
    config: AnalysisConfig
    samples: pd.DataFrame
    zone_summary: pd.DataFrame
    minute_summary: pd.DataFrame
    summary: dict[str, Any]
    previews: list[Path]
    zip_path: Path
    annotated_video: Path | None = None


def validate_zones(zones: list[Zone]) -> None:
    if not (1 <= len(zones) <= 8):
        raise ValueError('Cần từ 1 đến 8 khu vực.')
    if len({z.name.strip() for z in zones}) != len(zones) or any(not z.name.strip() for z in zones):
        raise ValueError('Tên khu vực không được để trống hoặc trùng nhau.')
    reserved = {'total','outside_zones','time_sec','sample_index','video_frame','time_label','minute_index'}
    if any(z.name.strip() in reserved or z.name.strip().endswith(('_mean','_max')) for z in zones):
        raise ValueError('Tên khu vực trùng tên cột dữ liệu dự phòng; hãy đổi tên khu vực.')
    for z in zones:
        if not (0 <= z.x1 < z.x2 <= 1 and 0 <= z.y1 < z.y2 <= 1):
            raise ValueError(f'Tọa độ khu vực không hợp lệ: {z.name}')
        if z.capacity < 1:
            raise ValueError(f'Sức chứa {z.name} phải ≥ 1.')


def zone_overlaps(zones: list[Zone]) -> list[tuple[str, str]]:
    out = []
    for i, a in enumerate(zones):
        for b in zones[i+1:]:
            if min(a.x2, b.x2) > max(a.x1, b.x1) and min(a.y2, b.y2) > max(a.y1, b.y1):
                out.append((a.name, b.name))
    return out


def build_partition_zones(names: list[str], capacities: list[int], orientation: str = 'vertical',
                          boundaries: list[float] | None = None) -> list[Zone]:
    if len(names) != len(capacities) or not names:
        raise ValueError('Tên và sức chứa khu vực không khớp.')
    boundaries = list(boundaries or [i / len(names) for i in range(1, len(names))])
    edges = [0.0] + boundaries + [1.0]
    if len(edges) != len(names) + 1 or any(edges[i+1] - edges[i] < 1e-6 for i in range(len(edges)-1)):
        raise ValueError('Ranh giới phải tăng dần và nằm trong (0, 1).')
    zones = []
    for i, (name, cap) in enumerate(zip(names, capacities)):
        if orientation == 'vertical':
            zones.append(Zone(name, edges[i], 0., edges[i+1], 1., int(cap)))
        elif orientation == 'horizontal':
            zones.append(Zone(name, 0., edges[i], 1., edges[i+1], int(cap)))
        else:
            raise ValueError('Chỉ hỗ trợ vertical/horizontal.')
    validate_zones(zones)
    return zones


def probe_video(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('Không mở được video. Hãy thử file MP4/H.264 hoặc AVI.')
    fps0 = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    valid_fps = math.isfinite(fps0) and 1 <= fps0 <= 240
    fps = fps0 if valid_fps else 25.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if w < 1 or h < 1:
        cap.release()
        raise ValueError('Không đọc được kích thước video.')
    cap.release()
    return VideoInfo(fps, max(0, frames), w, h, (frames/fps if frames else 0.), bool(valid_fps))


def first_frame(path: str | Path, second: float = 0.) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('Không mở được video.')
    if second > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(second) * 1000.)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise ValueError('Video không có khung hình hợp lệ tại thời điểm đã chọn.')
    return img


def assign_zone(foot_x: float, foot_y: float, width: int, height: int, zones: list[Zone]) -> int | None:
    # Edge detection: final endpoint is included to avoid losing people at image edges.
    nx, ny = foot_x / max(1, width), foot_y / max(1, height)
    for i, z in enumerate(zones):
        if z.x1 <= nx <= z.x2 and z.y1 <= ny <= z.y2:
            return i
    return None


def _rect(z: Zone, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    a = (int(round(z.x1 * width)), int(round(z.y1 * height)))
    b = (min(width - 1, int(round(z.x2 * width))), min(height - 1, int(round(z.y2 * height))))
    return a, b


def annotate_frame(frame: np.ndarray, zones: list[Zone], counts: dict[str, int] | None = None,
                   detections: list[dict] | None = None, roi: tuple[float, float, float, float] | None = None) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    layer = out.copy()
    for i, z in enumerate(zones):
        a, b = _rect(z, w, h)
        cv2.rectangle(layer, a, b, COLORS[i % len(COLORS)], thickness=-1)
    out = cv2.addWeighted(layer, .13, out, .87, 0)
    for i, z in enumerate(zones):
        a, b = _rect(z, w, h)
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(out, a, b, color, 2)
        label = f'{z.name}: {counts[z.name] if counts else "--"}'
        scale = max(.48, min(.85, w / 1400))
        tx = min(max(5, a[0] + 7), max(5, w-120))
        ty = max(22, min(h-7, a[1] + 27))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.rectangle(out, (tx-3, ty-th-5), (min(w-1, tx+tw+5), ty+6), (10, 25, 35), -1)
        cv2.putText(out, label, (tx,ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 2, cv2.LINE_AA)
    for d in detections or []:
        z_index = d.get('zone_index')
        color = COLORS[z_index % len(COLORS)] if z_index is not None else (190,190,190)
        x1, y1, x2, y2 = [int(x) for x in d['xyxy']]
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        cv2.circle(out, (int(d['foot_x']), min(h-1, int(d['foot_y']))), 4, color, -1)
    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(out, (int(w*x1),int(h*y1)), (int(w*x2),int(h*y2)), (0,220,220), 2)
    return out


def load_model(path: str | Path | None = None):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('Thiếu ultralytics. Hãy cài: python -m pip install -r requirements.txt') from exc
    chosen = Path(path) if path else MODEL_PATH
    if not chosen.exists():
        raise FileNotFoundError(f'Không thấy model: {chosen}')
    return YOLO(str(chosen))


def _resolve_device(value: str) -> str:
    if value != 'auto':
        return value
    try:
        import torch
        return '0' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


def _to_np(value: Any, dtype=float) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=dtype)
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def detect_persons(model: Any, frame: np.ndarray, cfg: AnalysisConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = _resolve_device(cfg.device)
    half = False
    if device != 'cpu':
        try:
            import torch
            half = bool(torch.cuda.is_available())
        except ImportError:
            pass
    kw = dict(classes=[0], conf=float(cfg.confidence), imgsz=int(cfg.image_size),
              max_det=600, iou=.55, device=device, half=half, verbose=False)
    if cfg.use_tracking:
        result = model.track(frame, tracker='bytetrack.yaml', persist=True, **kw)[0]
    else:
        result = model.predict(frame, **kw)[0]
    if result.boxes is None:
        return np.empty((0,4)), np.empty((0,)), np.empty((0,),dtype=int)
    boxes = _to_np(result.boxes.xyxy).reshape(-1,4)
    confidences = _to_np(result.boxes.conf).reshape(-1)
    raw_ids = getattr(result.boxes,'id',None)
    ids = _to_np(raw_ids,dtype=int).reshape(-1) if raw_ids is not None else np.full(len(boxes),-1,dtype=int)
    # Never drop detections without a track id: occupancy must reflect detections, not tracker matches.
    if len(ids) != len(boxes):
        ids = np.full(len(boxes),-1,dtype=int)
    return boxes, confidences, ids


def _pixel_roi(frame: np.ndarray, roi: tuple[float, float, float, float] | None) -> tuple[int,int,int,int]:
    h, w = frame.shape[:2]
    if roi is None:
        return 0,0,w,h
    x1,y1,x2,y2 = roi
    if not(0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError('ROI không hợp lệ.')
    a,b = max(0,int(x1*w)), max(0,int(y1*h))
    c,d = min(w,int(x2*w)), min(h,int(y2*h))
    if c <= a or d <= b:
        raise ValueError('ROI quá nhỏ.')
    return a,b,c,d


def summarize(samples: pd.DataFrame, zones: list[Zone], step_seconds: float,
              elapsed_sec: float) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    if samples.empty:
        raise ValueError('Không có khung hình nào được phân tích.')
    zone_rows = []
    total_assigned = sum(float(samples[z.name].sum()) for z in zones)
    for z in zones:
        vals = samples[z.name].astype(float)
        bucket = samples.assign(minute=(samples['time_sec']//60).astype(int))
        grouped = bucket.groupby('minute')[z.name].mean()
        best_minute = int(grouped.idxmax()) if not grouped.empty else 0
        over = vals > z.capacity
        zone_rows.append({
            'zone': z.name,
            'mean_people': round(float(vals.mean()), 2),
            'peak_people': int(vals.max()),
            'total_detection_events': int(vals.sum()),
            'share_detected_pct': round(100. * float(vals.sum()) / total_assigned, 2) if total_assigned else 0.,
            'capacity_reference': z.capacity,
            'over_capacity_samples': int(over.sum()),
            'over_capacity_pct': round(float(over.mean())*100, 2),
            'estimated_over_capacity_minutes': round(min(float(elapsed_sec),float(over.sum()) * step_seconds) / 60., 2),
            'busiest_minute_index': best_minute,
            'busiest_minute_label': f'{best_minute:02d}:00–{best_minute+1:02d}:00',
        })
    zs = pd.DataFrame(zone_rows).sort_values(['mean_people','peak_people'],ascending=False).reset_index(drop=True)
    tmp = samples.copy()
    tmp['minute_index'] = (tmp['time_sec']//60).astype(int)
    values = {'total': ['mean','max'], 'outside_zones': ['mean','max']}
    for z in zones:
        values[z.name] = ['mean','max']
    minute = tmp.groupby('minute_index',sort=True).agg(values)
    minute.columns = [f'{a}_{b}' for a,b in minute.columns]
    minute = minute.reset_index()
    minute['time_label'] = minute['minute_index'].map(lambda x: f'{int(x):02d}:00')
    minute = minute[['minute_index','time_label']+[x for x in minute.columns if x not in ['minute_index','time_label']]]
    # Cumulative detections count the SAME person repeatedly; deliberately not labeled 'unique people'.
    summary = {
        'camera_count': 1,
        'sampled_frames': int(len(samples)),
        'first_sample_second': round(float(samples['time_sec'].iloc[0]), 2),
        'last_sample_second': round(float(samples['time_sec'].iloc[-1]), 2),
        'analyzed_duration_sec': round(float(elapsed_sec),2),
        'mean_people_per_frame': round(float(samples['total'].mean()),2),
        'peak_people_in_frame': int(samples['total'].max()),
        'total_detection_events': int(samples['total'].sum()),
        'mean_unassigned_per_frame': round(float(samples['outside_zones'].mean()),2),
        'busiest_zone_by_average': str(zs.iloc[0]['zone']),
        'busiest_zone_mean': float(zs.iloc[0]['mean_people']),
        'busiest_zone_peak': int(zs.iloc[0]['peak_people']),
        'busiest_minute_total': int(minute.loc[minute['total_mean'].idxmax(), 'minute_index']),
        'method': 'YOLO11 person detection on sampled frames; bounding-box bottom center assigned to a video-image zone',
        'limitation': 'Counts are estimates per sampled video frame, not ground-truth or unique visitors; no real-world people/m² without calibration.',
    }
    return zs,minute,summary


def _session_folder(root: str | Path | None = None) -> Path:
    base = Path(root) if root else RUNTIME_DIR
    base.mkdir(parents=True,exist_ok=True)
    target = base / ('session_' + time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:7])
    target.mkdir(parents=True,exist_ok=False)
    return target


def process_video(path: str | Path, zones: list[Zone], config: AnalysisConfig,
                  model: Any | None = None,
                  progress: Callable[[float, str], None] | None = None,
                  output_root: str | Path | None = None,
                  source_label: str | None = None) -> AnalysisResult:
    """Analyze a *single* local video with one detector, no calibration/fusion.

    Default 1 fps: 30-minute footage -> approximately 1800 sampled frames.
    This is the sampled timeline, not 1800 unique humans.
    """
    validate_zones(zones)
    if config.sample_fps <= 0 or config.max_duration_sec <= 0 or config.start_sec < 0:
        raise ValueError('Tham số thời gian hoặc FPS mẫu không hợp lệ.')
    if not (0 < config.confidence <= 1.0):
        raise ValueError('Độ tin cậy phải trong (0,1].')
    info = probe_video(path)
    start = int(config.start_sec * info.fps)
    if info.frame_count and start >= info.frame_count:
        raise ValueError('Thời điểm bắt đầu nằm ngoài video.')
    last = start + max(1,int(math.ceil(config.max_duration_sec*info.fps)))
    if info.frame_count:
        last = min(last,info.frame_count)
    step = max(1,round(info.fps/config.sample_fps))
    expected = max(1, math.ceil((last-start)/step))
    step_seconds = step / info.fps
    if model is None:
        model = load_model()
    # Reset any stateful tracker when a cached YOLO instance is reused for another video.
    if config.use_tracking:
        predictor = getattr(model, 'predictor', None)
        for tracker in getattr(predictor, 'trackers', []) if predictor is not None else []:
            reset = getattr(tracker, 'reset', None)
            if callable(reset):
                reset()
    folder = _session_folder(output_root)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('Không đọc được video.')
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    writer = None
    annotated_path = folder / 'video_da_danh_dau.mp4'
    if config.create_annotated_video:
        playback_fps = max(8., min(30., expected / 120.)) # Around two minutes for 30min footage at 1fps
        writer = cv2.VideoWriter(str(annotated_path), cv2.VideoWriter_fourcc(*'mp4v'),
                                 playback_fps, (info.width,info.height))
        if not writer.isOpened():
            writer.release()
            writer = None
            annotated_path = None
    else:
        annotated_path = None
    preview_indices = set(int(x) for x in np.linspace(0, max(0,expected-1),num=min(5,expected)))
    previews: list[Path] = []
    samples: list[dict] = []
    track_ids: set[int] = set()
    total_elapsed = 0
    last_frame_idx = None
    try:
        for frame_idx in range(start,last):
            if not cap.grab():
                break
            if (frame_idx - start) % step:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            h,w = frame.shape[:2]
            rx1,ry1,rx2,ry2 = _pixel_roi(frame,config.roi)
            boxes, confs, ids = detect_persons(model,frame[ry1:ry2,rx1:rx2],config)
            count = {z.name: 0 for z in zones}
            people: list[dict] = []
            outside = 0
            for box,conf,tid in zip(boxes,confs,ids):
                x1,y1,x2,y2 = (float(box[0])+rx1,float(box[1])+ry1,float(box[2])+rx1,float(box[3])+ry1)
                foot_x,foot_y = (x1+x2)*.5,y2
                z_index = assign_zone(foot_x,foot_y,w,h,zones)
                if z_index is None:
                    outside += 1
                else:
                    count[zones[z_index].name] += 1
                if int(tid) >= 0:
                    track_ids.add(int(tid))
                people.append({'xyxy':(x1,y1,x2,y2),'confidence':float(conf),
                               'track_id':int(tid),'foot_x':foot_x,'foot_y':foot_y,'zone_index':z_index})
            now = frame_idx / info.fps
            entry = {'sample_index':len(samples),'video_frame':frame_idx,'time_sec':round(now,3),
                     'time_label':f'{int(now//60):02d}:{int(now%60):02d}',
                     'total':len(people),'outside_zones':outside,**count}
            samples.append(entry)
            processed = len(samples)-1
            if processed in preview_indices or writer is not None:
                annotated = annotate_frame(frame,zones,count,people,config.roi)
                if processed in preview_indices:
                    snap = folder/f'khung_hinh_{processed:05d}_{int(now):04d}s.jpg'
                    cv2.imwrite(str(snap),annotated,[cv2.IMWRITE_JPEG_QUALITY,84])
                    previews.append(snap)
                if writer is not None:
                    writer.write(annotated)
            if progress and (len(samples)==1 or len(samples)%max(1,expected//100)==0):
                progress(min(1.,(frame_idx-start+1)/max(1,last-start)),
                         f'Đã quét {len(samples):,}/{expected:,} khung mẫu — phút {int(now//60)}:{int(now%60):02d} — {len(people)} người trong khung')
            last_frame_idx = frame_idx
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if not samples:
        raise ValueError('Không đọc được khung hình hợp lệ trong khoảng thời gian chọn.')
    # Browser-friendly H.264 if ffmpeg is installed. Fallback retains OpenCV MP4V file.
    if annotated_path is not None and annotated_path.is_file() and shutil.which('ffmpeg'):
        h264_path = folder / 'temp_h264.mp4'
        try:
            command = ['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y',
                       '-i',str(annotated_path),'-an','-c:v','libx264','-preset','veryfast',
                       '-pix_fmt','yuv420p','-movflags','+faststart',str(h264_path)]
            completed = subprocess.run(command,capture_output=True,check=False,timeout=600)
            if completed.returncode == 0 and h264_path.exists() and h264_path.stat().st_size > 0:
                os.replace(h264_path,annotated_path)
        except (OSError,subprocess.TimeoutExpired):
            pass
        finally:
            if h264_path.exists():
                h264_path.unlink()
    samples_df = pd.DataFrame(samples)
    effective_end = min(last/info.fps, (last_frame_idx + step)/info.fps) if last_frame_idx is not None else start/info.fps
    duration = max(0.,effective_end - samples_df['time_sec'].iloc[0])
    zones_df, minutes_df, summary = summarize(samples_df,zones,step_seconds,duration)
    summary.update({
        'source_filename':Path(source_label).name if source_label else Path(path).name,
        'video_duration_sec':round(info.duration_sec,2),
        'video_fps':round(info.fps,3),
        'sampling_fps_effective':round(info.fps/step,3),
        'zone_count':len(zones),
        'zones': [asdict(z) for z in zones],
        'roi':config.roi,
        'tracking_enabled':bool(config.use_tracking),
        'estimated_track_ids':len(track_ids) if config.use_tracking else None,
        'tracked_id_warning':'IDs can fragment/restart; not a validated count of unique people.' if config.use_tracking else None,
    })
    samples_df.to_csv(folder/'so_nguoi_theo_khung.csv',index=False,encoding='utf-8-sig')
    zones_df.to_csv(folder/'tong_hop_theo_khu.csv',index=False,encoding='utf-8-sig')
    minutes_df.to_csv(folder/'tong_hop_theo_phut.csv',index=False,encoding='utf-8-sig')
    (folder/'ket_qua.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    write_html_report(folder/'bao_cao_demo.html',samples_df,zones_df,minutes_df,summary,previews)
    (folder/'README_KET_QUA.txt').write_text(
        'VAA FlowVision — 1 camera cố định\n'
        'Chỉ số total = số người phát hiện trong mỗi khung hình đã lấy mẫu, không phải số người duy nhất.\n'
        'total_detection_events = tổng số lần phát hiện qua các khung hình, một người có thể được tính nhiều lần.\n'
        'Khu vực dùng vị trí chân (đáy giữa bounding box) theo tọa độ ảnh.\n'
        'Giá trị công suất/sức chứa do người dùng đặt là ngưỡng minh họa, KHÔNG phải giới hạn an toàn đã xác thực.\n'
        'Không dùng báo cáo làm căn cứ duy nhất để điều phối khẩn cấp.\n',encoding='utf-8'
    )
    zip_path = folder/'tai_xuong_bao_cao.zip'
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
        for file in sorted(folder.iterdir()):
            if file.is_file() and file.name != zip_path.name:
                zf.write(file,file.name)
    if progress:
        progress(1.,'Đã hoàn thành phân tích và xuất báo cáo.')
    return AnalysisResult(folder,info,config,samples_df,zones_df,minutes_df,summary,previews,zip_path,annotated_path)


def write_html_report(path: str | Path, samples: pd.DataFrame, zs: pd.DataFrame, minute: pd.DataFrame,
                      info: dict, previews: list[Path]) -> None:
    """Portable Vietnamese report, no CDN, includes real annotated stills from processed video."""
    esc = lambda x: html.escape(str(x))
    table_rows = ''.join(
        f'<tr><td>{esc(r.zone)}</td><td>{r.mean_people:.2f}</td><td>{r.peak_people}</td>'
        f'<td>{r.share_detected_pct:.1f}%</td><td>{r.over_capacity_pct:.1f}%</td>'
        f'<td>{esc(r.busiest_minute_label)}</td></tr>' for r in zs.itertuples(index=False)
    )
    vals = minute['total_mean'].astype(float).to_numpy() if len(minute) else np.array([])
    if len(vals):
        hi = max(1.,float(np.max(vals)))
        xs = np.linspace(38,876,num=len(vals))
        points = ' '.join(f'{x:.1f},{192 - (v / hi)*155:.1f}' for x,v in zip(xs,vals))
    else:
        points = ''
    imgs = ''
    for p in previews:
        encoded = base64.b64encode(p.read_bytes()).decode('ascii')
        imgs += f'<figure><img loading="lazy" src="data:image/jpeg;base64,{encoded}" alt="Ảnh có bounding box từ video thực"><figcaption>{esc(p.name)}</figcaption></figure>'
    report = f'''<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Báo cáo 1 camera — VAA FlowVision</title>
<style>
:root{{--ink:#132c4b;--blue:#1753a3;--line:#e1e9f2;--bg:#f4f7fb}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 Arial,sans-serif}}
main{{max-width:1100px;margin:auto;padding:32px 22px}}header{{background:#112a4a;color:#fff;padding:32px;border-radius:14px}}
h1{{font-size:30px;margin:0 0 6px}}h2{{font-size:20px;margin:28px 0 12px}}
.kpis{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0}}
.card,.panel{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px}}.card b{{display:block;font-size:29px;color:var(--blue)}}.card span{{font-size:13px}}
.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{background:#edf3fa}}
figure{{margin:12px 0}}figure img{{max-width:100%;border-radius:10px}}figcaption{{font-size:12px;color:#637994}}
.gallery{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}.caution{{border-left:4px solid #df9f32;padding:14px;background:#fff8e6}}
@media(max-width:680px){{.kpis{{grid-template-columns:repeat(2,1fr)}}.gallery{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header><p>VAA FLOWVISION / SINGLE CAMERA</p><h1>Báo cáo phân tích mật độ tương đối và số người theo khu vực</h1>
<p>Một camera cố định · Video nguồn: {esc(info['source_filename'])} · {info['sampled_frames']} khung hình được lấy mẫu</p></header>
<div class="kpis"><div class="card"><b>{info['mean_people_per_frame']:.2f}</b><span>Số người trung bình / khung hình</span></div>
<div class="card"><b>{info['peak_people_in_frame']}</b><span>Đỉnh số người tại cùng 1 khung hình</span></div>
<div class="card"><b>{esc(info['busiest_zone_by_average'])}</b><span>Khu đông nhất theo trung bình</span></div>
<div class="card"><b>{info['busiest_minute_total']:02d}:00</b><span>Phút có trung bình số người cao nhất</span></div></div>
<h2>So sánh từng khu</h2><div class="panel table-wrap"><table><thead><tr><th>Khu vực</th><th>Trung bình</th><th>Cao nhất</th><th>Tỷ trọng lượt phát hiện</th><th>% mẫu vượt sức chứa</th><th>Phút cao điểm khu</th></tr></thead><tbody>{table_rows}</tbody></table></div>
<h2>Diễn biến theo phút</h2><div class="panel"><svg viewBox="0 0 900 220" role="img" aria-label="Biểu đồ trung bình số người theo phút" style="max-width:100%"><line x1="38" y1="192" x2="886" y2="192" stroke="#91a4be"/><polyline points="{points}" fill="none" stroke="#1753a3" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/><text x="38" y="212" fill="#5d7189" font-size="12">Đầu video</text><text x="780" y="212" fill="#5d7189" font-size="12">Cuối khoảng phân tích</text></svg></div>
<h2>Khung hình minh chứng</h2><div class="gallery">{imgs}</div>
<h2>Cách diễn giải</h2><p class="caution">Số người được ước lượng tại <strong>mỗi khung hình lấy mẫu</strong>, không phải tổng khách duy nhất đến địa điểm. "Tổng lượt phát hiện" ({info['total_detection_events']}) có tính lặp một người ở nhiều khung hình. Vùng A/B/C do người dùng vẽ trên ảnh; không suy ra người/m² hay ngưỡng an toàn thực tế từ ảnh chưa hiệu chuẩn. Báo cáo tạo từ dữ liệu video thực đã xử lý, không có dữ liệu mô phỏng được thay vào.</p>
<footer><p>Độ dài khoảng đã quét: {info['analyzed_duration_sec']/60:.1f} phút · Tần số lấy mẫu thực tế: {info['sampling_fps_effective']:.2f} fps · Model YOLO11n pretrained, không xác nhận độ chính xác tại địa điểm nếu chưa đối chiếu nhãn thực.</p></footer>
</main></body></html>'''
    Path(path).write_text(report,encoding='utf-8')
