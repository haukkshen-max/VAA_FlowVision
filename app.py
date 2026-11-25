"""VAA FlowVision / 1 fixed camera / person occupancy per zone."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.fixed_camera import (
    AnalysisConfig, Zone, annotate_frame, build_partition_zones,
    first_frame, load_model, probe_video, process_video, validate_zones, zone_overlaps,
    MODEL_PATH, RUNTIME_DIR,
)

st.set_page_config(page_title='VAA FlowVision | 1 Camera',page_icon='📷',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
:root {--primary:#184b85;--ink:#173455;--muted:#657d98}
.block-container {max-width:1480px;padding-top:1.2rem;padding-bottom:3rem}
h1,h2,h3 {letter-spacing:-.018em}
[data-testid="stMetric"] {background:#f4f7fb;border:1px solid #dce6f1;border-radius:9px;padding:15px 18px}
[data-testid="stSidebar"] {border-right:1px solid #dce6f1}
.eyebrow {color:#537296;font-size:12px;font-weight:750;letter-spacing:2px;text-transform:uppercase}
.hero {padding:22px 24px;background:#122c4e;color:white;border-radius:12px;margin-bottom:20px}
.hero h1{color:white;font-size:30px;margin:5px 0 6px}.hero p{margin:0;color:#c5d8ef}
.pill {display:inline-block;padding:5px 10px;border:1px solid #b7cee6;border-radius:7px;color:#236092;font-weight:650;font-size:12px;margin:0 6px 5px 0}
.note{padding:12px 15px;border-left:4px solid #d5a243;background:#fff9eb;border-radius:3px;margin:10px 0}
hr{border-color:#e3eaf3}
</style>''',unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def detector():
    return load_model(MODEL_PATH)

@st.cache_data(show_spinner=False,max_entries=12)
def cached_preview(path: str, stat_mtime: float, stat_size: int, second: float = 0.0):
    return first_frame(path,second)


def file_from_uploader(upload) -> Path:
    """Copy via stream (not UploadedFile.getvalue which doubles upload RAM)."""
    filename = str(upload.name)
    upload.seek(0)
    sample = upload.read(262144)
    signature = (filename,int(upload.size),hashlib.sha256(sample).hexdigest())
    upload.seek(0)
    old = st.session_state.get('input_video_cache')
    if old and old['signature']==signature and Path(old['path']).exists():
        return Path(old['path'])
    if old and Path(old['path']).exists():
        try:
            Path(old['path']).unlink()
        except OSError:
            pass
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower() or '.mp4'
    output = RUNTIME_DIR / f'upload_{uuid.uuid4().hex[:10]}{suffix}'
    with output.open('wb') as dst:
        shutil.copyfileobj(upload,dst,length=4*1024*1024)
    st.session_state['input_video_cache'] = {'signature':signature,'path':str(output)}
    return output


def _format_minutes(seconds: float) -> str:
    return f'{int(seconds//60)} phút {int(seconds%60)} giây'


def zone_controls(count: int, layout: str):
    st.markdown('#### Phân vùng quan sát')
    col_names = st.columns(count)
    names = []
    caps = []
    for idx, col in enumerate(col_names):
        with col:
            names.append(st.text_input(f'Tên khu {idx+1}',value=f'Khu {chr(65+idx)}',key=f'name_{idx}'))
            caps.append(int(st.number_input(f'Ngưỡng minh họa {idx+1}',min_value=1,max_value=10000,value=15,step=1,key=f'cap_{idx}')))
    if layout != 'Tùy chỉnh hình chữ nhật':
        orientation = 'vertical' if layout=='Chia theo chiều ngang ảnh (trái → phải)' else 'horizontal'
        fractions = []
        st.caption('Điều chỉnh ranh giới theo % chiều rộng/chiều cao video để phù hợp thực địa.')
        prev = 0
        for j in range(1,count):
            low = prev+5
            high = 100 - 5*(count-j)
            initial = max(low,min(high,round(j*100/count)))
            edge = int(st.slider(f'Vạch chia {j} (%)',min_value=low,max_value=high,value=initial,key=f'edge_{layout}_{count}_{j}'))
            fractions.append(edge/100)
            prev = edge
        return build_partition_zones(names,caps,orientation,fractions)
    zones = []
    st.caption('Tọa độ chuẩn hóa 0–100% so với toàn khung hình; vùng có thể đặt đúng khu ghế chờ, cửa ra, quầy...')
    for i in range(count):
        with st.expander(f'{names[i]} — vị trí hình chữ nhật',expanded=(i==0)):
            colx1,coly1,colx2,coly2 = st.columns(4)
            a = int(colx1.number_input('X trái (%)',0,99,round(i*100/count),key=f'z{i}x1'))
            b = int(coly1.number_input('Y trên (%)',0,99,0,key=f'z{i}y1'))
            c = int(colx2.number_input('X phải (%)',1,100,round((i+1)*100/count),key=f'z{i}x2'))
            d = int(coly2.number_input('Y dưới (%)',1,100,100,key=f'z{i}y2'))
            zones.append(Zone(names[i],a/100,b/100,c/100,d/100,caps[i]))
    validate_zones(zones)
    return zones


st.sidebar.markdown('### <span style="color:#275789">VAA FLOWVISION</span>',unsafe_allow_html=True)
st.sidebar.caption('ONE CAMERA / ZONE ANALYTICS')
st.sidebar.divider()
st.sidebar.subheader('01  ·  Nguồn video')
source_type = st.sidebar.radio('Chọn nguồn',('Tải video lên','Đường dẫn video trên máy'),label_visibility='collapsed')
video_path = None
file_label = ''
if source_type=='Tải video lên':
    upload = st.sidebar.file_uploader('Video từ 1 camera',type=['mp4','avi','mov','mkv','m4v'])
    if upload is not None:
        try:
            video_path = file_from_uploader(upload)
            file_label = str(upload.name)
        except Exception as exc:
            st.sidebar.error(f'Lỗi lưu video: {exc}')
else:
    p = st.sidebar.text_input('Đường dẫn đầy đủ của video',placeholder=r'C:\Videos\camera_30min.mp4')
    if p.strip():
        candidate = Path(p.strip().strip('"')).expanduser()
        if candidate.is_file():
            video_path = candidate
            file_label = candidate.name
        else:
            st.sidebar.warning('Chưa tìm thấy video. Đường dẫn phải nằm trên máy đang chạy ứng dụng.')

st.sidebar.divider()
st.sidebar.subheader('02  ·  Cấu hình AI')
sample_fps = st.sidebar.select_slider('Tốc độ quét (khung hình/giây)',options=[0.5,1.,2.,3.,5.],value=1.)
conf = st.sidebar.slider('Ngưỡng nhận diện người',min_value=.1,max_value=.8,value=.25,step=.05)
img_size = st.sidebar.select_slider('Độ phân giải suy luận',options=[480,640,960,1280],value=640)
max_minutes = int(st.sidebar.number_input('Thời lượng cần phân tích (phút)',min_value=1,max_value=180,value=30,step=1))
start_minute = float(st.sidebar.number_input('Bắt đầu từ phút thứ',min_value=0.,max_value=360.,value=0.,step=1.))
use_tracking = st.sidebar.checkbox('Theo dõi Track ID (thử nghiệm)',value=False,help='Track ID có thể thay đổi khi người bị che khuất hoặc lấy mẫu thưa; không coi đây là số khách duy nhất.')
create_video = st.sidebar.checkbox('Xuất video mẫu có ô nhận diện',value=False,help='Video đầu ra chỉ gồm các khung hình đã quét, phát nhanh và có thể tăng dung lượng báo cáo.')
device = st.sidebar.selectbox('Thiết bị',options=['auto','cpu','0'],format_func=lambda x:{'auto':'Tự động (GPU nếu có)','cpu':'CPU','0':'GPU CUDA 0'}[x])

st.markdown('''<div class="hero"><div class="eyebrow" style="color:#8fc4ef">VAA / CROWDFLOW</div><h1>1 Camera · Phân tích đám đông theo khu vực</h1><p>Quét một góc quay cố định → phát hiện người trong khung hình → thống kê Khu A, B, C → tìm khu đông và thời điểm cao điểm.</p></div>''',unsafe_allow_html=True)
st.markdown('<span class="pill">1 camera duy nhất</span><span class="pill">Video 30 phút hoặc dài hơn</span><span class="pill">YOLO11n / Person</span><span class="pill">Xuất báo cáo thật</span>',unsafe_allow_html=True)

if video_path is None:
    st.info('Bước 1: chọn video ở thanh bên trái. App hỗ trợ video camera cố định, không cần camera thứ hai hoặc hiệu chuẩn đa camera.')
    st.markdown('**Ví dụ cách hệ thống hoạt động:** Một đoạn video 30 phút, lấy mẫu 1 khung hình/giây, tương ứng khoảng **1.800 lần quét**. Kết quả sẽ cho biết trung bình một khung hình có bao nhiêu người, đỉnh đông nhất, khu nào đông hơn và phút cao điểm. Đây là ví dụ số *khung hình*, không phải số người đã đo được.')
    st.stop()

try:
    info = probe_video(video_path)
    st.markdown('### Video đầu vào')
    a,b,c,d = st.columns(4)
    a.metric('Nguồn video', '01 camera')
    b.metric('Thời lượng video',_format_minutes(info.duration_sec) if info.frame_count else 'Không rõ')
    c.metric('Độ phân giải',f'{info.width} × {info.height}')
    d.metric('FPS gốc',f'{info.fps:.1f}')
    if not info.fps_from_metadata:
        st.warning('Không đọc được FPS từ metadata, đang dùng 25 FPS tạm tính; mốc phút có thể sai, nên chuyển video sang MP4/H.264 có metadata chuẩn.')
    if info.frame_count and start_minute*60 >= info.duration_sec:
        st.error('Thời gian bắt đầu vượt quá thời lượng video. Hãy sửa mục “Bắt đầu từ phút thứ”.')
        st.stop()
    end_sec = min(info.duration_sec, start_minute*60 + max_minutes*60) if info.frame_count else start_minute*60+max_minutes*60
    estimate_samples = max(1,int((end_sec-start_minute*60) * min(info.fps,sample_fps)))
    st.caption(f'Video: {file_label} · Cửa sổ dự kiến {_format_minutes(max(0,end_sec-start_minute*60))} · xấp xỉ {estimate_samples:,} khung mẫu ({sample_fps:g} FPS); thời gian thực thi phụ thuộc GPU/CPU.')
except Exception as exc:
    st.error(f'Không mở được video: {exc}')
    st.stop()

configure_tab, analysis_tab, report_tab = st.tabs(['01  ·  Khoanh vùng & chạy AI','02  ·  Kết quả phân tích','03  ·  Xuất báo cáo demo'])
with configure_tab:
    layout = st.radio('Cách chia không gian trong 1 góc quay',
                      ('Chia theo chiều ngang ảnh (trái → phải)','Chia theo chiều dọc ảnh (trên → dưới)','Tùy chỉnh hình chữ nhật'),horizontal=True)
    count = st.selectbox('Số khu vực cần theo dõi',options=[2,3,4],index=1)
    try:
        zones = zone_controls(count,layout)
        validate_zones(zones)
        overlaps = zone_overlaps(zones)
        if overlaps:
            st.warning('Các khu giao nhau: '+', '.join(f'{a} / {b}' for a,b in overlaps)+'. Người ở vùng giao chỉ được tính vào khu khai báo trước; nên chỉnh lại để không bị chồng lấn.')
        left,right = st.columns([1.65,1],vertical_alignment='top')
        with right:
            st.markdown('#### Vùng quan sát của camera')
            crop_enabled = st.checkbox('Giới hạn quét trong ROI',value=False,help='Chỉ phát hiện người trong hình chữ nhật ROI; khu A/B/C vẫn đặt theo toàn bộ hình.')
            roi = None
            if crop_enabled:
                x1 = st.slider('ROI trái (%)',0,90,0)
                x2 = st.slider('ROI phải (%)',x1+1,100,100)
                y1 = st.slider('ROI trên (%)',0,90,0)
                y2 = st.slider('ROI dưới (%)',y1+1,100,100)
                roi = (x1/100,y1/100,x2/100,y2/100)
            st.caption('Ô màu phân chia trong tọa độ ảnh. Người được gán theo điểm giữa cạnh dưới của hộp phát hiện (vị trí chân ước lượng).')
            st.markdown('**Quy trình thực tế**')
            st.markdown('① Chọn video → ② Đặt khu vực → ③ Quét YOLO → ④ Xem biểu đồ → ⑤ Xuất ZIP.')
        with left:
            frame = cached_preview(str(video_path),video_path.stat().st_mtime,video_path.stat().st_size,start_minute*60)
            rendered = annotate_frame(frame,zones,roi=roi)
            st.image(cv2.cvtColor(rendered,cv2.COLOR_BGR2RGB),caption='Xem trước vị trí các khu vực trên chính camera đã chọn',use_container_width=True)
        st.markdown('<div class="note">Kết quả là <b>ước lượng số người trong từng khung hình lấy mẫu</b>. Hệ thống không nhận dạng danh tính, không tự xác định số người duy nhất trong 30 phút, và không đo người/m² khi camera chưa được hiệu chuẩn.</div>',unsafe_allow_html=True)
        current_cfg = AnalysisConfig(confidence=conf,image_size=img_size,sample_fps=sample_fps,
                                     start_sec=start_minute*60,max_duration_sec=max_minutes*60,
                                     device=device,use_tracking=use_tracking,create_annotated_video=create_video,roi=roi)
        current_config_signature = repr((zones,current_cfg))
        start_run = st.button('▶  BẮT ĐẦU PHÂN TÍCH VIDEO',type='primary',use_container_width=True)
        if start_run:
            progressbar = st.progress(0,text='Đang chuẩn bị mô hình nhận diện...')
            try:
                model = detector()
                result = process_video(video_path,zones,current_cfg,model=model,
                                       progress=lambda value,message:progressbar.progress(value,text=message),
                                       source_label=file_label)
                # Keep human-friendly source filename in visible UI (internal session path still stored on disk).
                result.summary['display_filename'] = file_label
                st.session_state['analysis_result'] = result
                st.session_state['analysis_source'] = str(video_path)
                st.session_state['analysis_config_signature'] = current_config_signature
                st.success('Đã phân tích xong. Mở tab “02 · Kết quả phân tích” hoặc “03 · Xuất báo cáo demo”.')
            except Exception as exc:
                progressbar.empty()
                st.error(f'Phân tích thất bại: {type(exc).__name__}: {exc}')
                st.info('Gợi ý: hạ độ phân giải 640 hoặc 480, lấy mẫu 0.5 FPS trên CPU, kiểm tra video mở được bằng VLC và cài đủ requirements.txt.')
    except ValueError as exc:
        st.error(str(exc))

result = st.session_state.get('analysis_result')
with analysis_tab:
    if result is None:
        st.info('Chưa có kết quả. Chuyển sang tab 01, khoanh vùng và bấm “Bắt đầu phân tích video”.')
    else:
        if st.session_state.get('analysis_source')!=str(video_path):
            st.warning('Kết quả bên dưới thuộc video đã quét trước đó. Hãy chạy lại để cập nhật video mới.')
        elif st.session_state.get('analysis_config_signature')!=locals().get('current_config_signature'):
            st.warning('Bạn đã thay đổi phân khu hoặc cấu hình sau lần quét. Kết quả này thuộc cấu hình cũ; hãy chạy lại để cập nhật.')
        s = result.summary
        st.markdown('### Tổng quan sau khi quét video')
        a,b,c,d = st.columns(4)
        a.metric('Trung bình / khung',f"{s['mean_people_per_frame']:g} người")
        b.metric('Đỉnh đồng thời',f"{s['peak_people_in_frame']} người")
        c.metric('Khu đông nhất',s['busiest_zone_by_average'])
        d.metric('Phút cao điểm',f"{s['busiest_minute_total']:02d}:00")
        st.caption(f"Đã xử lý {s['sampled_frames']:,} khung hình lấy mẫu / {_format_minutes(s['analyzed_duration_sec'])}. Số lượt nhận diện cộng dồn: {s['total_detection_events']:,}; một người có thể được tính lại ở các khung khác.")
        if s.get('tracking_enabled'):
            st.warning(f"Đếm được {s['estimated_track_ids']} Track ID trong các khung lấy mẫu. Đây chỉ là số ID ước lượng, KHÔNG phải số người duy nhất do mất bám vết và phân mảnh ID.")
        st.divider()
        st.markdown('### Khu vực nào đông hơn?')
        zs = result.zone_summary.copy()
        labels = zs['zone'].tolist()
        fig = go.Figure()
        fig.add_bar(x=labels,y=zs['mean_people'],name='Số người trung bình',marker_color='#21578e')
        fig.add_bar(x=labels,y=zs['peak_people'],name='Đỉnh trong một khung',marker_color='#25a596')
        fig.update_layout(barmode='group',height=365,margin=dict(t=25,l=10,b=15,r=10),yaxis_title='Người / khung hình',legend_orientation='h')
        st.plotly_chart(fig,use_container_width=True)
        named = zs.rename(columns={'zone':'Khu vực','mean_people':'TB người / khung','peak_people':'Đỉnh người','total_detection_events':'Tổng lượt phát hiện lặp','share_detected_pct':'Tỷ trọng lượt phát hiện (%)','capacity_reference':'Ngưỡng minh họa','over_capacity_samples':'Số mẫu vượt ngưỡng','over_capacity_pct':'Tỷ lệ mẫu vượt ngưỡng (%)','estimated_over_capacity_minutes':'Phút vượt ngưỡng ước tính','busiest_minute_label':'Khoảng phút cao điểm','busiest_minute_index':'Chỉ số phút cao điểm'})
        st.dataframe(named.drop(columns=['Chỉ số phút cao điểm']),use_container_width=True,hide_index=True)
        if float(zs['over_capacity_pct'].max())>0:
            z = zs.loc[zs['over_capacity_pct'].idxmax()]
            st.warning(f"{z['zone']} vượt ngưỡng tham chiếu {int(z['capacity_reference'])} người ở khoảng {z['over_capacity_pct']}% khung được lấy mẫu. Ngưỡng này do người dùng đặt, không phải ngưỡng an toàn đã kiểm định.")
        st.markdown('### Diễn biến số người theo phút')
        minute = result.minute_summary
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=minute['minute_index'],y=minute['total_mean'],name='Tổng số người TB',mode='lines+markers',line=dict(color='#163f75',width=3)))
        for i,z in enumerate(result.zone_summary['zone'].tolist()):
            fig2.add_trace(go.Scatter(x=minute['minute_index'],y=minute[f'{z}_mean'],name=z,mode='lines',line=dict(width=2)))
        fig2.update_layout(height=380,margin=dict(l=10,r=10,t=20,b=10),xaxis_title='Phút tính từ đầu video',yaxis_title='Người trung bình / khung',legend_orientation='h',hovermode='x unified')
        st.plotly_chart(fig2,use_container_width=True)
        st.markdown('### Bản đồ nhiệt theo phút và khu vực')
        zones_present = [z['name'] for z in result.summary['zones']]
        fig3 = go.Figure(go.Heatmap(z=[minute[f'{name}_mean'].to_list() for name in zones_present],x=minute['minute_index'],y=zones_present,
                                    colorscale='Blues',colorbar=dict(title='Người TB'),hovertemplate='Khu %{y}<br>Phút %{x}<br>%{z:.1f} người<extra></extra>'))
        fig3.update_layout(height=260+40*len(zones_present),margin=dict(l=12,r=12,t=16,b=10),xaxis_title='Phút tính từ đầu video')
        st.plotly_chart(fig3,use_container_width=True)
        st.markdown('### Khung hình kiểm chứng (trích từ video đã quét)')
        cols = st.columns(min(3,max(1,len(result.previews))))
        for i,p in enumerate(result.previews):
            with cols[i%len(cols)]:
                st.image(str(p),caption=p.name,use_container_width=True)
        if result.annotated_video is not None and result.annotated_video.exists():
            st.markdown('### Video demo có đánh dấu người')
            st.caption('Chỉ gồm khung hình đã quét; phát tua nhanh để xem các thời điểm trong video, không phải bản ghi nguyên tốc độ.')
            st.video(str(result.annotated_video))
        st.info('Video 30 phút không có nghĩa là 30 phút xử lý liên tục theo thời gian thực: hệ thống phân tích offline, lấy mẫu các khung dọc video để tiết kiệm thời gian. Kết quả chỉ chính xác đến mức mô hình và tần suất lấy mẫu cho phép.')

with report_tab:
    if result is None:
        st.info('Chưa có dữ liệu để xuất báo cáo. Hãy thực hiện phân tích tại tab 01.')
    else:
        s=result.summary
        if st.session_state.get('analysis_source')!=str(video_path) or st.session_state.get('analysis_config_signature')!=locals().get('current_config_signature'):
            st.warning('Đây là kết quả của lần phân tích trước. Chạy lại video với cấu hình hiện tại để xuất báo cáo mới.')
        st.markdown('### Kịch bản demo có số liệu thật từ video')
        demo = (f"Hệ thống sử dụng một camera cố định để phân tích video {s['analyzed_duration_sec']/60:.1f} phút, "
                f"đã quét {s['sampled_frames']:,} khung hình. Trung bình, mỗi khung hình có {s['mean_people_per_frame']} người, "
                f"cao nhất là {s['peak_people_in_frame']} người cùng lúc. Khu đông nhất là {s['busiest_zone_by_average']}, "
                f"trung bình {s['busiest_zone_mean']} người và có lúc đạt {s['busiest_zone_peak']} người. "
                f"Khoảng cao điểm của toàn khung hình nằm gần phút thứ {s['busiest_minute_total']}. "
                'Đây là số người hiện diện ước lượng trong những khung hình lấy mẫu, không phải tổng số khách khác nhau đã đi qua.')
        st.write(demo)
        st.download_button('⬇ Tải toàn bộ báo cáo ZIP (CSV + HTML + ảnh + JSON)',
                           data=result.zip_path.read_bytes(),file_name=f'VAA_FlowVision_1Cam_{result.session_dir.name}.zip',
                           mime='application/zip',type='primary',use_container_width=True)
        col1,col2,col3=st.columns(3)
        with col1:
            st.download_button('Báo cáo HTML',data=(result.session_dir/'bao_cao_demo.html').read_bytes(),file_name='bao_cao_1_camera.html',mime='text/html',use_container_width=True)
        with col2:
            st.download_button('Thống kê từng phút CSV',data=(result.session_dir/'tong_hop_theo_phut.csv').read_bytes(),file_name='thong_ke_tung_phut.csv',mime='text/csv',use_container_width=True)
        with col3:
            st.download_button('Thống kê khu vực CSV',data=(result.session_dir/'tong_hop_theo_khu.csv').read_bytes(),file_name='thong_ke_tung_khu.csv',mime='text/csv',use_container_width=True)
        st.markdown('**Trong gói ZIP có:** số người theo khung hình, tổng hợp từng phút, tổng hợp từng khu, 5 ảnh có khung nhận diện, báo cáo HTML đọc offline, JSON metadata, và video đã đánh dấu nếu bạn chọn xuất.')
        st.caption('Dữ liệu chạy và video bạn tải lên nằm trên máy chủ Streamlit đang sử dụng (thường là máy cá nhân). Có thể xóa thư mục data/runtime khi không còn cần dữ liệu; video gốc không được đưa vào ZIP kết quả.')

st.divider()
st.caption('VAA FlowVision — Phiên bản 1 camera cố định • YOLO11n pretrained • Không nhận dạng khuôn mặt / danh tính • Báo cáo dùng cho demo, chưa thay thế đo đạc hiện trường hay đánh giá an toàn chính thức.')
