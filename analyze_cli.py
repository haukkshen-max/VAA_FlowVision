"""CLI for processing long video files outside the Streamlit UI."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from src.fixed_camera import AnalysisConfig, Zone, build_partition_zones, load_model, process_video


def parse_args():
    p=argparse.ArgumentParser(description='VAA FlowVision — 1 camera / people counting by image zones')
    p.add_argument('--video',required=True,help='MP4/AVI/MOV/MKV path from ONE fixed camera')
    p.add_argument('--duration-min',type=float,default=30,help='Minutes of video to analyze (default 30)')
    p.add_argument('--start-min',type=float,default=0,help='Start at minute in the video')
    p.add_argument('--sample-fps',type=float,default=1.,help='Sampling frequency, default 1 FPS')
    p.add_argument('--confidence',type=float,default=.25)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--device',choices=['auto','cpu','0'],default='auto')
    p.add_argument('--zones-json',type=str,help='Optional file with a list of rectangular zone objects')
    p.add_argument('--orientation',choices=['vertical','horizontal'],default='vertical')
    p.add_argument('--tracking',action='store_true',help='Experimental track IDs, NOT accurate unique persons')
    p.add_argument('--export-video',action='store_true',help='Export sampled annotated frames to MP4')
    p.add_argument('--model',type=str,help='Override bundled YOLO11n model path')
    p.add_argument('--output',type=str,default='data/runtime',help='Output folder')
    return p.parse_args()


def main():
    args=parse_args()
    if args.zones_json:
        data=json.loads(Path(args.zones_json).read_text(encoding='utf-8'))
        zones=[Zone(**item) for item in data]
    else:
        zones=build_partition_zones(['Khu A','Khu B','Khu C'],[15,15,15],args.orientation)
    cfg=AnalysisConfig(confidence=args.confidence,image_size=args.imgsz,sample_fps=args.sample_fps,
                       start_sec=args.start_min*60,max_duration_sec=args.duration_min*60,
                       device=args.device,use_tracking=args.tracking,create_annotated_video=args.export_video)
    model=load_model(args.model)
    latest={'bucket':-1}
    def progress(value,message):
        step=int(value*20)
        if step!=latest['bucket']:
            print(f'{value*100:5.1f}% | {message}',flush=True)
            latest['bucket']=step
    result=process_video(args.video,zones,cfg,model=model,progress=progress,
                         output_root=args.output,source_label=Path(args.video).name)
    print('\n=== KẾT QUẢ THỰC TỪ VIDEO ===')
    print(json.dumps({k:result.summary[k] for k in ['sampled_frames','mean_people_per_frame',
             'peak_people_in_frame','busiest_zone_by_average','busiest_minute_total','total_detection_events']},
             ensure_ascii=False,indent=2))
    print('Báo cáo:',result.zip_path.resolve())
    print('LƯU Ý: total_detection_events là lượt phát hiện CÓ LẶP, không phải số người duy nhất.')

if __name__=='__main__':
    main()
