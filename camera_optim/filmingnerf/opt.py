import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, 
                        default= "./CameraExtractor/data")
    parser.add_argument('--seq_name', type=str,
                        default='follow')
    parser.add_argument('--out_dir', type=str, 
                        default= "./CameraExtractor/seq_output")
    
    parser.add_argument('--t', type=int,
                        default=0)
    parser.add_argument('--start_id', type=int,
                        default=0)
    parser.add_argument('--end_id', type=int,
                        default=180)
    parser.add_argument('--track_sid', type=int,
                        default=1)
    parser.add_argument('--track_eid', type=int,
                        default=2)
    
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--render_n_samples', type=int, 
                        default= 100)
    parser.add_argument('--render_camera_radius', type=float,
                        default=3)
    parser.add_argument('--near_far', type=tuple,
                        default=(1, 20))
    parser.add_argument('--nerf_train_img_hw', type=tuple,
                        default=(64, 150))
    parser.add_argument('--nerf_render_img_hw', type=int, nargs='+',
                        default=[64, 150])
# (64, 150) (108, 144) (108, 192),(64,128)
    parser.add_argument('--downsample', type=float,
                    default=10)
    parser.add_argument('--aabb_scale', type=int,
                        default=1)
    
    parser.add_argument('--nerf_train_epochs', type=int,
                        default=20)
    
    parser.add_argument('--cam_optim_setps', type=int,
                        default=300)
    

    
    
    return parser.parse_args()