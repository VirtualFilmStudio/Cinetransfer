def check_params(cfg):
    seq_name = cfg.seq_name
    if seq_name in ['arc2', 'follow2']:
        data_root = '../data'
        track_sid = 0
        track_eid = 1
        nerf_train_img_hw = (64, 150)
        nerf_render_img_hw = [64, 150]
        downsample = 16.9
        cfg.end_id = 66
    elif seq_name in ['shot1']:
        data_root = '../data3'
        track_sid = 0
        track_eid = 2
        nerf_train_img_hw = (64, 150)
        nerf_render_img_hw = [64, 150]
        downsample = 27.3
    elif seq_name in ['shot5']:
        data_root = '../data3'
        track_sid = 0
        track_eid = 4
        nerf_train_img_hw = (64, 150)
        nerf_render_img_hw = [64, 150]
        downsample = 27.3
    elif seq_name in ['shot2', 'shot8', 'shot10', 'shot12']:
        data_root = '../data3'
        track_sid = 0
        track_eid = 1
        nerf_train_img_hw = (64, 150)
        nerf_render_img_hw = [64, 150]
        downsample = 27.3
    elif seq_name in ['shot4', 'shot11']:
        data_root = '../data3'
        track_sid = 0
        track_eid = 1
        nerf_train_img_hw = (108, 192)
        nerf_render_img_hw = [108, 192]
        downsample = 20
    elif seq_name in ['shot7']:
        data_root = '../data3'
        track_sid = 0
        track_eid = 2
        nerf_train_img_hw = (108, 144)
        nerf_render_img_hw = [108, 144]
        downsample = 20
    elif seq_name in ['014960_mpii']:
        data_root = '../data5'
        track_sid = 1
        track_eid = 2
        nerf_train_img_hw = (68, 128)
        nerf_render_img_hw = [68, 128]
        downsample = 10
    elif seq_name in ['014384_mpii']:
        data_root = '../data5'
        track_sid = 0
        track_eid = 1
        nerf_train_img_hw = (72, 128)
        nerf_render_img_hw = [72, 128]
        downsample = 10
    elif seq_name in ['lalaland-4']:
        data_root = '../data4'
        track_sid = 0
        track_eid = 1
        nerf_train_img_hw = (64, 150)
        nerf_render_img_hw = [64, 150]
        downsample = 27.3
    elif seq_name in ['lalaland-3', 'lalaland-5']:
        data_root = '../data4'
        track_sid = 0
        track_eid = 4
        nerf_train_img_hw = (64, 150)
        nerf_render_img_hw = [64, 150]
        downsample = 27.3
    elif seq_name in ['lalaland-7','lalaland-8']:
        data_root = '../data4'
        track_sid = 0
        track_eid = 2
        nerf_train_img_hw = (64, 150)
        nerf_render_img_hw = [64, 150]
        downsample = 27.3
    elif seq_name == 'lalaland-dance_1':
        data_root = '../data4'
        track_sid = 0
        track_eid = 2
        nerf_train_img_hw = (64, 150)
        nerf_render_img_hw = [64, 150]
        downsample = 27.3
    elif 'lalaland-dance' in seq_name:
        data_root = '../data4'
        track_sid = 0
        track_eid = 2
        nerf_train_img_hw = (64, 150)
        nerf_render_img_hw = [64, 150]
        downsample = 16.9
    elif seq_name in ['nolan-1','nolan-11', 'nolan-12', 'nolan-15']:
        data_root = '../data6'
        track_sid = 0
        track_eid = 1
        nerf_train_img_hw = (64, 150)
        nerf_render_img_hw = [64, 150]
        downsample = 27.3
    elif seq_name == 'nolan-2':
        data_root = '../data6'
        track_sid = 3
        track_eid = 4
        nerf_train_img_hw = (108, 192)
        nerf_render_img_hw = [108, 192]
        downsample = 20
    elif seq_name == 'nolan-9':
        data_root = '../data6'
        track_sid = 3
        track_eid = 4
        nerf_train_img_hw = (108, 192)
        nerf_render_img_hw = [108, 192]
        downsample = 20
    elif seq_name in ['nolan-3', 'nolan-5', 'nolan-7', 'nolan-8','nolan-10', 'nolan-17', 'nolan-21', 'nolan-22', 'nolan-24']:
        data_root = '../data6'
        track_sid = 0
        track_eid = 1
        nerf_train_img_hw = (108, 192)
        nerf_render_img_hw = [108, 192]
        downsample = 20
    elif seq_name == 'nolan-4':
        data_root = '../data6'
        track_sid = 2
        track_eid = 3
        nerf_train_img_hw = (108, 192)
        nerf_render_img_hw = [108, 192]
        downsample = 20
    elif seq_name in ['nolan-14', 'nolan-18', 'nolan-23']:
        data_root = '../data6'
        track_sid = 0
        track_eid = 2
        nerf_train_img_hw = (108, 192)
        nerf_render_img_hw = [108, 192]
        downsample = 20
    elif seq_name in ['nolan-6']:
        data_root = '../data6'
        track_sid = 0
        track_eid = 1
        nerf_train_img_hw = (108, 192)
        nerf_render_img_hw = [108, 192]
        downsample = 20
        cfg.end_id = 146
    elif seq_name == 'nolan-16':
        data_root = '../data6'
        track_sid = 0
        track_eid = 1
        start_id = 55
        nerf_train_img_hw = (64, 150)
        nerf_render_img_hw = [64, 150]
        downsample = 27.3
        cfg.start_id = start_id
        cfg.t = start_id
        cfg.end_id = 124
    elif seq_name in ['nolan-19']:
        data_root = '../data6'
        track_sid = 0
        track_eid = 1
        # start_id = 20
        nerf_train_img_hw = (64, 150)
        nerf_render_img_hw = [64, 150]
        downsample = 27.3
        # cfg.t = start_id
        cfg.end_id = 125
    elif seq_name in ['nolan-13']:
        data_root = '../data6'
        track_sid = 0
        track_eid = 3
        nerf_train_img_hw = (108, 192)
        nerf_render_img_hw = [108, 192]
        downsample = 20
    elif seq_name == 'matrix':
        data_root = '../data8'
        track_sid = 0
        track_eid = 2
        nerf_train_img_hw = (108, 192)
        nerf_render_img_hw = [108, 192]
        downsample = 20
    elif seq_name in ['inception']:
        data_root = '../data8'
        track_sid = 0
        track_eid = 1
        nerf_train_img_hw = (64, 150)
        nerf_render_img_hw = [64, 150]
        downsample = 27.3
        cfg.end_id = 110
    elif seq_name == 'add1':
        data_root = '../data9'
        track_sid = 0
        track_eid = 3
        nerf_train_img_hw = (108, 192)
        nerf_render_img_hw = [108, 192]
        downsample = 20
    elif seq_name in ['add2','add3','add4']:
        data_root = '../data9'
        track_sid = 0
        track_eid = 3
        nerf_train_img_hw = (108, 144)
        nerf_render_img_hw = [108, 144]
        downsample = 20
    elif seq_name == 'add5':
        data_root = '../data9'
        track_sid = 0
        track_eid = 1
        nerf_train_img_hw = (108, 192)
        nerf_render_img_hw = [108, 192]
        downsample = 20
    elif seq_name in ['demo11', 'demo13', 'demo17', 'demo18']:
        data_root = '../data10'
        track_sid = 0
        track_eid = 5
        nerf_train_img_hw = (108, 192)
        nerf_render_img_hw = [108, 192]
        downsample = 20
    elif seq_name in ['demo19']:
        data_root = '../data10'
        track_sid = 2
        track_eid = 3
        nerf_train_img_hw = (108, 192)
        nerf_render_img_hw = [108, 192]
        downsample = 20
    elif seq_name in ['demo22', 'demo26', 'demo30']:
        data_root = '../data10'
        track_sid = 0
        track_eid = 2
        nerf_train_img_hw = (64, 150)
        nerf_render_img_hw = [64, 150]
        downsample = 27.3
    elif seq_name in ['demo31']:
        data_root = '../data10'
        track_sid = 0
        track_eid = 2
        nerf_train_img_hw = (64, 150)
        nerf_render_img_hw = [64, 150]
        downsample = 27.3

    cfg.data_root = data_root
    cfg.track_sid = track_sid
    cfg.track_eid = track_eid
    cfg.nerf_train_img_hw = nerf_train_img_hw
    cfg.nerf_render_img_hw = nerf_render_img_hw
    cfg.downsample = downsample
    return cfg