class DefaultConfig(object):

    data_dir = '/YOUR_HOME_DIR/' # fill in your own directory
    train_data_root = data_dir+"dataset/"
    test_data_root = data_dir+"dataset/"
    load_model = False
    loaded_model_name = 'NAFNet1'
    load_model_path = data_dir+"checkpoints/nafnet1_stylegan2.pth"
    save_model = True
    save_model_path = data_dir+"checkpoints/nafnet1_stylegan2.pth"
    rnan_load_model_path = data_dir+"checkpoints/rnan_model_deeper.pth"
    rnan_save_model_path = data_dir+"checkpoints/rnan_model_deeper.pth"
    resnext_load_model_path = data_dir+"checkpoints/resnext_model_deeper.pth"
    resnext_save_model_path = data_dir+"checkpoints/resnext_model_deeper.pth"

    seed = 43
    batch_size = 1 #14
    use_gpu = True
    gpu_id = '3'
    trainer = 'combine'
    num_workers = 4
    print_iter = False # print training info every print_freq epochs
    print_freq = 20
    img_size = 112
    optimizer = 'adam' #'sgd'
    use_sam = False

    max_epoch = 450
    lr = 0.001
    lr_decay = 0.96
    weight_decay = 0

    #resnext
    cardinality = 8
    depth = 29
    base_width = 64
    widen_factor = 4
    nlabels = 2

    #rnan
    n_low_body = 1
    n_resgroups = 5
    n_high_body = 1
    n_resblocks = 16
    n_feats = 64
    reduction = 16
    scale = [1]
    n_colors = 3
    res_scale = 1

    # nablock
    n_nablock   = 2
    # xception style
    in_channels = 32
    inter_channels = 64
    out_channels = 32
    n_blocks = 3
    reps = 3
    # dense style
    n_layers    = [6, 12, 24, 6]
    growth_rate = 12
    # res style
    res_n_layers= [2, 2, 2]

    hid_dim     = 128 #64
    n_heads     = 8
    dropout     = 0.0#0.3

    # dataset and model configurations
    # check load and save model configurations
    model_name  = 'NAFNet1'
    mid_loss_weight = 0.5
    dataset     = 'stylegan2'
    train_noise = None
    test_noise  = None
    noise_scale = 0

opt = DefaultConfig()
