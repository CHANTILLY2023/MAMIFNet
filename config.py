import os
root = './data'
datasets_root = os.path.join(root,'COD_dataset')
cod_training_root = os.path.join(datasets_root, 'TrainDataset')
chameleon_path = os.path.join(datasets_root, 'TestDataset/CHAMELEON')
camo_path = os.path.join(datasets_root, 'TestDataset/CAMO')
cod10k_path = os.path.join(datasets_root, 'TestDataset/COD10K')
nc4k_path = os.path.join(datasets_root, 'TestDataset/NC4K')
pvtv2_checkpoint_dir = 'PVTv2_Seg/checkpoint/'