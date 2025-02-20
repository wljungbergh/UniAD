_base_ = ["../_base_/datasets/nus-3d.py",
          "../_base_/default_runtime.py",
          "./base_e2e.py"]

#./projects/work_dirs/stage1_track_map/base_track_map/dos-02202319/epoch_1.pth
load_from = "ckpts/uniad_base_track_map_uno.pth"