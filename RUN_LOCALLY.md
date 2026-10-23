# Running the GaRLIO pipeline locally

## 1. Setup
```
pip install -r requirements.txt
```

## 2. Run the unit tests (should show 53 passed)
```
PYTHONPATH=src python3 -m pytest tests/ -v
```

## 3. Extract your bag file (if not already done)
```
python3 scripts/extract_snail_radar.py /path/to/data1.bag real_data/extracted_data --max_seconds 15
```
This creates real_data/extracted_data/{imu.npy, radar.npz, lidar.npz}

## 4. Put ground truth in place
Copy ref_tls_T_xt32.csv and deviations.csv (from the Snail-Radar ref_trajs download,
sequence 20230920/data1) into:
```
real_data/ground_truth/
```

## 5. Run the demos
```
PYTHONPATH=src python3 scripts/demo_propagation.py
PYTHONPATH=src python3 scripts/demo_dynamic_removal.py
PYTHONPATH=src python3 scripts/demo_full_pipeline.py
PYTHONPATH=src python3 scripts/demo_real_data_pipeline.py
PYTHONPATH=src python3 scripts/evaluate_against_ground_truth.py
```
Output plots are saved to outputs/
