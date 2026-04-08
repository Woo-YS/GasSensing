import os
import glob
import pickle

import numpy as np
import pandas as pd

import time

SEED = 42


def build_samples(data_type):
    """
    [Original Style] PPM을 로드하지 않고 X, y_index, y_onehot만 반환합니다.
    """
    # 1. 경로 설정
    if data_type == 'del': dir_name = 'delta'
    elif data_type == 'pkl': dir_name = 'pickle'
    elif data_type == 'cls_pkl': dir_name = 'cls_pickle'
    else: dir_name = data_type 
    data_path = f"data/{dir_name}"
    if not os.path.exists(data_path): data_path = f"data/{data_type}"
    print(f"📂 Loading data from: {data_path}")

    # 2. 데이터 로드 (순서 보장)
    raw_data = {}
    # ⚡ 리눅스 정렬 문제 해결을 위해 sorted 필수
    files = sorted(glob.glob(os.path.join(data_path, "*.pkl")))
    
    for path in files:
        filename = os.path.splitext(os.path.basename(path))[0]
        try: gas = filename.split("_")[0] 
        except: continue
        if gas not in ["acetone", "benzene", "toluene"]: continue
        try:
            df_raw = pd.read_pickle(path)
            x_raw = df_raw.to_numpy() if hasattr(df_raw, "to_numpy") else np.array(df_raw)
            if x_raw.shape[0] > x_raw.shape[1]: x_raw = x_raw.T
            raw_data[gas] = x_raw
        except Exception as e: print(f"❌ Error loading {filename}: {e}")

    # 3. 데이터 병합 (PPM 로드 X)
    X_list, y_index_list, y_onehot_list = [], [], []
    target_gases = ["acetone", "benzene", "toluene"]
    
    for idx, gas in enumerate(target_gases):
        if gas not in raw_data: continue
        x_val = raw_data[gas]
        
        X_list.append(x_val)
        count = len(x_val)
        y_index_list.append(np.full(count, idx))
        onehot = np.zeros((count, 3))
        onehot[:, idx] = 1
        y_onehot_list.append(onehot)
        print(f"   🔹 [{gas}] Loaded: {count} samples")

    if not X_list: raise ValueError("❌ 로드된 데이터가 없습니다.")
        
    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y_index = np.concatenate(y_index_list, axis=0).astype(np.int64)
    y_onehot = np.concatenate(y_onehot_list, axis=0).astype(np.float32)
    
    print(f"✅ Total Data Loaded: {len(X)} samples")
    # ⚡ 3개만 반환 (로컬과 동일)
    return X, y_index, y_onehot



def get_next_run_dir(base_name="runs", parent_dir="./checkpoint"):
    """
    [DDP 동기화 버전]
    Rank 0가 폴더를 결정하고 생성하면, 나머지 Rank는 그 경로를 공유받습니다.
    """
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    
    # 현재 프로세스의 Rank 확인 (DDP 환경변수)
    # 환경변수가 없으면 0(단일 GPU)으로 간주
    rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # 경로 공유를 위한 임시 파일
    sync_file = os.path.join(parent_dir, ".temp_run_config")

   
    if rank == 0:
        # 기존 임시 파일이 있다면 삭제 
        if os.path.exists(sync_file):
            try:
                os.remove(sync_file)
            except:
                pass

        # 1. 'checkpoint/runs' 확인
        target_dir = os.path.join(parent_dir, base_name)
        
        # 2. runs1, runs2... 찾기
        i = 1
        while os.path.exists(target_dir):
            target_dir = os.path.join(parent_dir, f"{base_name}{i}")
            i += 1
        
        # 3. 폴더 생성
        os.makedirs(target_dir, exist_ok=True)
        
        # 4. 결정된 경로를 임시 파일에 기록 
        with open(sync_file, "w") as f:
            f.write(target_dir)
            
        return target_dir


    else:
        
        timeout = 10
        start_time = time.time()
        
        while not os.path.exists(sync_file):
            time.sleep(0.5) # 0.5초씩 대기
            if time.time() - start_time > timeout:

                return os.path.join(parent_dir, base_name)
        
        # 파일이 생기면 읽어서 경로 확인
        try:
            with open(sync_file, "r") as f:
                target_dir = f.read().strip()
            return target_dir
        except:
            # 읽기 실패 시 예외 처리
            return os.path.join(parent_dir, base_name)