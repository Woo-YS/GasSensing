import os
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import wandb
from datetime import datetime

# Sklearn
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

# PyTorch & Lightning
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# Custom Modules
from src.dataset import GasDataModule
from src.utils import SEED, build_samples
from src.lightning_reg import GasRegModel
from src.lightning_cls import GasClsModel

warnings.filterwarnings("ignore")
L.seed_everything(SEED)

# ==============================================================================
# [1] 회귀 (Regression) 실행 로직
# ==============================================================================
def run_regression(args):
    # Regression용 데이터 로드 함수 (내부 정의)
    def load_data(target_gas, data_dir):
        print(f"📂 [{target_gas}] 데이터 로드 시작...")
        label_path = os.path.join(data_dir, "ppm_label_renew", f"{target_gas}_label_ppm.csv")
        pkl_path = os.path.join(data_dir, "pickle", f"{target_gas}_merge.pkl")
        
        if not os.path.exists(pkl_path) or not os.path.exists(label_path):
            print("❌ 데이터 파일이 없습니다.")
            return None, None
            
        y_data = np.loadtxt(label_path, delimiter=',', skiprows=1)
        y = y_data if y_data.ndim == 1 else y_data[:, 0]
        
        df_raw = pd.read_pickle(pkl_path)
        data_numpy = df_raw.to_numpy() if hasattr(df_raw, "to_numpy") else np.array(df_raw)
        x = data_numpy[:, 1:].T.astype(np.float32)

        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]
        mask = y > 0
        return x[mask], y[mask]

    X, y = load_data(args.target_gas, "./data")
    if X is None: return

    y_max_val = np.max(y)
    y_scaled = y / y_max_val
    print(f"⚖️ Max PPM: {y_max_val} (Scaled 0~1)")

    # Hold-out Split
    X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
        X, y_scaled, test_size=0.2, random_state=SEED, shuffle=True
    )
    final_test_dataset = list(zip(X_test_final, y_test_final))

    # K-Fold (Regression은 일반 KFold)
    kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
    group_name = f"{args.target_gas}_{args.model_name}_REG_KFold"

    best_fold_score = float('inf')
    best_fold_idx = 0

    print(f"\n🚀 [Regression] K-Fold 시작 (Gas: {args.target_gas})")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_full, y_train_full)):
        print(f"\n🔄 [Fold {fold+1}/5] ...")
        
        X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
        y_tr, y_val = y_train_full[train_idx], y_train_full[val_idx]

        # Regression은 task='reg' 명시 (list(zip) 전달)
        dm = GasDataModule(
            train_data=list(zip(X_tr, y_tr)), 
            val_data=list(zip(X_val, y_val)), 
            batch_size=args.batch, 
            task='reg'
        )
        model = GasRegModel(args.model_name, input_length=X.shape[1], max_ppm=y_max_val)

        logger = WandbLogger(project="Gas-Integrated", name=f"{args.target_gas}_Reg_Fold{fold+1}", group=group_name, reinit=True)
        ckpt = ModelCheckpoint(monitor='val_loss', mode='min', dirpath=args.save, filename=f'{args.model_name}-Fold{fold+1}-{{val_loss:.4f}}', save_top_k=1)
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=15)

        trainer = L.Trainer(accelerator=args.device, devices=1, max_epochs=args.epoch, logger=logger, callbacks=[ckpt, early_stopping])
        trainer.fit(model, dm)
        
        score = trainer.validate(model, dm)[0]['val_loss']
        if score < best_fold_score:
            best_fold_score = score
            best_fold_idx = fold + 1
        
        wandb.finish()

    # Final Test
    print(f"\n🏆 Best Fold: {best_fold_idx} (Loss: {best_fold_score:.5f})")
    best_ckpt = glob.glob(os.path.join(args.save, f"{args.model_name}-Fold{best_fold_idx}-*.ckpt"))[0]
    
    best_model = GasRegModel.load_from_checkpoint(
        best_ckpt, model_name=args.model_name, input_length=X.shape[1], max_ppm=y_max_val, 
        weights_only=False 
    )
    
    test_logger = WandbLogger(project="Gas-Integrated", name=f"{args.target_gas}_Reg_Test", group=group_name, job_type="test", reinit=True)
    
    # Test DataModule (task='reg')
    test_dm = GasDataModule(test_data=final_test_dataset, batch_size=args.batch, task='reg')
    L.Trainer(accelerator=args.device, devices=1, logger=test_logger).test(best_model, test_dm)
    wandb.finish()


# ==============================================================================
# [2] 분류 (Classification) 실행 로직
# ==============================================================================
def run_classification(args):
    print(f"\n🚀 [Classification] 데이터 로드 및 전처리 (Type: {args.data_type})")
    
    # Classification 데이터 로드 (src.utils.build_samples 사용)
    X, y_index, y_onehot = build_samples(args.data_type)

    # Stratified Split
    X_train, X_test, y_idx_train, y_idx_test, y_hot_train, y_hot_test = train_test_split(
        X, y_index, y_onehot, test_size=0.2, random_state=SEED, stratify=y_index
    )

    group_name = f"{args.model_name}_CLS_KFold_{datetime.now().strftime('%m%d_%H%M')}"
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    best_loss = float('inf')
    best_ckpt_path = None
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_idx_train)):
        print(f"\n🔄 [Fold {fold+1}/5] ...")
        
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_hot_train[tr_idx], y_hot_train[val_idx]

        # Classification은 task='cls' 명시 (Tuple 전달 -> 내부에서 zip)
        dm = GasDataModule(
            train_data=(X_tr, y_tr), 
            val_data=(X_val, y_val), 
            batch_size=args.batch, 
            task='cls'
        )
        
        model = GasClsModel(args.model_name, input_length=X.shape[1], num_classes=3)
        
        logger_name = f"Cls_{args.model_name}_Fold{fold+1}" if args.data_type == 'pkl' else f"Cls_filtered_{args.model_name}_Fold{fold+1}"
        logger = WandbLogger(project="Gas-Integrated", name=logger_name, group=group_name, reinit=True)
        
        ckpt = ModelCheckpoint(monitor='val_loss', mode='min', dirpath=args.save, filename=f'CLS-{args.model_name}-Fold{fold+1}-{{val_loss:.4f}}', save_top_k=1)
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=15)
        
        trainer = L.Trainer(accelerator=args.device, devices=1, max_epochs=args.epoch, logger=logger, callbacks=[ckpt, early_stopping])
        trainer.fit(model, dm)
        
        score = ckpt.best_model_score.item()
        if score < best_loss:
            best_loss = score
            best_ckpt_path = ckpt.best_model_path
        
        wandb.finish()

    # Final Test
    print(f"\n🏆 Best Val Loss: {best_loss:.5f}")
    
    best_model = GasClsModel.load_from_checkpoint(
        best_ckpt_path, model_name=args.model_name, input_length=X.shape[1], num_classes=3,
        weights_only=False
    )
    
    test_logger = WandbLogger(project="Gas-Integrated", name=f"Cls_{args.model_name}_Test", group=group_name, reinit=True)
    
    # Test DataModule (task='cls')
    test_dm = GasDataModule(test_data=(X_test, y_hot_test), batch_size=args.batch, task='cls')
    L.Trainer(accelerator=args.device, devices=1, logger=test_logger).test(best_model, test_dm)
    wandb.finish()
    
    
# ==============================================================================
# [3] 통합 파이프라인 (Pipeline: CLS -> REG) 실행 로직
# ==============================================================================
def run_pipeline(args):
    print(f"\n🚀 [Pipeline] 통합 추론 시작 (Classification -> Regression)")

    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    print(f"⚙️ Device set to: {device}")
    
    # ---------------------------------------------------------
    # 1. 모델 로드
    # ---------------------------------------------------------
    print("📥 모델 로드 중...")
    
    if not args.ckpt_cls:
        raise ValueError("❌ 분류 모델 체크포인트(--ckpt_cls)가 필요합니다.")
    
    # [분류 모델]
    cls_model = GasClsModel.load_from_checkpoint(
        args.ckpt_cls, model_name=args.model_name, input_length=7300, num_classes=3, weights_only=False
    ).to(device)
    cls_model.eval()
    
    # [회귀 모델]
    reg_models = {}
    target_gases = ["acetone", "benzene", "toluene"] 
    
    for i, gas in enumerate(target_gases):
        ckpt_path = os.path.join(args.ckpt_dir, f"Best_{gas}.ckpt") 
        if not os.path.exists(ckpt_path):
             print(f"⚠️ 경고: {gas} 회귀 모델 파일이 없습니다 ({ckpt_path}).")
             continue
        
        # ⚡ [수정 1] max_ppm=1.0 인자 제거! 
        # (체크포인트에 저장된 원래 max_ppm 값을 쓰도록 함)
        reg_model = GasRegModel.load_from_checkpoint(
            ckpt_path, model_name=args.model_name, input_length=7300, weights_only=False 
        ).to(device)
        
        reg_model.eval()
        reg_models[i] = reg_model
        
        # 디버깅: 로드된 max_ppm 확인
        print(f"   🔹 [{gas}] Model Loaded | Max PPM: {reg_model.hparams.max_ppm}")
        
    print("✅ 모든 모델 로드 완료!")

    # ---------------------------------------------------------
    # 2. 데이터 및 PPM 라벨 로드 (동기화)
    # ---------------------------------------------------------
    print("📂 데이터 및 PPM 라벨 로드 중...")
    
    # (1) 데이터 로드 (X, y_index) - build_samples는 보통 0ppm이 제거된 데이터를 반환한다고 가정
    X, y_index, _ = build_samples(args.data_type)
    
    # (2) PPM 라벨 로드
    y_ppm_list = []
    for gas in target_gases:
        label_path = f"./data/ppm_label_renew/{gas}_label_ppm.csv"
        try:
            ppm_data = np.loadtxt(label_path, delimiter=',', skiprows=1)
            ppm = ppm_data if ppm_data.ndim == 1 else ppm_data[:, 0]
            
            # ⚡ [수정 2] 0ppm 제거 필터 적용! (X 데이터와 줄 맞추기)
            # reg_main.py와 동일하게 0보다 큰 값만 남깁니다.
            ppm = ppm[ppm > 0]
            
            y_ppm_list.append(ppm)
        except Exception as e:
            print(f"❌ PPM 로드 실패 ({gas}): {e}")
            return

    # 전체 PPM 합치기
    y_ppm_all = np.concatenate(y_ppm_list)
    
    # 🚨 데이터 개수 검증 (매우 중요)
    if len(X) != len(y_ppm_all):
        print(f"\n⚠️ [CRITICAL WARNING] 데이터 개수 불일치 발생!")
        print(f"   - X (Input) count: {len(X)}")
        print(f"   - y (PPM) count  : {len(y_ppm_all)}")
        print("   -> 데이터 정렬이 어긋나서 회귀 성능이 망가질 수 있습니다.")
        print("   -> utils.py의 build_samples가 데이터를 어떻게 자르는지 확인이 필요합니다.")
        # 강제로 길이를 맞춰서 진행 (임시 조치)
        min_len = min(len(X), len(y_ppm_all))
        X = X[:min_len]
        y_index = y_index[:min_len]
        y_ppm_all = y_ppm_all[:min_len]
    
    # (3) Split (똑같은 시드로 섞어서 Test셋 분리)
    _, X_test, _, y_idx_test, _, y_ppm_test = train_test_split(
        X, y_index, y_ppm_all, test_size=0.2, random_state=SEED, stratify=y_index
    )
    
    # DataModule
    dm = GasDataModule(test_data=(X_test, y_idx_test), batch_size=args.batch, task='cls')
    dm.setup(stage='test')
    dataloader = dm.test_dataloader()

    # ---------------------------------------------------------
    # 3. 추론 루프
    # ---------------------------------------------------------
    total = 0
    cls_correct = 0
    
    true_ppms = []
    pred_ppms = []
    
    print("\n🔍 추론 진행 중...")
    
    # 배치 처리를 위한 인덱스 트래커
    current_idx = 0 

    with torch.no_grad():
        for batch in dataloader:
            x, y_cls_true = batch 
            x = x.to(device)
            y_cls_true = y_cls_true.to(device)
            
            # [1] 분류
            cls_logits = cls_model(x)
            cls_preds = torch.argmax(cls_logits, dim=1) 
            
            batch_size = x.size(0)
            
            # 현재 배치의 정답 PPM 가져오기
            batch_true_ppms = y_ppm_test[current_idx : current_idx + batch_size]
            current_idx += batch_size

            for k in range(batch_size):
                total += 1
                curr_x = x[k].unsqueeze(0)
                pred_idx = cls_preds[k].item()
                true_idx = int(torch.argmax(y_cls_true[k]).item() if y_cls_true.dim() > 1 else y_cls_true[k].item())
                
                # 인덱스 범위 보호
                if k < len(batch_true_ppms):
                    true_ppm_val = batch_true_ppms[k]
                else:
                    break 

                # 분류 정확도 체크
                if pred_idx == true_idx:
                    cls_correct += 1
                    
                    # [2] 회귀 (분류 성공 시)
                    if pred_idx in reg_models:
                        reg_model = reg_models[pred_idx]
                        
                        # 예측 (0~1) -> 복원 (x Max_PPM)
                        raw_pred = reg_model(curr_x).item()
                        
                        # 저장된 max_ppm 사용
                        max_p = reg_model.hparams.max_ppm
                        pred_ppm_val = raw_pred * max_p
                        
                        true_ppms.append(true_ppm_val)
                        pred_ppms.append(pred_ppm_val)

    # ---------------------------------------------------------
    # 4. 결과 리포트
    # ---------------------------------------------------------
    acc = cls_correct / total * 100
    
    true_ppms = np.array(true_ppms)
    pred_ppms = np.array(pred_ppms)
    
    epsilon = 1e-7
    mape = np.mean(np.abs((true_ppms - pred_ppms) / (true_ppms + epsilon))) * 100
    mspe = np.mean(((true_ppms - pred_ppms) / (true_ppms + epsilon)) ** 2) * 100
    rmse = np.sqrt(np.mean((true_ppms - pred_ppms) ** 2))

    print(f"\n📊 [Final Pipeline Result]")
    print(f"   - Total Samples: {total}")
    print(f"   - Classification Accuracy: {acc:.2f}%")
    print("-" * 40)
    print(f"   - 📉 Regression Metrics (for correctly classified samples):")
    print(f"     • RMSE : {rmse:.4f} ppm")
    print(f"     • MAPE : {mape:.4f} %")
    print(f"     • MSPE : {mspe:.4f} %")
    print("-" * 40)


# ==============================================================================
# Main Entry Point
# ==============================================================================
def main(args):
    if not os.path.exists(args.save): os.makedirs(args.save)
    
    if args.task == 'reg':
        run_regression(args)
    elif args.task == 'cls':
        run_classification(args)
    elif args.task == 'predict': 
        run_pipeline(args)
    else:
        print("❌ 잘못된 Task입니다. 'reg' 또는 'cls'를 선택하세요.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # [공통 인자]
    parser.add_argument('--task', type=str, required=True, choices=['reg', 'cls','predict'], help='reg: 회귀, cls: 분류')
    parser.add_argument('-mn', '--model_name', type=str, default='cnn1d')
    parser.add_argument('-b', '--batch_size', dest='batch', type=int, default=128)
    parser.add_argument('-e', '--epoch', type=int, default=300)
    parser.add_argument('-dv', '--device', type=str, default='gpu')
    parser.add_argument('-s', '--save_path', dest='save', type=str, default='./checkpoint/')

    # [Regression 전용 인자]
    parser.add_argument('-tg', '--target_gas', type=str, default='benzene', help='회귀 분석 대상 가스')

    # [Classification 전용 인자]
    parser.add_argument('-dt', '--data_type', type=str, default='del', help='pkl 또는 del (분류 데이터 로드 방식)')
    
    parser.add_argument('--ckpt_cls', type=str, help='분류 모델 체크포인트 경로 (.ckpt)')
    
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoint', help='회귀 모델들이 있는 폴더 경로')
    
    args = parser.parse_args()
    main(args)