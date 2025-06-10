# 필수 패키지 설치: 터미널에서 실행
# pip install pandas numpy plotly scikit-learn scikit-learn-extra umap-learn finance-datareader python-dateutil tqdm kneed requests

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids
import umap.umap_ as umap # umap-learn의 UMAP 모듈 임포트 방식
from sklearn.metrics import silhouette_score
from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging
from tqdm import tqdm
import warnings
import sys
from kneed import KneeLocator
import requests # requests는 직접 사용되지 않지만, FinanceDataReader 내부 의존성일 수 있음
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import OneHotEncoder # 현재 코드에서 직접 사용되지 않음

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# FinanceDataReader를 전역 변수로 선언 (check_and_install_packages에서 초기화)
fdr = None

def check_and_install_packages() -> bool:
    """필요한 패키지가 설치되어 있는지 확인하고 fdr 모듈을 전역으로 설정"""
    required_packages = [
        'pandas', 'numpy', 'plotly', 'scikit-learn',
        'finance-datareader', 'scikit-learn-extra',
        'umap-learn', 'python-dateutil', 'tqdm', 'kneed', 'requests'
    ]
    missing_packages = []
    for package_name in required_packages:
        try:
            if package_name == 'pandas': import pandas
            elif package_name == 'numpy': import numpy
            elif package_name == 'plotly': import plotly
            elif package_name == 'scikit-learn': import sklearn
            elif package_name == 'finance-datareader': import FinanceDataReader
            elif package_name == 'scikit-learn-extra': from sklearn_extra.cluster import KMedoids
            elif package_name == 'umap-learn': import umap.umap_
            elif package_name == 'python-dateutil': import dateutil
            elif package_name == 'tqdm': import tqdm
            elif package_name == 'kneed': import kneed
            elif package_name == 'requests': import requests
        except ImportError:
            missing_packages.append(package_name)

    if missing_packages:
        print("-" * 50)
        print("오류: 다음 필수 패키지가 설치되어 있지 않습니다.")
        for pkg in missing_packages:
            install_command = f"pip install {pkg}"
            # 특정 패키지의 설치명이 다른 경우 처리
            if pkg == 'umap-learn': install_command = "pip install umap-learn"
            elif pkg == 'scikit-learn-extra': install_command = "pip install scikit-learn-extra"
            print(f"- {pkg} (설치 명령어: {install_command})")
        print("터미널에서 위 명령어를 실행하여 패키지를 설치해주세요.")
        print("-" * 50)
        return False

    global fdr
    try:
        import FinanceDataReader as fdr_module
        fdr = fdr_module
        print("모든 필수 패키지가 설치되어 있습니다.")
        return True
    except ImportError:
        print("Critical Error: FinanceDataReader를 임포트할 수 없습니다. 설치를 다시 확인해주세요.")
        # 이 경우 fdr이 None으로 남아 이후 코드 실행 시 오류 발생 가능성 있음
        return False


def collect_user_input() -> dict:
    """사용자의 투자 성향과 선호 시장을 설문으로 수집"""
    print("-" * 50)
    print("ETF 추천 설문")
    print("-" * 50)
    questions = {
        "risk_tolerance":    "1) 리스크 감수 수준 (1: 매우 낮음, 2: 낮음, 3: 중간, 4: 높음, 5: 매우 높음): ",
        "investment_horizon":"2) 투자 예상 기간 (1: 1년 미만, 2: 1-3년, 3: 3-5년, 4: 5-10년, 5: 10년 초과): ",
        "goal":              "3) 주요 투자 목표 (1: 원금 보존, 2: 안정적 수익, 3: 시장 평균, 4: 자산 증식, 5: 고수익): ",
        "market_preference": "4) 선호 투자 시장 (1: 한국(KR), 2: 미국(US), 3: 상관없음): ",
        "experience":        "5) 투자 경험 (1: 초보, 2: 중급, 3: 고급): ",
        "loss_aversion":     "6) 손실 회피 성향 (1: 매우 낮음, 2: 낮음, 3: 중간, 4: 높음, 5: 매우 높음): ", # 질문 수정: 5가 매우 높음
        "theme_preference":  "7) 특정 테마 선호도 (1: 없음, 2: 기술, 3: 에너지, 4: 헬스케어, 5: 기타): "
    }
    user_profile = {}
    for key, question in questions.items():
        while True:
            try:
                ans_str = input(question)
                if not ans_str: # 빈 입력 처리
                    print("오류: 값을 입력해주세요.")
                    continue
                ans = int(ans_str)
                
                valid_options = []
                if key == 'market_preference':
                    valid_options = [1, 2, 3]
                elif key in ['risk_tolerance', 'investment_horizon', 'goal', 'experience', 'loss_aversion', 'theme_preference']:
                    valid_options = [1, 2, 3, 4, 5]
                
                if ans in valid_options:
                    user_profile[key] = ans
                    break
                else:
                    print(f"오류: {', '.join(map(str, valid_options))} 중 하나를 입력해주세요.")
            except ValueError:
                print("오류: 숫자를 입력해주세요.")
    print("-" * 50)
    return user_profile

def fetch_risk_free_rate(start_date_str: str, end_date_str: str) -> float:
    """
    무위험 수익률 조회 (KOFR, CD91, TB3MS 순으로 시도)
    FinanceDataReader 최신 버전에 맞춰 티커 사용
    """
    if fdr is None:
        logging.error("FinanceDataReader가 초기화되지 않았습니다. 무위험 수익률 조회 불가.")
        return 0.03 # 기본값 반환

    # 1. KOFR (한국 무위험지표금리)
    try:
        kofr_data = fdr.DataReader('KOFR', start_date_str, end_date_str)
        if not kofr_data.empty and 'KOFR' in kofr_data.columns:
            avg_rate = kofr_data['KOFR'].mean()
            if pd.notna(avg_rate): # NaN이 아닌 경우
                logging.info(f"KOFR 평균 사용: {avg_rate/100:.4f}")
                return avg_rate / 100
    except Exception as e:
        logging.warning(f"KOFR 수익률 조회 실패: {e}")

    # 2. CD91 (91일물 CD 금리)
    try:
        cd91_data = fdr.DataReader('CD91', start_date_str, end_date_str) # 또는 'CD (91일)' 등 확인 필요
        if not cd91_data.empty:
            # CD 금리는 보통 'Close' 또는 티커명 컬럼에 저장됨
            col_name = 'Close' if 'Close' in cd91_data.columns else cd91_data.columns[0]
            avg_rate = cd91_data[col_name].mean()
            if pd.notna(avg_rate):
                logging.info(f"CD91 ({col_name} 컬럼) 평균 사용: {avg_rate/100:.4f}")
                return avg_rate / 100
    except Exception as e:
        logging.warning(f"CD91 수익률 조회 실패: {e}")

    # 3. TB3MS (미국 3개월 재무부 채권) - FRED 통해 조회 시도
    try:
        # FinanceDataReader는 FRED 데이터를 가져올 때 'FRED:' 접두사를 사용합니다.
        tbill = fdr.DataReader('FRED:TB3MS', start_date_str, end_date_str)
        if not tbill.empty and 'TB3MS' in tbill.columns:
            avg_rate = tbill['TB3MS'].mean()
            if pd.notna(avg_rate):
                logging.info(f"TB3MS (FRED) 평균 사용: {avg_rate/100:.4f}")
                return avg_rate / 100
    except Exception as e:
        logging.warning(f"TB3MS (FRED) 수익률 조회 실패: {e}")

    logging.warning("모든 무위험 수익률 조회 실패. 기본값 0.03 (3%) 사용.")
    return 0.03

def calculate_risk_metrics(returns: pd.DataFrame, risk_free_rate: float = 0.0) -> pd.DataFrame:
    """종합 리스크 메트릭 계산"""
    metrics = pd.DataFrame(index=returns.columns)
    annual_factor = 252 # 주식 시장 개장일 기준

    metrics['Annual Return'] = returns.mean() * annual_factor
    metrics['Annual Volatility'] = returns.std() * np.sqrt(annual_factor)

    # 분모가 0 또는 매우 작은 값일 경우 Sharpe Ratio 등이 무한대가 되는 것 방지
    metrics['Sharpe Ratio'] = np.where(
        np.abs(metrics['Annual Volatility']) > 1e-6,
        (metrics['Annual Return'] - risk_free_rate) / metrics['Annual Volatility'],
        0
    )

    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    metrics['Max Drawdown'] = drawdown.min()

    # 음의 수익률만 사용하여 Downside Risk 계산
    downside_returns = returns[returns < 0].fillna(0) # NaN 발생 방지
    metrics['Downside Risk'] = downside_returns.std() * np.sqrt(annual_factor)
    metrics['Sortino Ratio'] = np.where(
        np.abs(metrics['Downside Risk']) > 1e-6,
        (metrics['Annual Return'] - risk_free_rate) / metrics['Downside Risk'],
        0
    )
    
    metrics['Calmar Ratio'] = np.where(
        np.abs(metrics['Max Drawdown']) > 1e-6, # Max Drawdown은 음수이므로 절대값으로 비교
        metrics['Annual Return'] / (-metrics['Max Drawdown']),
        0
    )

    metrics['Skewness'] = returns.skew()
    metrics['Kurtosis'] = returns.kurt() # Excess Kurtosis (정규분포 Kurtosis = 3)

    # 데이터 기간이 1년(252일) 이상일 경우에만 롤링 지표 계산
    if len(returns) >= annual_factor:
        metrics['Rolling Return'] = returns.rolling(window=annual_factor).mean().iloc[-1] * annual_factor
        metrics['Rolling Volatility'] = returns.rolling(window=annual_factor).std().iloc[-1] * np.sqrt(annual_factor)
        metrics['Recent Return'] = returns.iloc[-annual_factor:].mean() * annual_factor
        metrics['Recent Volatility'] = returns.iloc[-annual_factor:].std() * np.sqrt(annual_factor)
    else:
        # 데이터 부족 시 NaN으로 채우고, 나중에 fillna(0) 처리
        metrics['Rolling Return'] = np.nan
        metrics['Rolling Volatility'] = np.nan
        metrics['Recent Return'] = np.nan
        metrics['Recent Volatility'] = np.nan
        logging.info("데이터 기간이 1년 미만으로 Rolling/Recent 지표는 Annual 지표로 대체될 수 있습니다.")
        # 이 경우, fillna(0) 대신 Annual 값으로 채우는 것도 고려할 수 있으나, 현재는 0으로 채움

    metrics = metrics.fillna(0).replace([np.inf, -np.inf], 0) # 모든 NaN과 inf 값을 0으로 처리
    return metrics


def fetch_etf_data_with_retry(tickers: list, start: str, end: str, max_retries: int = 3):
    """
    개선된 ETF 데이터 수집 함수
    - 한국 ETF(6자리 숫자 코드): exchange='KRX'
    - 그 외 티커: 기본 데이터소스
    """
    if fdr is None:
        logging.error("FinanceDataReader가 초기화되지 않았습니다. ETF 데이터 수집 불가.")
        return pd.DataFrame(), []
        
    data = pd.DataFrame()
    successful_tickers = []
    failed_tickers = []

    for tk in tqdm(tickers, desc="ETF 데이터 수집"):
        is_kr = tk.isdigit() and len(tk) == 6  # 한국 ETF 여부 판단

        for attempt in range(1, max_retries + 1):
            try:
                df_raw = None
                if is_kr:
                    df_raw = fdr.DataReader(tk, start, end, exchange='KRX')
                else:
                    df_raw = fdr.DataReader(tk, start, end)

                if df_raw is None or df_raw.empty:
                    logging.warning(f"[{tk}] 데이터 없음 (시도 {attempt}/{max_retries})")
                    if attempt == max_retries: failed_tickers.append(tk)
                    continue

                close_col = 'Close' if 'Close' in df_raw.columns else ('Adj Close' if 'Adj Close' in df_raw.columns else None)
                if close_col is None:
                    logging.warning(f"[{tk}] Close 또는 Adj Close 컬럼 없음 (시도 {attempt}/{max_retries})")
                    if attempt == max_retries: failed_tickers.append(tk)
                    continue
                
                series = df_raw[close_col].copy() # SettingWithCopyWarning 방지
                series.replace([np.inf, -np.inf], np.nan, inplace=True) # 무한대 값 NaN으로 변경
                
                # 보간 전에 NaN이 아닌 값이 충분히 있는지 확인
                if series.notna().sum() < 2: # 최소 2개의 유효한 데이터 포인트가 있어야 보간 가능
                    logging.warning(f"[{tk}] 유효한 데이터 포인트 부족 (<2) (시도 {attempt}/{max_retries})")
                    if attempt == max_retries: failed_tickers.append(tk)
                    continue

                series.interpolate(method='linear', limit_direction='both', inplace=True) # 양방향 선형 보간
                series.ffill(inplace=True)
                series.bfill(inplace=True)

                if series.isnull().all(): # 보간 후에도 모든 값이 NaN이면 스킵
                    logging.warning(f"[{tk}] 보간 후에도 유효한 종가 데이터 없음 (시도 {attempt}/{max_retries})")
                    if attempt == max_retries: failed_tickers.append(tk)
                    continue
                
                # 중복 인덱스 제거 (첫 번째 값 유지)
                series = series[~series.index.duplicated(keep='first')]
                data[tk] = series
                successful_tickers.append(tk)
                break  # 성공 시 재시도 루프 탈출

            except Exception as e:
                logging.warning(f"[{tk}] 수집 중 오류 (시도 {attempt}/{max_retries}): {e}")
                if attempt == max_retries:
                    failed_tickers.append(tk)

    if data.empty:
        logging.error("수집된 ETF 데이터가 전혀 없습니다. 티커 리스트와 기간을 확인해주세요.")
    else:
        data = data.dropna(how='all', axis=1) # 모든 값이 NaN인 열 제거
        logging.info(f"ETF 데이터 수집 완료: 성공 {len(successful_tickers)}개, 실패 {len(failed_tickers)}개")
        if failed_tickers:
            logging.warning(f"수집 실패 티커: {', '.join(failed_tickers)}")
            
    return data, successful_tickers


def optimize_clustering(data: pd.DataFrame, k_range=range(2, 11), random_state=42):
    """클러스터링 최적화 (UMAP 차원 축소 후 다양한 알고리즘 시도)"""
    if data.empty or len(data) < max(k_range): # 데이터가 K 최대값보다 작으면 일부 알고리즘 문제 발생
        logging.error(f"클러스터링 입력 데이터 부족 (데이터 수: {len(data)}, K 최대값: {max(k_range)}) 또는 데이터 없음.")
        # 데이터가 있어도 UMAP 결과가 비어있을 수 있으므로, 반환 타입을 일관되게 유지
        # (차원 축소된 데이터, 레이블 배열)
        # 빈 UMAP 결과는 (n, 0) 형태일 수 있고, 레이블은 (n,) 형태여야 함.
        # 이 경우, UMAP 결과는 (len(data), 0) 또는 (len(data), data.shape[1]) 등으로 하고, 레이블은 모두 0으로.
        # 더 나은 방법은 None, None 반환 후 호출부에서 처리.
        # 현재는 (빈 UMAP 데이터, 빈 레이블)로 처리 중인데, 레이블은 0으로 채워진 것을 기대함.
        return np.array([]).reshape(0, 3), np.zeros(len(data) if not data.empty else 0, dtype=int)


    scaler = RobustScaler()
    try:
        # 데이터에 NaN이나 Inf가 있으면 스케일링 에러 발생 가능
        scaled_data = scaler.fit_transform(data.replace([np.inf, -np.inf], np.nan).fillna(0))
    except ValueError as e:
        logging.error(f"데이터 스케일링 중 오류: {e}. 데이터에 NaN/Inf 값 확인 필요.")
        return np.array([]).reshape(0, 3), np.zeros(len(data), dtype=int)

    best_umap_data = None
    # UMAP 적용 전 데이터 샘플 수 확인
    if len(scaled_data) < 2: # UMAP은 최소 2개의 샘플 필요
        logging.warning("UMAP 적용에 필요한 최소 데이터 샘플(2) 부족. UMAP 없이 원본 스케일 데이터를 사용 (최대 3차원).")
        # 차원 축소 없이 진행하거나, PCA 등 다른 방법 사용. 여기서는 원본 데이터의 첫 3개 feature 사용 시도.
        # 또는 n_components를 데이터 feature 수와 같게 하거나 줄임.
        # 현재 로직에서는 UMAP이 필수적이므로, 여기서 클러스터링 실패로 간주하고 기본 레이블 반환.
        # 하지만 아래 로직에서 n_neighbors 조정으로 일부 커버 가능.
        # best_umap_data = scaled_data[:, :min(3, scaled_data.shape[1])] # 원본 데이터 일부 사용
        # 이 경우, UMAP 없이 바로 KMeans로 넘어가는 로직이 필요. 또는 UMAP 실패 처리.
        # 현재는 UMAP 파라미터 조정으로 시도.
        pass # 아래 로직에서 n_neighbors 조정

    if len(scaled_data) >= 2: # UMAP 적용 가능한 최소 샘플
        best_umap_score = -np.inf # 실루엣 스코어는 -1에서 1 사이
        
        # UMAP 파라미터 튜닝 (데이터 크기에 따라 n_neighbors 조정)
        n_neighbors_options = [5, 10, 15]
        min_dist_options = [0.0, 0.1, 0.2] # min_dist는 0.0도 가능

        for n_neighbors_val_raw in n_neighbors_options:
            # n_neighbors는 (샘플 수 - 1)보다 클 수 없음, 최소 1
            n_neighbors_val = min(n_neighbors_val_raw, max(1, len(scaled_data) - 1))
            if n_neighbors_val == 0 and len(scaled_data) == 1 : n_neighbors_val = 1 # 극단적 케이스 방어
            
            for min_dist_val in min_dist_options:
                try:
                    # n_components는 실제 사용될 차원 수, 여기서는 3으로 고정
                    # 데이터 차원보다 n_components가 클 수 없음. 입력 데이터(data)의 feature 수 확인 필요.
                    # scaled_data.shape[1] (원본 feature 수) 와 n_components=3 비교.
                    current_n_components = min(3, scaled_data.shape[1])
                    if current_n_components == 0 : continue # feature가 없으면 UMAP 불가

                    umap_reducer = umap.UMAP(
                        n_components=current_n_components,
                        n_neighbors=n_neighbors_val,
                        min_dist=min_dist_val,
                        random_state=random_state,
                        # low_memory=True # 데이터가 매우 클 경우 고려
                    )
                    umap_data_candidate = umap_reducer.fit_transform(scaled_data)
                    
                    # UMAP 결과로 임시 클러스터링하여 실루엣 점수 평가 (예: K=3 또는 적절한 K)
                    # 이 평가 K는 최종 K와 다를 수 있음. UMAP 표현의 질을 평가하기 위함.
                    # 너무 작은 데이터셋에서는 실루엣 점수 자체가 불안정할 수 있음.
                    if len(umap_data_candidate) >= 2: # 실루엣 점수 계산 최소 샘플
                        # 임시 클러스터링 k는 umap_data_candidate 수보다 작아야 함
                        temp_k = min(3, max(2, len(umap_data_candidate) -1)) 
                        if temp_k < 2 : continue # 클러스터링 불가

                        temp_kmeans = KMeans(n_clusters=temp_k, n_init='auto' if hasattr(KMeans(), 'n_init') and KMeans().n_init == 'auto' else 10, random_state=random_state)
                        temp_labels = temp_kmeans.fit_predict(umap_data_candidate)
                        if len(set(temp_labels)) > 1: # 두 개 이상의 클러스터가 형성된 경우에만 점수 계산
                            score = silhouette_score(umap_data_candidate, temp_labels)
                            if score > best_umap_score:
                                best_umap_score = score
                                best_umap_data = umap_data_candidate
                except Exception as e:
                    logging.warning(f"UMAP 튜닝 중 오류 (n_neighbors={n_neighbors_val}, min_dist={min_dist_val}): {e}")
    
    # UMAP 튜닝에서 좋은 결과를 못 찾았거나, 데이터가 너무 작아 튜닝을 건너뛴 경우 기본 UMAP 실행
    if best_umap_data is None:
        if len(scaled_data) >= 2 and scaled_data.shape[1] > 0 : # 데이터 샘플과 feature가 있어야 함
            # n_neighbors 기본값은 보통 15, min_dist는 0.1
            default_n_neighbors = min(15, max(1, len(scaled_data) - 1))
            default_n_components = min(3, scaled_data.shape[1])
            if default_n_components > 0:
                try:
                    umap_reducer = umap.UMAP(n_components=default_n_components, n_neighbors=default_n_neighbors, min_dist=0.1, random_state=random_state)
                    best_umap_data = umap_reducer.fit_transform(scaled_data)
                except Exception as e:
                    logging.error(f"기본 UMAP 적용 실패: {e}. 클러스터링을 위한 차원 축소 불가.")
                    return scaled_data[:, :default_n_components] if default_n_components > 0 else scaled_data, np.zeros(len(scaled_data), dtype=int) # 원본 데이터 (일부) 또는 빈 레이블
            else: # feature가 없는 경우
                 best_umap_data = scaled_data # UMAP 적용 불가, 원본 사용
        elif scaled_data.shape[1] > 0 : # 데이터는 있지만 샘플이 1개인 경우 등
            best_umap_data = scaled_data[:, :min(3, scaled_data.shape[1])] # UMAP 없이 원본 데이터 사용
        else: # 데이터가 아예 비어있거나 feature가 없는 경우
            logging.error("UMAP 적용 불가: 입력 데이터에 feature가 없습니다.")
            return np.array([]).reshape(0,3), np.zeros(len(data), dtype=int)


    umap_data = best_umap_data
    if umap_data is None or umap_data.shape[0] == 0: # UMAP 결과가 비정상적인 경우
        logging.error("UMAP 결과 데이터가 유효하지 않습니다. 클러스터링 중단.")
        return np.array([]).reshape(0,3), np.zeros(len(data), dtype=int)

    wcss = [] # Within-cluster sum of squares
    # K 값 범위는 umap_data 샘플 수보다 작아야 함
    # valid_k = [k for k in k_range if k <= len(umap_data)] -> K는 샘플 수보다 작아야 함 (k < len(umap_data))
    # K는 최소 2 이상이어야 함
    valid_k_upper_bound = len(umap_data) # K는 샘플 수와 같을 수 없음 (KMeans)
    valid_k_list = [k for k in k_range if k >= 2 and k < valid_k_upper_bound]

    if not valid_k_list:
        logging.warning(f"UMAP 결과 데이터 수({len(umap_data)})가 너무 작아 유효한 K 값 범위가 없습니다. (기존 k_range: {k_range}). 기본 K=1 또는 데이터 수로 클러스터링 시도.")
        # 이 경우, 모든 데이터를 하나의 클러스터로 처리하거나, 각 데이터를 개별 클러스터로 처리
        if len(umap_data) > 0:
            # 데이터가 하나라도 있으면, 모든 데이터에 레이블 0을 할당하거나, 
            # K=len(umap_data)로 하고 각자 다른 레이블을 줄 수 있지만, 의미가 없음.
            # 차라리 단일 클러스터로 처리.
            # 또는 K를 len(umap_data) 미만의 최댓값으로 설정. 예: len(umap_data)가 3이면 K=2
            if len(umap_data) == 1: return umap_data, np.array([0], dtype=int)
            # fallback_k = max(1, len(umap_data) -1) if len(umap_data) > 1 else 1
            # km = KMeans(n_clusters=fallback_k, ...).fit(umap_data)
            # return umap_data, km.labels_
            return umap_data, np.zeros(len(umap_data), dtype=int) # 단일 클러스터
        else:
            return umap_data, np.array([], dtype=int) # UMAP 결과가 비었으면 빈 레이블

    for k_val in valid_k_list:
        try:
            km = KMeans(n_clusters=k_val, n_init='auto' if hasattr(KMeans(), 'n_init') and KMeans().n_init == 'auto' else 10, random_state=random_state)
            km.fit(umap_data)
            wcss.append(km.inertia_)
        except Exception as e:
            logging.warning(f"KMeans 실행 중 오류 (K={k_val}): {e}. 해당 K값은 WCSS 계산에서 제외됩니다.")
            # valid_k_list에서 해당 k_val을 제거해야 KneeLocator 입력과 길이가 맞음
            # 하지만 이미 순회 중이므로, wcss에 해당 값이 안들어가는 것으로 처리됨.
            # KneeLocator에 전달되는 x, y의 길이가 맞는지 확인 필요.
            # 이 경우, wcss에 값이 안 쌓이면 valid_k_list와 wcss 길이가 달라짐.
            # 따라서 wcss에 None이나 np.nan을 넣고 나중에 처리하거나, 해당 k를 valid_k_list에서 빼야 함.
            # 여기서는 일단 그대로 두고, KneeLocator 실행 전 wcss와 valid_k_list 길이 체크

    # WCSS와 valid_k_list의 길이가 다를 수 있으므로, WCSS가 계산된 K만 사용
    processed_k_for_elbow = [k for i, k in enumerate(valid_k_list) if i < len(wcss)] # wcss가 짧으면 그 길이에 맞춤

    best_k_val = 3 # 기본 K 값
    if len(processed_k_for_elbow) >= 2 and len(wcss) >= 2: # KneeLocator는 최소 2개의 포인트 필요
        try:
            # WCSS가 계산된 K값만 KneeLocator에 전달
            kl = KneeLocator(processed_k_for_elbow, wcss, curve='convex', direction='decreasing', S=1.0) # S는 민감도
            if kl.elbow:
                best_k_val = kl.elbow
                logging.info(f"Elbow method 최적 K: {best_k_val}")
            else:
                logging.warning("Elbow point를 찾지 못했습니다. 기본 K=3 또는 다른 방법으로 K 결정 시도.")
        except Exception as e:
            logging.error(f"KneeLocator 오류: {e}. 기본 K=3 사용.")
    else:
        logging.warning(f"WCSS/K 값 부족 ({len(wcss)}/{len(processed_k_for_elbow)})으로 Elbow method 사용 불가. 기본 K=3 사용.")
        if valid_k_list: # 유효한 K가 하나라도 있었다면
            best_k_val = min(best_k_val, valid_k_list[-1]) # K가 너무 크지 않도록 조정
            if len(umap_data) > 0: best_k_val = min(best_k_val, len(umap_data)-1 if len(umap_data)>1 else 1)


    # best_k_val이 데이터 수보다 크거나 같으면 조정
    if best_k_val >= len(umap_data) and len(umap_data) > 0:
        best_k_val = max(1, len(umap_data) - 1 if len(umap_data) > 1 else 1)
    
    if best_k_val == 0 and len(umap_data) > 0: best_k_val = 1 # 최소 1개 클러스터

    best_result = {"score": -np.inf, "k": best_k_val, "labels": None, "name": "N/A"}
    
    # 클러스터링 알고리즘 실행 (KMeans, KMedoids)
    # DBSCAN은 K를 사용하지 않으므로 별도 처리 또는 제외. 여기서는 eps, min_samples 파라미터 튜닝 필요.
    # 현재는 고정 파라미터로 시도.
    algorithms_requiring_k = {
        'KMeans': lambda k_param: KMeans(n_clusters=k_param, n_init='auto' if hasattr(KMeans(), 'n_init') and KMeans().n_init == 'auto' else 10, random_state=random_state),
        'KMedoids': lambda k_param: KMedoids(n_clusters=k_param, random_state=random_state)
    }

    # best_k_val이 1이면 (클러스터 1개), 실루엣 점수 계산 불가. 모든 레이블을 0으로.
    if best_k_val < 2:
        logging.warning(f"최적 K가 {best_k_val}로 단일 클러스터를 의미. 모든 데이터에 레이블 0 할당.")
        best_result.update({"name": "SingleCluster", "labels": np.zeros(len(umap_data), dtype=int), "score": 0, "k": 1})
    else: # K >= 2
        for algo_name, model_factory in algorithms_requiring_k.items():
            try:
                model_instance = model_factory(best_k_val)
                labels_pred = model_instance.fit_predict(umap_data)
                
                # 유효한 클러스터가 2개 이상 형성되었는지, 모든 샘플이 노이즈(-1)로 처리되지 않았는지 확인
                unique_labels = set(labels_pred)
                if len(unique_labels) < 2 or (len(unique_labels) == 1 and -1 in unique_labels):
                    logging.warning(f"{algo_name} (K={best_k_val}): 유효한 클러스터 형성 실패 (레이블: {unique_labels}).")
                    continue
                
                # -1 (노이즈) 레이블이 있는 경우 실루엣 점수 계산 시 제외하거나, DBSCAN에만 해당 처리
                # 여기서는 -1이 없는 경우에만 실루엣 점수 계산 (KMeans, KMedoids는 -1 없음)
                score = silhouette_score(umap_data, labels_pred)
                logging.info(f"{algo_name} (K={best_k_val}), Silhouette Score = {score:.4f}")
                if score > best_result["score"]:
                    best_result.update({"name": algo_name, "labels": labels_pred, "score": score, "k": best_k_val})
            except Exception as e:
                logging.error(f"{algo_name} (K={best_k_val}) 실행 중 오류: {e}")

        # DBSCAN 시도 (eps, min_samples 값은 데이터 스케일에 따라 민감하게 조정 필요)
        # 예시로 UMAP 결과의 표준편차 등을 이용해 eps를 추정할 수 있음. 여기서는 고정값 사용.
        try:
            # DBSCAN은 샘플 수가 min_samples보다 커야 함
            dbscan_min_samples = min(5, max(1, len(umap_data) -1)) # min_samples는 1 이상, 샘플 수보다 작게
            if len(umap_data) >= dbscan_min_samples and dbscan_min_samples > 0 :
                # eps 값은 데이터의 스케일에 따라 매우 중요. UMAP 결과의 표준편차를 이용한 추정 등 고려 가능.
                # 여기서는 간단히 0.5로 고정.
                db_model = DBSCAN(eps=0.5, min_samples=dbscan_min_samples) 
                labels_db = db_model.fit_predict(umap_data)
                unique_labels_db = set(labels_db)
                
                # DBSCAN 결과에서 노이즈(-1)를 제외하고 유효 클러스터가 2개 이상인지 확인
                # 또는 노이즈를 포함하더라도, 노이즈가 아닌 클러스터가 1개 이상인지 등 기준 필요.
                # 여기서는 노이즈 제외 유효 클러스터 1개 이상 + 전체 유효 클러스터 2개 이상.
                valid_clusters_db = [l for l in unique_labels_db if l != -1]
                if len(valid_clusters_db) >= 1 and len(unique_labels_db) >=2 : # 노이즈가 아닌 클러스터가 있고, 전체 레이블 종류 2개 이상
                    # DBSCAN의 경우, 노이즈 포인트(-1)를 제외하고 실루엣 점수 계산
                    if -1 in unique_labels_db:
                        # 노이즈 제외 데이터와 레이블로 점수 계산
                        non_noise_indices = [i for i, label in enumerate(labels_db) if label != -1]
                        if len(non_noise_indices) >= 2 and len(set(labels_db[non_noise_indices])) >=2: # 노이즈 아닌게 2개 이상, 그들만의 클러스터도 2개 이상
                             score_db = silhouette_score(umap_data[non_noise_indices], labels_db[non_noise_indices])
                        else: score_db = -1 # 점수 계산 불가
                    else: # 노이즈가 없는 경우
                        score_db = silhouette_score(umap_data, labels_db)
                    
                    logging.info(f"DBSCAN (eps=0.5, min_samples={dbscan_min_samples}), Silhouette Score = {score_db:.4f}, Unique Labels = {unique_labels_db}")
                    if score_db > best_result["score"]:
                        best_result.update({"name": "DBSCAN", "labels": labels_db, "score": score_db, "k": len(valid_clusters_db)}) # k는 유효 클러스터 수
                else:
                    logging.warning(f"DBSCAN: 유효한 클러스터 형성 실패 (레이블: {unique_labels_db}).")
        except Exception as e:
            logging.error(f"DBSCAN 실행 중 오류: {e}")


    # 어떤 알고리즘도 성공하지 못한 경우, 또는 점수가 매우 낮은 경우 KMeans Fallback
    if best_result["labels"] is None or best_result["score"] < -0.5 : # 점수가 매우 낮거나 결과가 없는 경우
        logging.warning(f"모든 클러스터링 시도에서 유의미한 결과 얻지 못함 (최고 점수: {best_result['score']:.2f}). KMeans Fallback (K=3 또는 조정된 K) 사용.")
        fallback_k = min(3, max(1, len(umap_data) -1 if len(umap_data) > 1 else 1)) # fallback K는 1,2,3 중 데이터에 맞게
        if fallback_k > 0 and len(umap_data) >= fallback_k :
            try:
                km_fallback = KMeans(n_clusters=fallback_k, n_init='auto' if hasattr(KMeans(), 'n_init') and KMeans().n_init == 'auto' else 10, random_state=random_state)
                labels_fallback = km_fallback.fit_predict(umap_data)
                best_result.update({"name": f"KMeans Fallback", "labels": labels_fallback, "k": fallback_k, "score": -1 if fallback_k < 2 else silhouette_score(umap_data, labels_fallback)})
            except Exception as e:
                logging.error(f"KMeans Fallback 실행 오류: {e}. 모든 ETF를 단일 그룹으로 처리.")
                best_result["labels"] = np.zeros(len(umap_data), dtype=int) # 최종 실패 시 단일 클러스터
                best_result["k"] = 1
        else: # fallback K 조차 실행 불가
            best_result["labels"] = np.zeros(len(umap_data), dtype=int)
            best_result["k"] = 1


    logging.info(f"최종 선택된 클러스터링 알고리즘: {best_result['name']}, K={best_result['k']}, Silhouette Score={best_result['score']:.4f}")
    # 반환되는 레이블이 data의 원래 인덱스와 매칭되어야 함. umap_data는 data와 행 순서 동일.
    # 만약 data에서 일부가 누락되어 scaled_data가 되었다면, 여기서 반환되는 labels는 scaled_data 기준.
    # 현재는 data -> scaled_data -> umap_data 모두 행 개수 및 순서가 동일하다고 가정.
    # (실제로는 data에서 dropna 등으로 행이 줄어들 수 있음. 함수 초입 clustering_input에서 이미 처리됨)
    final_labels = best_result["labels"]
    if final_labels is None: # 만약 labels가 None으로 남는 극단적 경우 방지
        final_labels = np.zeros(len(umap_data), dtype=int)

    return umap_data, final_labels


def derive_user_quantitative_indicators(user_profile: dict) -> dict:
    """정량적 사용자 선호 지표 도출"""
    # 리스크 스코어: risk_tolerance와 loss_aversion 조합 (1~5점 척도)
    # loss_aversion: 1(매우낮음) ~ 5(매우높음) -> 높을수록 리스크 회피적. 따라서 (6 - loss_aversion) 사용
    risk_score = (user_profile['risk_tolerance'] + (6 - user_profile['loss_aversion'])) / 2.0
    
    # 기대 수익률 수준: goal에 따라 매핑 (1: 원금보존 ~ 5: 고수익)
    expected_return_map = {1: 0.02, 2: 0.05, 3: 0.08, 4: 0.12, 5: 0.15} # 연 수익률
    expected_return = expected_return_map.get(user_profile['goal'], 0.08) # 기본 중간값
    
    # 시장 선호도 점수: market_preference (1: KR, 2: US, 3: 상관없음)
    market_preference_scores = {
        1: {'KR': 1.0, 'US': 0.0},  # 한국만
        2: {'KR': 0.0, 'US': 1.0},  # 미국만
        3: {'KR': 0.5, 'US': 0.5}   # 상관없음 (균등 선호)
    }
    market_scores = market_preference_scores.get(user_profile['market_preference'], {'KR': 0.5, 'US': 0.5})
    
    # 테마 선호도 (1: 없음, 2: 기술, 3: 에너지, 4: 헬스케어, 5: 기타)
    # 이 정보는 derive_user_quantitative_indicators에서 직접 벡터화하기보다,
    # 추천 단계에서 활용하는 것이 더 직관적일 수 있음. 여기서는 원본 값 유지.
    # theme_vector는 현재 추천 로직에서 직접 사용되지 않으므로, 사용자 프로필의 값 자체를 전달.
    
    return {
        'risk_score': risk_score,               # 1 ~ 5 (높을수록 리스크 선호)
        'expected_return': expected_return,     # 예: 0.02 ~ 0.15
        'market_scores': market_scores,         # {'KR': float, 'US': float}
        'user_theme_preference_code': user_profile['theme_preference'] # 테마 코드 (1~5)
    }

def load_user_etf_preferences(file_path: str = 'user_etf_preferences.xlsx') -> pd.DataFrame:
    """
    가정된 사용자-ETF 선호도 데이터 로드.
    파일에는 'risk_tolerance', 'investment_horizon', 'goal', 'experience', 
    'loss_aversion', 'theme_preference', 'preferred_etfs' (쉼표로 구분된 티커 문자열) 열이 있어야 함.
    """
    try:
        df = pd.read_excel(file_path)
        # 필수 컬럼 확인
        required_cols = ['risk_tolerance', 'investment_horizon', 'goal', 'experience', 
                         'loss_aversion', 'theme_preference', 'preferred_etfs']
        if not all(col in df.columns for col in required_cols):
            logging.error(f"사용자-ETF 선호도 파일({file_path})에 필수 컬럼이 부족합니다. ({required_cols})")
            return pd.DataFrame()
        logging.info(f"사용자-ETF 선호도 데이터 로드 완료: {file_path} ({len(df)} 명의 사용자 데이터)")
        return df
    except FileNotFoundError:
        logging.warning(f"사용자-ETF 선호도 파일({file_path})을 찾을 수 없습니다. 협업 필터링은 생략됩니다.")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"사용자-ETF 선호도 데이터 로드 중 오류: {e}")
        return pd.DataFrame()


def collaborative_filtering_recommendation(user_profile: dict, metrics_df: pd.DataFrame, user_etf_pref_df: pd.DataFrame, top_n_similar_users: int = 5) -> list:
    """협업 필터링을 통한 ETF 추천 (가장 유사한 Top-N 사용자들의 선호 ETF 종합)"""
    if user_etf_pref_df.empty:
        logging.info("사용자-ETF 선호도 데이터가 없어 협업 필터링을 생략합니다.")
        return []

    # 현재 사용자의 프로필 벡터 (설문 항목 기준)
    current_user_vector_values = [
        user_profile['risk_tolerance'],
        user_profile['investment_horizon'],
        user_profile['goal'],
        user_profile['experience'],
        user_profile['loss_aversion'],
        user_profile['theme_preference']
    ]
    current_user_vector = np.array(current_user_vector_values).reshape(1, -1)

    # 다른 사용자들의 프로필 벡터
    # user_etf_pref_df에서 해당 컬럼만 추출하여 numpy 배열로 변환
    other_users_profile_features = ['risk_tolerance', 'investment_horizon', 'goal', 'experience', 'loss_aversion', 'theme_preference']
    other_users_vectors = user_etf_pref_df[other_users_profile_features].values

    # 코사인 유사도 계산
    if other_users_vectors.shape[0] == 0: # 다른 사용자 데이터가 없는 경우
        logging.warning("협업 필터링을 위한 다른 사용자 데이터가 없습니다.")
        return []
        
    similarities = cosine_similarity(current_user_vector, other_users_vectors).flatten()

    # 유사도가 높은 Top-N 사용자 인덱스 찾기
    # 자기 자신과의 유사도(만약 있다면)를 제외하기 위해, 현재 사용자가 user_etf_pref_df에 포함되지 않는다고 가정.
    # 만약 포함된다면, 해당 인덱스 제외 필요.
    # 여기서는 가장 유사한 N명을 사용.
    # argsort는 오름차순 정렬된 인덱스를 반환하므로, [-top_n_similar_users:]로 가장 큰 값 N개 선택 후, ::-1로 내림차순.
    # N명보다 사용자 수가 적을 수 있으므로 min 사용
    num_users_to_consider = min(top_n_similar_users, len(similarities))
    if num_users_to_consider == 0: return []

    similar_user_indices = np.argsort(similarities)[-num_users_to_consider:][::-1]
    
    # 유사한 사용자들이 선호한 ETF들을 집계
    cf_recommended_etfs = {} # ETF 티커: 빈도수 또는 총 유사도 점수
    for idx in similar_user_indices:
        user_similarity_score = similarities[idx]
        # preferred_etfs 컬럼은 쉼표로 구분된 문자열로 가정
        etf_list_str = user_etf_pref_df.iloc[idx]['preferred_etfs']
        if pd.isna(etf_list_str) or not isinstance(etf_list_str, str): continue

        preferred_etfs_for_user = [etf.strip() for etf in etf_list_str.split(',') if etf.strip()]
        for etf_ticker in preferred_etfs_for_user:
            if etf_ticker in metrics_df.index: # 현재 분석 대상 ETF 목록에 있는 것만 추천
                cf_recommended_etfs[etf_ticker] = cf_recommended_etfs.get(etf_ticker, 0) + user_similarity_score # 유사도 가중치 부여

    # 빈도수 또는 누적 유사도 점수가 높은 순으로 정렬된 ETF 리스트 반환
    if not cf_recommended_etfs:
        logging.info("협업 필터링 결과, 추천할 만한 ETF가 없습니다 (유사 사용자 선호 ETF 부재 또는 필터링됨).")
        return []

    sorted_cf_etfs = sorted(cf_recommended_etfs.items(), key=lambda item: item[1], reverse=True)
    final_cf_recs = [etf_item[0] for etf_item in sorted_cf_etfs]
    
    logging.info(f"협업 필터링 추천 (상위 {len(final_cf_recs)}개): {', '.join(final_cf_recs[:5])}{'...' if len(final_cf_recs) > 5 else ''}")
    return final_cf_recs


def match_user_to_cluster(user_quantitative_indicators: dict, metrics_df: pd.DataFrame) -> tuple:
    """사용자 정량 지표와 ETF 클러스터 특성 간 유사도 기반 매칭"""
    if 'Cluster' not in metrics_df.columns:
        logging.error("ETF 메트릭 데이터에 'Cluster' 열이 없습니다. 사용자-클러스터 매칭 불가.")
        return -1, [] # 매칭된 클러스터 ID, 추천 티커 리스트

    # 사용자 벡터: 리스크 스코어(1-5)와 기대 수익률(0.02-0.15) 사용
    # 스케일링을 고려해야 할 수 있으나, 여기서는 원본 값 사용.
    # 리스크 스코어는 높을수록 리스크 선호, 기대수익률도 높을수록 고수익 추구.
    # ETF의 Annual Volatility (리스크), Annual Return (수익률)과 방향성 일치.
    user_vector = np.array([
        user_quantitative_indicators['risk_score'], 
        user_quantitative_indicators['expected_return']
    ]).reshape(1, -1)

    # ETF 특징 벡터: 연간 변동성(Annual Volatility)과 연간 수익률(Annual Return) 사용
    # 이 값들의 스케일이 사용자 벡터와 매우 다를 수 있음. 코사인 유사도는 방향성을 보므로 스케일 불변.
    # 다만, 값의 범위가 너무 다르면 특정 축에 편향될 수 있음. 여기서는 그대로 사용.
    if 'Annual Volatility' not in metrics_df.columns or 'Annual Return' not in metrics_df.columns:
        logging.error("ETF 메트릭에 Annual Volatility 또는 Annual Return이 없습니다. 매칭 불가.")
        return -1, []
        
    etf_features_for_matching = metrics_df[['Annual Volatility', 'Annual Return']].copy()
    
    # 결측값 처리 (만약을 위해)
    etf_features_for_matching.fillna(0, inplace=True)
    
    if etf_features_for_matching.empty:
        logging.error("유효한 ETF 특징 벡터가 없습니다. 매칭 불가.")
        return -1, []

    # 코사인 유사도 계산 (사용자 벡터 vs 각 ETF의 특징 벡터)
    # (1,2) vs (N,2) -> (1,N) 결과. flatten()으로 (N,) 배열로.
    similarity_scores = cosine_similarity(user_vector, etf_features_for_matching.values).flatten()
    metrics_df['UserSimilarity'] = similarity_scores # 원본 metrics_df에 유사도 저장 (나중에 정렬에 사용)

    # 클러스터별 평균 사용자 유사도 계산
    # 클러스터 레이블이 없는 ETF는 제외 (만약 있다면)
    if metrics_df['Cluster'].isnull().any():
        logging.warning("일부 ETF에 클러스터 레이블이 없습니다. 해당 ETF는 매칭에서 제외됩니다.")
        # metrics_df_clustered = metrics_df.dropna(subset=['Cluster'])
        # cluster_avg_similarity = metrics_df_clustered.groupby('Cluster')['UserSimilarity'].mean()
    # 현재는 클러스터링 실패 시 0으로 채우므로, 모든 ETF에 클러스터가 있다고 가정.
    
    # 모든 ETF가 단일 클러스터(예: 0)로 묶인 경우, 해당 클러스터가 최선이 됨.
    if metrics_df['Cluster'].nunique() == 1:
        best_cluster_id = metrics_df['Cluster'].unique()[0]
        logging.info(f"모든 ETF가 단일 클러스터({best_cluster_id})에 속합니다. 해당 클러스터를 사용합니다.")
    else:
        cluster_avg_similarity = metrics_df.groupby('Cluster')['UserSimilarity'].mean()
        if cluster_avg_similarity.empty:
            logging.error("클러스터별 평균 유사도 계산 실패. 매칭 불가.")
            return -1, []
        best_cluster_id = cluster_avg_similarity.idxmax() # 평균 유사도가 가장 높은 클러스터 ID

    # 선택된 최적 클러스터에 속하는 ETF들을 사용자 유사도 순으로 정렬하여 추천
    # (주의: 여기서 UserSimilarity는 개별 ETF와 사용자의 유사도)
    # 클러스터 선택은 '평균' 유사도로, 클러스터 내 추천은 '개별' 유사도로.
    
    # 필터링 전: 선택된 클러스터 내 모든 ETF (유사도 높은 순)
    # UserSimilarity가 metrics_df에 추가되었으므로, 이를 사용.
    # best_cluster_etfs_df = metrics_df[metrics_df['Cluster'] == best_cluster_id].sort_values('UserSimilarity', ascending=False)
    # recommended_tickers_in_cluster = best_cluster_etfs_df.index.tolist()

    # 사용자 시장 선호도 필터링은 match_user_to_cluster 함수 외부(main)에서 처리하거나, 여기서 처리
    # 여기서는 일단 클러스터 내 모든 ETF를 유사도 순으로 반환하고, main에서 추가 필터링 및 테마 적용.
    # 또는, 시장 선호도에 따라 클러스터 내 ETF를 먼저 필터링하고, 그 중에서 유사도 높은 것 추천.

    # 시장 선호도 적용 방식:
    # 1. 선호 시장 ETF만으로 클러스터 매칭 (입력 metrics_df를 먼저 필터링) -> 클러스터 구성이 달라질 수 있음
    # 2. 전체 ETF로 클러스터 매칭 후, 결과 클러스터 내에서 선호 시장 ETF 필터링 (현재 방식과 유사)
    # 2번 방식이 클러스터의 일반적 특성을 먼저 파악하고 세부 조정하는 것이므로 더 적합할 수 있음.

    # 여기서는 best_cluster_id와, 해당 클러스터 내 ETF들을 유사도 순으로 정렬한 metrics_df의 서브셋을 반환
    # (또는 티커 리스트만)
    
    # 최종 추천은 클러스터 내에서 유사도 높은 순 + 시장 선호도 + 테마 선호도 고려
    # 이 함수는 "최적 클러스터 ID"와 "해당 클러스터 내 ETF들의 정렬된 리스트(또는 metrics)"를 반환하는 역할.
    
    # 선택된 클러스터 내 ETF들을 유사도(UserSimilarity) 순으로 정렬
    cluster_etfs_df = metrics_df[metrics_df['Cluster'] == best_cluster_id].sort_values('UserSimilarity', ascending=False)
    
    # 시장 선호도 필터링 (user_quantitative_indicators에 market_scores 있음)
    market_pref_code = None # 1: KR, 2: US, 3: Both
    # user_quantitative_indicators에서 market_preference 원본 값 필요.
    # collect_user_input()의 user_profile['market_preference']를 써야 함.
    # 이 함수 파라미터에 user_profile 자체를 넘기거나, market_preference 값을 추가로 받아야 함.
    # 임시로, user_quantitative_indicators에 'market_preference_code'가 있다고 가정.
    # (derive_user_quantitative_indicators 함수에 추가하거나, main에서 user_profile 전달)
    # 여기서는 user_profile을 직접 받는 것으로 수정.

    # === 함수 시그니처 변경 필요 ===
    # def match_user_to_cluster(user_profile: dict, user_quantitative_indicators: dict, metrics_df: pd.DataFrame) -> tuple:
    # market_pref_code = user_profile['market_preference']
    #
    # filtered_cluster_etfs_df = cluster_etfs_df
    # if market_pref_code == 1: # 한국 선호
    #     filtered_cluster_etfs_df = cluster_etfs_df[cluster_etfs_df['Market'] == 'KR']
    # elif market_pref_code == 2: # 미국 선호
    #     filtered_cluster_etfs_df = cluster_etfs_df[cluster_etfs_df['Market'] == 'US']
    # # market_pref_code == 3 (상관없음)이면 필터링 안 함
    #
    # # 만약 필터링 후 ETF가 하나도 없다면, 필터링 전 목록 사용 (선택사항)
    # if filtered_cluster_etfs_df.empty and not cluster_etfs_df.empty :
    #     logging.info(f"선호 시장({market_pref_code}) 필터링 결과 클러스터 {best_cluster_id} 내 ETF 없음. 필터링 전 목록 일부 사용 고려.")
    #     # recommended_tickers = cluster_etfs_df.index.tolist() # 필터링 없이 전체 사용
    #     # 또는 빈 리스트 반환하여 상위에서 처리
    #     recommended_tickers = []
    # else:
    #     recommended_tickers = filtered_cluster_etfs_df.index.tolist()
    #
    # 위 시장 선호도 필터링은 main에서 최종 추천 시 하는 것이 더 깔끔할 수 있음.
    # 여기서는 best_cluster_id와, 해당 클러스터 전체의 정렬된 티커 리스트 반환.
    
    recommended_tickers_from_cluster = cluster_etfs_df.index.tolist()

    logging.info(f"사용자에게 가장 적합한 ETF 클러스터 ID: {best_cluster_id} (평균 유사도 기반)")
    logging.info(f"클러스터 {best_cluster_id} 내 ETF 수: {len(recommended_tickers_from_cluster)}")
    
    return best_cluster_id, recommended_tickers_from_cluster


def plot_risk_return_matrix(metrics_df: pd.DataFrame, file_name="etf_risk_return_matrix.html"):
    """위험-수익 매트릭스 시각화 (클러스터 정보 포함)"""
    if metrics_df.empty:
        logging.warning("시각화할 데이터가 없습니다.")
        return

    plot_data = metrics_df.copy()
    # Cluster 컬럼이 없을 경우 대비 (숫자형이어야 plotly에서 연속형으로 인식하지 않도록 str 변환)
    plot_data['Cluster'] = plot_data.get('Cluster', pd.Series(0, index=plot_data.index)).astype(str)
    
    # Plotly Express의 size 파라미터는 양수여야 함. Sharpe Ratio가 음수일 수 있으므로 조정.
    # 예: (Sharpe Ratio + смещение) * 스케일링 또는 고정 크기 사용.
    # 여기서는 샤프지수를 기반으로 하되, 음수이거나 0에 가까우면 최소 크기 보장.
    # min_plot_size = 5, scaling_factor = 5, offset = 2 (Sharpe -1 ~ 1 가정 시 (1~3)*5 = 5~15 크기)
    plot_data['PlotSize'] = np.maximum(5, (plot_data['Sharpe Ratio'].fillna(0) + 2) * 5)


    fig = px.scatter(
        plot_data,
        x='Annual Volatility',
        y='Annual Return',
        color='Cluster', # 클러스터별 색상
        size='PlotSize',  # 점 크기 (샤프지수 등에 기반)
        hover_name=plot_data.index, # 마우스 오버 시 ETF 티커 표시
        hover_data={ # 마우스 오버 시 추가 정보
            'Market': True, # ETF 시장 (KR/US)
            'Sharpe Ratio': ':.2f',
            'Max Drawdown': ':.2%',
            'Sortino Ratio': ':.2f',
            'Calmar Ratio': ':.2f',
            'Cluster': True, # 클러스터 ID
            'UserSimilarity': ':.2f' if 'UserSimilarity' in plot_data.columns else False # 사용자 유사도 (있을 경우)
        },
        title='ETF 위험-수익 매트릭스 (클러스터링 결과)',
        labels={
            'Annual Volatility': '연간 변동성 (Annual Volatility)',
            'Annual Return': '연간 수익률 (Annual Return)',
            'Cluster': '클러스터 ID'
        },
        color_discrete_sequence=px.colors.qualitative.Plotly # 다채로운 색상 사용
    )

    fig.update_layout(
        width=1200, # 차트 너비
        height=800, # 차트 높이
        template='plotly_white', # 배경 테마
        legend_title_text='클러스터 ID',
        title_x=0.5, # 제목 중앙 정렬
        xaxis_title="연간 변동성 (위험)",
        yaxis_title="연간 수익률",
        # x축, y축 범위 자동 조정 또는 명시적 설정 가능
        # xaxis_range=[min_vol, max_vol],
        # yaxis_range=[min_ret, max_ret],
    )
    
    # y=x 기준선 추가 (예: 샤프지수 1에 해당하는 선 등) - 선택사항
    # fig.add_shape(type="line", x0=0, y0=0, x1=plot_data['Annual Volatility'].max(), y1=plot_data['Annual Volatility'].max(), line=dict(color="grey", dash="dash"))

    # Plotly 렌더링: 기본 브라우저, 실패 시 HTML 파일로 저장
    try:
        # Colab, Jupyter 등 환경에 따라 적절한 렌더러 자동 선택 시도
        # pio.renderers.default = "jupyterlab" or "colab" or "vscode" or "browser"
        # 여기서는 우선 browser 시도
        pio.renderers.default = 'browser'
        fig.show()
    except Exception as e:
        logging.error(f"Plotly 브라우저 시각화 오류: {e}. HTML 파일로 저장합니다.")
        try:
            fig.write_html(file_name)
            print(f"시각화가 브라우저에 표시되지 않았습니다. 대신 '{file_name}' 파일로 저장되었습니다.")
            print(f"로컬 환경에서 해당 HTML 파일을 열어 확인하세요.")
        except Exception as e_html:
            logging.error(f"HTML 파일 저장 실패: {e_html}")


# --- Main Execution ---
def main():
    """메인 실행 함수"""
    if not check_and_install_packages() or fdr is None: # fdr 모듈 로드 확인
        sys.exit("필수 패키지 설치 또는 FinanceDataReader 초기화 실패. 프로그램을 종료합니다.")

    # 분석 대상 ETF 목록 (한국 + 미국)
    # 테마 정보 예시 (향후 외부 파일이나 API로 관리하는 것이 이상적)
    # {티커: 테마명} 형식. 테마명은 사용자 입력과 일치하도록 (기술, 에너지, 헬스케어 등)
    global etf_theme_map 
    etf_theme_map = {
        # 기술 관련
        'QQQ': '기술', 'XLK': '기술', 'SOXX': '기술', 'BOTZ': '기술', # BOTZ는 로보틱스/AI
        'ARKK': '기술', # ARKK는 광범위 혁신 기술
        '133690': '기술', # KODEX 반도체
        '232080': '기술', # TIGER TOP10 IT ETN (ETN은 ETF와 다르지만 예시로 포함)
        '371460': '기술', # TIGER Fn 반도체 TOP10
        '379800': '기술', # KODEX K-메타버스액티브
        '453950': '기술', # KODEX 미국반도체MV
        '309210': '기술', # TIGER KRX BBIG K-뉴딜 (BBIG 중 IT 비중 높음)
        '114800': '기술', # KODEX IT (삼성전자 비중 높음)

        # 에너지 관련
        'XLE': '에너지', 'USO': '에너지', # USO는 원유 선물
        'URA': '에너지', # URA는 우라늄 (원자력)
        'TAN': '에너지', # TAN은 태양광
        'ICLN': '에너지', # ICLN은 글로벌 청정 에너지
        # '102110': '에너지', # TIGER 200 에너지화학레버리지 (레버리지는 제외 고려)

        # 헬스케어 관련
        'XLV': '헬스케어',
        '277630': '헬스케어', # KODEX 바이오
        '305720': '헬스케어', # TIGER KRX헬스케어
        # '371870': '헬스케어', # KODEX K-이노베이션액티브 (헬스케어 포함 가능성, 주 운용 전략 확인 필요)

        # 기타 시장 지수, 채권, 원자재 등은 특정 테마로 분류하지 않거나 '기타'로 간주
        'SPY': '시장지수', 'DIA': '시장지수', 'IWM': '시장지수', 'VTI': '시장지수', 'VOO': '시장지수',
        'AGG': '채권', 'TLT': '채권', 'BND': '채권',
        'GLD': '원자재', 'SLV': '원자재', 'GDX': '원자재', # GDX는 금광회사
        '069500': '시장지수', # KODEX 200
        # ... (나머지 ETF들도 필요시 테마 분류) ...
    }
    # 사용자 입력 테마 코드와 매핑 (질문 번호와 맞춤)
    global user_theme_code_to_name_map
    user_theme_code_to_name_map = {
        2: '기술',
        3: '에너지',
        4: '헬스케어',
        # 1: 없음, 5: 기타는 특정 테마명으로 매핑하지 않음
    }


    kr_etfs_provided = [
    '069500',  # KODEX 200
    '102110',  # TIGER 200 에너지화학레버리지
    '114800',  # KODEX IT
    '132030',  # TIGER 200 헬스케어
    '133690',  # KODEX 반도체
    '148020',  # TIGER 200 금융
    '153130',  # KODEX 배당성장
    '232080',  # TIGER TOP10 IT
    '251340',  # ARIRANG 고배당주
    '278530',  # KODEX 차이나항셍테크
    '277630',  # KODEX 바이오
    '309210',  # TIGER KRX BBIG K-뉴딜
    '305720',  # TIGER KRX헬스케어
    '364990',  # KODEX China H
    '371460',  # TIGER Fn 반도체 TOP10
    '379800',  # KODEX K-메타버스액티브
    '381170',  # KODEX MSCI Korea
    '453950',  # KODEX 미국반도체MV
    '091160',  # KOSEF 국고채3년
    '069660',  # ARIRANG 고배당채권혼합
    '280940',  # ARIRANG 코스닥150
    '114460',  # KODEX 코스닥150
    '130680',  # KODEX 자동차
    '305050',  # KODEX 2차전지산업
    '379780',  # KODEX K-뉴딜
    '261240',  # TIGER 일본니케이225
    '381560',  # KODEX 미국나스닥100
    '148070',  # ARIRANG S&P글로벌인터네셔널
    '236360',  # TIGER 선진국MSCI
    '260780'  # KOSEF 미국달러인덱스선물레버리지
    ]
    us_etfs_provided = [    'SPY', 'VOO', 'VTI', 'IWM', 'QQQ',
    # 섹터별 대표
    'XLK', 'XLF', 'XLY', 'XLP', 'XLI', 'XLU', 'XLC', 'XLB',
    # 스타일·규모
    'VTV', 'VUG', 'VB', 'VEA', 'VWO',
    # 채권
    'AGG', 'BND', 'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'TIP',
    # 원자재·대체
    'GLD', 'SLV', 'DBC', 'USO', 'UNG', 'PPLT',
    # 테마·혁신
    'ARKK', 'BOTZ', 'TAN', 'ICLN', 'PBW', 'PLUG',
    # REITs
    'VNQ', 'SCHH', 'IYR',
    # 국제·신흥국
    'EFA', 'EEM', 'IEFA', 'VWO', 'EMB',
    # 기타 인기 ETF
    'SCHD', 'DIA', 'EWY', 'EWZ', 'EWU', 'EWH', 'EWG', 'EWC', 'EWJ', 'EWT'
    ]
    all_initial_tickers = sorted(list(set(kr_etfs_provided + us_etfs_provided)))

    # --- 1. 사용자 입력 수집 ---
    user_profile = collect_user_input()
    print("\n--- 사용자 프로필 요약 ---")
    for key, val in user_profile.items():
        print(f"- {key}: {val}")
    print("-" * 25)

    # --- 2. 분석 기간 설정 ---
    end_date_dt = datetime.now()
    # 투자 기간 (horizon_map)에 따라 데이터 시작 날짜 동적 설정
    # 1: 1년 미만(1년 데이터), 2: 1-3년(3년 데이터), ..., 5: 10년 초과(10년 데이터)
    # 데이터가 길수록 안정적이나, 너무 길면 과거 패턴이 현재와 다를 수 있음.
    # 여기서는 사용자가 선택한 기간을 최대한 반영하되, 최대 10년으로 제한.
    horizon_years_map = {1: 1, 2: 3, 3: 5, 4: 10, 5: 10} 
    data_period_years = horizon_years_map.get(user_profile['investment_horizon'], 5) # 기본 5년
    start_date_dt = end_date_dt - relativedelta(years=data_period_years)
    
    start_date_str = start_date_dt.strftime('%Y-%m-%d')
    end_date_str = end_date_dt.strftime('%Y-%m-%d')
    logging.info(f"데이터 분석 기간: {start_date_str} ~ {end_date_str} ({data_period_years}년)")

    # --- 3. ETF 가격 데이터 수집 ---
    etf_price_data, successful_tickers = fetch_etf_data_with_retry(all_initial_tickers, start_date_str, end_date_str)
    
    min_etfs_for_analysis = 5 # 분석에 필요한 최소 ETF 수 (클러스터링 등 고려)
    if len(successful_tickers) < min_etfs_for_analysis:
        print(f"오류: 분석 가능한 ETF 데이터가 {len(successful_tickers)}개로 너무 적습니다 (최소 {min_etfs_for_analysis}개 필요). 프로그램을 종료합니다.")
        return
    
    # 성공적으로 수집된 티커만 이후 분석에 사용
    # etf_price_data는 이미 successful_tickers 기준으로 컬럼이 필터링되어 있음.

    # --- 4. 수익률 계산 ---
    # 일별 로그 수익률 사용: log(P_t / P_{t-1})
    returns_df = np.log(etf_price_data / etf_price_data.shift(1))
    # 첫 행은 NaN이므로 제거, 모든 값이 NaN인 행/열도 제거
    returns_df = returns_df.iloc[1:].dropna(axis=0, how='all').dropna(axis=1, how='all')

    if returns_df.empty or returns_df.shape[1] < min_etfs_for_analysis: # 수익률 계산 후에도 ETF 수 체크
        print("오류: 유효한 수익률 데이터가 부족하여 분석을 진행할 수 없습니다.")
        return

    # --- 5. 리스크 메트릭 계산 ---
    risk_free_rate_val = fetch_risk_free_rate(start_date_str, end_date_str)
    logging.info(f"산출된 연간 무위험 수익률: {risk_free_rate_val:.4f} ({risk_free_rate_val*100:.2f}%)")
    
    metrics_df = calculate_risk_metrics(returns_df, risk_free_rate_val)
    # 각 ETF가 한국(KR)인지 미국(US)인지 구분하는 'Market' 컬럼 추가
    metrics_df['Market'] = ['KR' if tk.isdigit() and len(tk) == 6 else 'US' for tk in metrics_df.index]

    # --- 6. 클러스터링 ---
    # 클러스터링에 사용할 주요 특징(features) 선택
    # Sharpe, Sortino, Calmar는 이미 Return과 Volatility를 반영하므로, 과다포함 주의.
    # Skewness, Kurtosis는 분포의 비대칭성과 꼬리 위험을 나타냄.
    clustering_features = [
        'Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Max Drawdown',
        'Sortino Ratio', 'Calmar Ratio', 'Skewness', 'Kurtosis'
        # 'Rolling Return', 'Rolling Volatility', 'Recent Return', 'Recent Volatility' # 기간 짧으면 NaN 많음
    ]
    # 사용 가능한 feature만 선택 (데이터 기간에 따라 Rolling/Recent 지표가 없을 수 있음)
    available_clustering_features = [f for f in clustering_features if f in metrics_df.columns]
    clustering_input_df = metrics_df[available_clustering_features].copy()
    
    # 클러스터링 입력 데이터에서 무한대 값 및 NaN 처리 (RobustScaler 전에)
    clustering_input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    clustering_input_df.fillna(0, inplace=True) # NaN은 0으로 대체 (또는 평균/중앙값으로 대체 고려)

    if clustering_input_df.shape[0] < min_etfs_for_analysis: # 클러스터링할 ETF가 너무 적으면 의미 없음
        print(f"오류: 클러스터링에 사용할 ETF 데이터가 {clustering_input_df.shape[0]}개로 부족합니다.")
        metrics_df['Cluster'] = 0 # 모든 ETF를 단일 클러스터로 처리
        logging.warning("클러스터링 스킵. 모든 ETF를 단일 그룹(0)으로 간주합니다.")
    else:
        try:
            # k_range는 최대 클러스터링할 ETF 수 미만이어야 함.
            # 예: ETF 10개면 k는 2~9까지 가능. range(2, len(clustering_input_df))
            # optimize_clustering 내부에서 k_range와 데이터 수 비교하므로, 여기선 최대 10개로.
            max_k_for_clustering = min(10, clustering_input_df.shape[0] -1 if clustering_input_df.shape[0] >1 else 1)
            
            umap_result_data, cluster_labels = optimize_clustering(
                clustering_input_df, 
                k_range=range(2, max_k_for_clustering + 1 if max_k_for_clustering >=2 else 3), # k 최소 2, 최대 max_k
                random_state=42
            )
            
            # optimize_clustering 결과 레이블을 metrics_df에 할당
            # cluster_labels는 clustering_input_df의 인덱스 순서와 동일해야 함.
            if cluster_labels is not None and len(cluster_labels) == len(clustering_input_df):
                metrics_df['Cluster'] = cluster_labels
            else:
                # 이 경우, clustering_input_df와 metrics_df의 인덱스가 다를 수 있음 (만약 중간에 필터링 되었다면)
                # 하지만 현재 로직에서는 동일.
                logging.error("클러스터링 결과 레이블 길이가 원본 데이터와 불일치. 클러스터링 실패로 간주.")
                metrics_df['Cluster'] = 0 # 실패 시 단일 클러스터
        except Exception as e:
            print(f"클러스터링 과정에서 예외 발생: {e}")
            metrics_df['Cluster'] = 0 # 예외 발생 시 단일 클러스터
            logging.warning("클러스터링 오류로 모든 ETF를 단일 그룹(0)으로 간주합니다.")

    # --- 7. 사용자 정량 지표 도출 및 클러스터 매칭 ---
    user_quantitative_indicators = derive_user_quantitative_indicators(user_profile)
    
    # 사용자-클러스터 매칭 (UserSimilarity 점수도 metrics_df에 추가됨)
    # 이 함수는 user_profile 정보도 필요로 할 수 있음 (시장 선호도 등). 여기서는 일단 user_quantitative_indicators만.
    # match_user_to_cluster 함수 내부에서 UserSimilarity 계산 후 metrics_df에 추가함.
    matched_cluster_id, tickers_in_best_cluster = match_user_to_cluster(user_quantitative_indicators, metrics_df)

    # --- 8. 시각화 ---
    plot_risk_return_matrix(metrics_df, "etf_risk_return_matrix_final.html")

    # --- 9. 협업 필터링 추천 (선택적) ---
    # 사용자-ETF 선호도 데이터 파일 경로 (필요시 수정)
    user_etf_pref_file = 'C:\\python_works\\personal_projects\\user_etf_preferences.xlsx'
    user_etf_pref_data = load_user_etf_preferences(user_etf_pref_file)
    
    cf_recommendations = []
    if not user_etf_pref_data.empty:
        cf_recommendations = collaborative_filtering_recommendation(user_profile, metrics_df, user_etf_pref_data)
    else:
        print(f"\n안내: 협업 필터링을 위한 사용자 선호도 데이터('{user_etf_pref_file}')가 없어 해당 추천은 생략됩니다.")
        print("이 기능을 사용하려면, 해당 파일에 다른 사용자들의 프로필과 선호 ETF 정보를 입력해주세요.")
        print("필수 컬럼: risk_tolerance, investment_horizon, goal, experience, loss_aversion, theme_preference, preferred_etfs (쉼표로 구분된 티커)")


    # --- 10. 최종 추천 결합 및 결과 제시 ---
    # 클러스터 기반 추천 (tickers_in_best_cluster)과 협업 필터링 추천(cf_recommendations) 결합
    # 중복 제거 후 최종 추천 목록 생성
    
    # 기본 추천은 클러스터 기반
    final_recommended_tickers = list(dict.fromkeys(tickers_in_best_cluster)) # 순서 유지하며 중복 제거

    # 협업 필터링 결과를 추가 (만약 있다면, 클러스터 추천 결과 앞에 추가하거나 가중치 부여)
    # 여기서는 set으로 합친 후 다시 리스트로 변환 (순서 일부 변경될 수 있음)
    # 좀 더 정교하게는, 두 추천 목록에 가중치를 두거나, 우선순위를 정할 수 있음.
    # 예: CF 추천 결과를 우선적으로 몇 개 보여주고, 나머지는 클러스터 기반으로 채움.
    # 여기서는 CF 결과를 클러스터 결과에 추가하고, 이후 정렬에서 종합적으로 고려.
    if cf_recommendations:
        # 기존 클러스터 추천 결과에 없는 CF 추천만 추가
        for cf_tk in cf_recommendations:
            if cf_tk not in final_recommended_tickers:
                final_recommended_tickers.append(cf_tk)
        logging.info(f"협업 필터링 추천 {len(cf_recommendations)}개가 클러스터 기반 추천에 추가/고려됩니다.")


    print("\n" + "=" * 60)
    print(" 최종 ETF 추천 결과".center(60))
    print("=" * 60)

    if matched_cluster_id != -1 and final_recommended_tickers:
        print(f"✅ 사용자 투자 성향에 가장 적합한 ETF 그룹(클러스터 ID): {matched_cluster_id}")
        
        # 추천된 티커들에 대한 메트릭 정보 가져오기 (metrics_df에서)
        # final_recommended_tickers 중 metrics_df.index에 없는 것이 있을 수 있으므로 필터링
        valid_final_tickers = [tk for tk in final_recommended_tickers if tk in metrics_df.index]
        if not valid_final_tickers:
            print("\n⚠️ 추천 가능한 ETF가 최종적으로 없습니다 (필터링 또는 데이터 부재).")
        else:
            recommendation_details_df = metrics_df.loc[valid_final_tickers].copy()

            # 1. 사용자 시장 선호도 필터링
            market_pref_code = user_profile['market_preference']
            if market_pref_code == 1: # 한국만 선호
                recommendation_details_df = recommendation_details_df[recommendation_details_df['Market'] == 'KR']
            elif market_pref_code == 2: # 미국만 선호
                recommendation_details_df = recommendation_details_df[recommendation_details_df['Market'] == 'US']
            # market_pref_code == 3 (상관없음)이면 모든 시장 포함

            if recommendation_details_df.empty:
                print(f"\n⚠️ 선호하시는 시장({['한국', '미국', '상관없음'][market_pref_code-1]})에는 현재 추천 ETF가 없습니다.")
            else:
                # 2. 사용자 테마 선호도 반영 (정렬 가중치)
                user_selected_theme_code = user_profile['theme_preference']
                preferred_theme_name = user_theme_code_to_name_map.get(user_selected_theme_code)

                recommendation_details_df['ThemeMatchScore'] = 0 # 테마 일치 점수 컬럼
                if preferred_theme_name: # 특정 테마 선호 시
                    logging.info(f"사용자 선호 테마: {preferred_theme_name}")
                    for tk in recommendation_details_df.index:
                        etf_actual_theme = etf_theme_map.get(tk) # 미리 정의된 ETF-테마 맵
                        if etf_actual_theme == preferred_theme_name:
                            recommendation_details_df.loc[tk, 'ThemeMatchScore'] = 1 # 일치 시 1점
                            logging.info(f"ETF {tk}는 선호 테마 '{preferred_theme_name}'와 일치합니다.")
                
                # 최종 정렬: 1순위 테마일치(내림), 2순위 사용자유사도(내림), 3순위 샤프지수(내림)
                # UserSimilarity는 match_user_to_cluster에서 계산됨. 없으면 샤프지수만으로.
                sort_by_cols = ['ThemeMatchScore']
                sort_ascending_flags = [False]

                if 'UserSimilarity' in recommendation_details_df.columns:
                    sort_by_cols.append('UserSimilarity')
                    sort_ascending_flags.append(False)
                
                sort_by_cols.append('Sharpe Ratio') # 백업 정렬 기준
                sort_ascending_flags.append(False)

                sorted_recommendations_df = recommendation_details_df.sort_values(
                    by=sort_by_cols,
                    ascending=sort_ascending_flags
                )

                print(f"\n✨ 추천 ETF 목록 ({len(sorted_recommendations_df)}개, 우선순위 정렬):")
                top_n_to_display = 10
                for i, ticker_code in enumerate(sorted_recommendations_df.index):
                    if i >= top_n_to_display:
                        if len(sorted_recommendations_df) > top_n_to_display:
                            print(f"  ... (상위 {top_n_to_display}개 외 {len(sorted_recommendations_df) - top_n_to_display}개 ETF 더 보기 가능)")
                        break
                    
                    etf_info = sorted_recommendations_df.loc[ticker_code]
                    market_info = etf_info['Market']
                    user_sim_score = f"{etf_info.get('UserSimilarity',0):.2f}" # 없을 경우 0
                    annual_ret_pct = f"{etf_info['Annual Return']*100:.2f}%"
                    annual_vol_pct = f"{etf_info['Annual Volatility']*100:.2f}%"
                    sharpe_val = f"{etf_info['Sharpe Ratio']:.2f}"
                    theme_match_score_val = etf_info.get('ThemeMatchScore', 0)
                    
                    theme_indicator = ""
                    if preferred_theme_name and theme_match_score_val == 1:
                        theme_indicator = f"(선호 테마: {preferred_theme_name} 일치)"
                    elif preferred_theme_name and etf_theme_map.get(ticker_code): # 선호테마는 있지만 불일치, ETF테마는 있는경우
                         theme_indicator = f"(테마: {etf_theme_map.get(ticker_code)})"


                    print(f"  - {ticker_code} ({market_info}) {theme_indicator}")
                    print(f"      유사도: {user_sim_score}, 연수익률: {annual_ret_pct}, 연변동성: {annual_vol_pct}, 샤프지수: {sharpe_val}")

        # 선택된 클러스터의 평균 통계 정보 (참고용)
        if matched_cluster_id in metrics_df['Cluster'].unique():
            print(f"\n📊 참고: 클러스터 {matched_cluster_id}의 평균 통계:")
            try:
                # numeric_only=True는 이후 버전에서 기본값이 될 수 있음
                cluster_avg_stats = metrics_df[metrics_df['Cluster'] == matched_cluster_id].select_dtypes(include=np.number).mean()
                # 주요 지표만 선택해서 보여주기
                display_stats_keys = ['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Max Drawdown', 'UserSimilarity']
                stats_to_display = {k: v for k, v in cluster_avg_stats.items() if k in display_stats_keys}
                for stat_name, stat_val in stats_to_display.items():
                    print(f"  - 평균 {stat_name}: {stat_val:.4f}")
            except Exception as e:
                print(f"  클러스터 통계 계산 중 오류: {e}")
        
    elif matched_cluster_id != -1 : # 추천 클러스터는 있으나, 그 안에서 필터링 후 ETF가 없는 경우
        print(f"⚠️ 추천된 클러스터({matched_cluster_id}) 내에서는 사용자님의 세부 조건(시장/테마)에 맞는 ETF를 찾지 못했습니다.")
        # 해당 클러스터의 모든 ETF라도 보여줄지 여부
        all_in_cluster = metrics_df[metrics_df['Cluster'] == matched_cluster_id].index.tolist()
        if all_in_cluster:
            print(f"\n참고: 클러스터 {matched_cluster_id}에 포함된 전체 ETF 목록 (최대 10개):")
            print(f"  - {', '.join(all_in_cluster[:10])}{'...' if len(all_in_cluster) > 10 else ''}")
            print("  이 ETF들을 직접 검토해보시는 것을 추천합니다.")
    else: # matched_cluster_id가 -1 (매칭 실패)
        print("❌ 분석 결과, 현재 사용자님의 투자 성향에 적합한 ETF를 추천하기 어렵습니다.")
        print("   입력값을 다양하게 변경해보시거나, 분석 대상 ETF 목록을 늘려보는 것을 고려해보세요.")

    print("\n" + "=" * 60)
    print(" 분석 완료 ".center(60, "="))
    print("=" * 60)


if __name__ == "__main__":
    main()