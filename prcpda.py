import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.isotonic import IsotonicRegression


# --- 설정 (이전과 동일) ---
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 클래스 1: 강수 이벤트 추출 및 ECDF 생성
# ==============================================================================
class PrecipitationEventManager:
    """
    시간별 강수량 데이터로부터 강수 이벤트를 정의, 추출하고,
    각 이벤트의 무차원 ECDF(경험적 누적 분포 함수)를 생성합니다.
    """
    def __init__(self, hourly_df, min_duration_hours=10, min_total_precip=2.5, 
                 max_gap_hours=2, precip_threshold=0.1, target_segments=100, zero_precip_ratio_threshold=0.6):
        self.hourly_df = hourly_df.copy()
        self.target_segments = target_segments
        
        # 이벤트 정의 기준
        self.min_duration_hours = min_duration_hours
        self.min_total_precip = min_total_precip
        self.max_gap_hours = max_gap_hours
        self.precip_threshold = precip_threshold
        self.zero_precip_ratio_threshold = zero_precip_ratio_threshold  # 이벤트 내 무강수 시간 비율 임계값
        
        # 분석 결과 저장 변수
        self.precipitation_events = []
        self.ecdf_patterns = np.array([])

    def run(self):
        """이벤트 추출 및 ECDF 생성 파이프라인을 실행합니다."""
        print("--- PrecipitationEventManager 시작 ---")
        self._extract_precipitation_events()
        self._create_ecdf_patterns()
        print("✔️ 이벤트 추출 및 ECDF 생성 완료.")
        return self.precipitation_events, self.ecdf_patterns

    def _extract_precipitation_events(self):
        """연속적인 강수 이벤트를 추출합니다."""
        print("1단계: 강수 이벤트 추출 중...")
        is_wet_hour = self.hourly_df['precip_mm'] > self.precip_threshold
        smoothing_window = 2 * self.max_gap_hours + 1
        is_part_of_event = is_wet_hour.rolling(window=smoothing_window, center=True, min_periods=1).max().astype(bool)
        event_groups = (is_part_of_event != is_part_of_event.shift()).cumsum()

        for _, event_data in self.hourly_df.groupby(event_groups):
            if not is_part_of_event[event_data.index[0]]:
                continue

            duration = len(event_data)
            total_precip = event_data['precip_mm'].sum()
            if (event_data['precip_mm'] == 0).sum() / duration > self.zero_precip_ratio_threshold:
                continue

            if duration >= self.min_duration_hours and total_precip >= self.min_total_precip:
                self.precipitation_events.append({
                    'start_time': event_data.index[0],
                    'end_time': event_data.index[-1],
                    'duration_hours': duration,
                    'total_precip_mm': total_precip,
                    'hourly_precip': event_data['precip_mm'].values
                })
        print(f"추출된 강수 이벤트 수: {len(self.precipitation_events)}")

    def _create_ecdf_patterns(self):
        """추출된 이벤트로부터 무차원 ECDF 패턴을 생성합니다."""
        print("2단계: 무차원 ECDF 패턴 생성 중...")
        if not self.precipitation_events:
            print("경고: ECDF를 생성할 강수 이벤트가 없습니다.")
            return

        ecdf_list = []
        for event in self.precipitation_events:
            # 시간 길이를 정규화 (보간)
            original_indices = np.linspace(0, 1, len(event['hourly_precip']))
            target_indices = np.linspace(0, 1, self.target_segments)
            normalized_precip = np.interp(target_indices, original_indices, event['hourly_precip'])
            
            # 누적 강수량 계산 및 정규화
            cumulative = np.cumsum(normalized_precip)
            if cumulative[-1] > 0:
                ecdf_list.append(cumulative / cumulative[-1])
        
        self.ecdf_patterns = np.array(ecdf_list)

# ==============================================================================
# 클래스 2: HuffCurveAnalyzer (위에 제시된 수정된 버전 사용)
# ==============================================================================
class HuffCurveAnalyzer:
    """
    ECDF 패턴을 클러스터링 없이 직접 Huff 유형으로 분류하고,
    각 유형의 평균 곡선에 6차 다항식 회귀를 통해 대표 Huff 곡선을 생성합니다.
    """
    def __init__(self, ecdf_patterns):
        self.ecdf_patterns = ecdf_patterns
        
        # 분석 결과 저장 변수
        self.pattern_labels = []      # 각 패턴에 대한 Huff 라벨 리스트
        self.huff_averages = {}       # 라벨별 평균 ECDF 곡선
        self.polynomial_fits = {}     # 라벨별 6차 회귀식

    def run(self):
        """Huff 곡선 분석 파이프라인을 실행합니다."""
        print("\n--- HuffCurveAnalyzer (직접 라벨링 방식) 시작 ---")
        if self.ecdf_patterns.size == 0:
            print("경고: 분석할 ECDF 패턴이 없습니다.")
            return None, None
            
        self._label_patterns_by_huff_type()
        self._group_and_average_by_label()
        self._fit_polynomial_to_averages()
        print("✔️ Huff 곡선 분석 완료.")
        return self.pattern_labels, self.huff_averages

    def _label_patterns_by_huff_type(self):
        """개별 ECDF 패턴에 Huff 유형(Quartile) 라벨을 할당합니다."""
        print("1단계: 개별 ECDF 패턴에 Huff 유형 라벨링 중...")
        time_points = np.linspace(0, 1, self.ecdf_patterns.shape[1])
        
        for pattern in self.ecdf_patterns:
            # 각 사분위(quartile)에 내린 비의 양 계산
            q1_precip = np.interp(0.25, time_points, pattern)
            q2_precip = np.interp(0.50, time_points, pattern) - q1_precip
            q3_precip = np.interp(0.75, time_points, pattern) - (q1_precip + q2_precip)
            q4_precip = 1.0 - (q1_precip + q2_precip + q3_precip)
            
            quartile_precips = [q1_precip, q2_precip, q3_precip, q4_precip]
            max_quartile = np.argmax(quartile_precips) + 1
            
            self.pattern_labels.append(f"Quartile {max_quartile}")
        print(f"총 {len(self.pattern_labels)}개의 패턴 라벨링 완료.")

    def _group_and_average_by_label(self):
        """라벨별로 패턴을 그룹화하고, 각 그룹의 평균 ECDF를 계산합니다."""
        print("2단계: 라벨별 패턴 그룹화 및 평균 ECDF 계산 중...")
        # Pandas DataFrame을 사용하여 쉽게 그룹화 및 평균 계산
        df = pd.DataFrame(self.ecdf_patterns)
        df['label'] = self.pattern_labels
        
        grouped_means = df.groupby('label').mean()
        
        for label, avg_pattern in grouped_means.iterrows():
            self.huff_averages[label] = avg_pattern.values
            count = df['label'].value_counts()[label]
            print(f"  - {label} 유형 평균 계산 완료 (n={count})")

    def _fit_polynomial_to_averages(self):
        """각 Huff 유형의 평균 ECDF에 6차 다항식을 적합시킵니다."""
        print("3단계: 6차 다항 회귀식 계산 중...")
        x = np.linspace(0, 1, self.ecdf_patterns.shape[1])
        
        for label, avg_pattern in self.huff_averages.items():
            coeffs = np.polyfit(x, avg_pattern, 6)
            self.polynomial_fits[label] = np.poly1d(coeffs)
            print(f"  - {label} 유형의 회귀식 계산 완료.")

    def _fit_models_to_averages(self):
        """
        ★수정: 각 Huff 유형의 평균 ECDF에 두 가지 모델을 적합시킵니다.
        1. 6차 다항식 (단순 근사)
        2. 등위 회귀 (비감소 보장)
        """
        print("3단계: 모델 피팅 중 (다항 회귀 및 등위 회귀)...")
        x = np.linspace(0, 1, self.ecdf_patterns.shape[1])
        
        # 등위 회귀 결과를 저장할 새로운 딕셔너리
        self.isotonic_fits = {}

        for label, avg_pattern in self.huff_averages.items():
            # 1. 기존 6차 다항 회귀
            coeffs = np.polyfit(x, avg_pattern, 6)
            self.polynomial_fits[label] = np.poly1d(coeffs)
            
            # 2. 등위 회귀
            # IsotonicRegression 모델은 x값이 1차원 배열이어야 하므로 .reshape(-1, 1) 사용
            iso_reg = IsotonicRegression(y_min=0, y_max=1, increasing=True)
            self.isotonic_fits[label] = iso_reg.fit(x, avg_pattern)

            print(f"  - {label} 유형의 모델 피팅 완료.")

# ==============================================================================
# 클래스 3: AnalysisVisualizer (수정됨)
# ==============================================================================
class AnalysisVisualizer:
    """분석 결과를 시각화하는 클래스입니다."""
    def __init__(self, analyzer_results):
        self.analyzer = analyzer_results

    def plot_all_ecdfs_by_group(self, ecdf_patterns):
        """Huff 유형별 모든 ECDF 패턴과 평균 곡선을 시각화합니다."""
        huff_labels = sorted(self.analyzer.huff_averages.keys())
        n_groups = len(huff_labels)
        if n_groups == 0:
            return

        fig, axes = plt.subplots((n_groups + 1) // 2, 2, figsize=(15, 5 * ((n_groups + 1) // 2)), sharex=True, sharey=True)
        axes = axes.flatten()
        colors = plt.cm.viridis(np.linspace(0, 1, n_groups))
        
        patterns_df = pd.DataFrame(ecdf_patterns)
        patterns_df['label'] = self.analyzer.pattern_labels

        for i, label in enumerate(huff_labels):
            group_patterns = patterns_df[patterns_df['label'] == label].drop('label', axis=1).values
            
            for pattern in group_patterns:
                axes[i].plot(pattern, color=colors[i], alpha=0.1)
            
            avg_pattern = self.analyzer.huff_averages[label]
            axes[i].plot(avg_pattern, color=colors[i], linewidth=3)
            axes[i].set_title(f'{label} 유형 (n={len(group_patterns)})', fontsize=14)
            axes[i].grid(True, alpha=0.5)

        for j in range(n_groups, len(axes)):
            fig.delaxes(axes[j])

        fig.supxlabel('정규화된 시간', fontsize=12)
        fig.supylabel('누적 강수 비율', fontsize=12)
        fig.suptitle('Huff 유형별 ECDF 패턴 상세 시각화', fontsize=18)
        plt.tight_layout()
        plt.show()

    def plot_huff_curves_with_regression(self):
        """회귀 분석으로 생성된 Huff 곡선을 시각화하고 회귀식을 출력합니다."""
        huff_labels = sorted(self.analyzer.huff_averages.keys())
        if not huff_labels:
            return
            
        plt.figure(figsize=(12, 8))
        x_smooth = np.linspace(0, 1, 200)
        colors = plt.cm.viridis(np.linspace(0, 1, len(huff_labels)))

        print("\n--- 각 Huff 유형별 6차 다항 회귀식 ---")
        for i, label in enumerate(huff_labels):
            poly_func = self.analyzer.polynomial_fits[label]
            y_fit = poly_func(x_smooth)
            
            plt.plot(x_smooth, y_fit, color=colors[i], linewidth=2.5, label=f'{label}')
            print(f"\n[{label}]")
            print(poly_func)

        plt.title('Huff 유형별 6차 다항 회귀 곡선', fontsize=16)
        plt.xlabel('정규화된 시간 (Duration 0 to 1)')
        plt.ylabel('누적 강수 비율 (Cumulative Ratio)')
        plt.grid(True, alpha=0.5)
        plt.legend()
        plt.show()

# ==============================================================================
# --- 실행 파이프라인 ---
# ==============================================================================
if __name__ == '__main__':
    # 1. 데이터 로드
    df_hourly = pd.read_csv("./data/kma_precip_108_20010101_20241231_with_rolling.csv")
    df_hourly["datetime_kst"] = pd.to_datetime(df_hourly["datetime_kst"])
    df_hourly.set_index("datetime_kst", inplace=True)

    # 2. 클래스 인스턴스화 및 실행
    # 2-1. 이벤트 및 ECDF 생성
    event_manager = PrecipitationEventManager(df_hourly)
    events, ecdf_patterns = event_manager.run()

    if ecdf_patterns.size > 0:
        # 2-2. Huff 곡선 분석 (n_clusters 불필요)
        huff_analyzer = HuffCurveAnalyzer(ecdf_patterns)
        pattern_labels, huff_averages = huff_analyzer.run()
        
        # 2-3. 결과 시각화
        visualizer = AnalysisVisualizer(huff_analyzer)
        visualizer.plot_all_ecdfs_by_group(ecdf_patterns)
        visualizer.plot_huff_curves_with_regression()
    else:
        print("\n분석할 ECDF 패턴이 없어 파이프라인을 종료합니다.")