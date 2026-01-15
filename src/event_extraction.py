"""
강수 이벤트 추출 및 ECDF 패턴 생성
"""

import numpy as np
import pandas as pd


class PrecipitationEventManager:
    """강수 이벤트를 추출하고 ECDF 패턴을 생성하는 클래스"""
    
    def __init__(self, hourly_df, min_duration_hours=10, min_total_precip=2.5, 
                 max_gap_hours=2, precip_threshold=0.1, target_segments=100, 
                 zero_precip_ratio_threshold=0.6):
        """
        Parameters
        ----------
        hourly_df : pd.DataFrame
            시간 단위 강수 데이터 (datetime index, 'precip_mm' column)
        min_duration_hours : int
            최소 지속시간 (시간)
        min_total_precip : float
            최소 총 강수량 (mm)
        max_gap_hours : int
            허용 최대 무강수 간격 (시간)
        precip_threshold : float
            강수 판정 기준 (mm)
        target_segments : int
            정규화된 시간 구간 수
        zero_precip_ratio_threshold : float
            이벤트 내 무강수 시간 비율 임계값
        """
        self.hourly_df = hourly_df.copy()
        self.target_segments = target_segments
        
        self.min_duration_hours = min_duration_hours
        self.min_total_precip = min_total_precip
        self.max_gap_hours = max_gap_hours
        self.precip_threshold = precip_threshold
        self.zero_precip_ratio_threshold = zero_precip_ratio_threshold
        
        self.precipitation_events = []
        self.ecdf_patterns = np.array([])

    def run(self):
        """이벤트 추출 및 ECDF 생성 실행"""
        print("  > Extracting precipitation events...")
        self._extract_precipitation_events()
        self._create_ecdf_patterns()
        return self.precipitation_events, self.ecdf_patterns

    def _extract_precipitation_events(self):
        """강수 이벤트 추출"""
        is_wet_hour = self.hourly_df['precip_mm'] > self.precip_threshold
        smoothing_window = 2 * self.max_gap_hours + 1
        is_part_of_event = is_wet_hour.rolling(
            window=smoothing_window, center=True, min_periods=1
        ).max().astype(bool)
        
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
                
        print(f"    - Total Events Extracted: {len(self.precipitation_events)}")

    def _create_ecdf_patterns(self):
        """ECDF 패턴 생성"""
        if not self.precipitation_events:
            return

        ecdf_list = []
        for event in self.precipitation_events:
            original_indices = np.linspace(0, 1, len(event['hourly_precip']))
            target_indices = np.linspace(0, 1, self.target_segments)
            normalized_precip = np.interp(
                target_indices, original_indices, event['hourly_precip']
            )
            
            cumulative = np.cumsum(normalized_precip)
            if cumulative[-1] > 0:
                ecdf_list.append(cumulative / cumulative[-1])
        
        self.ecdf_patterns = np.array(ecdf_list)