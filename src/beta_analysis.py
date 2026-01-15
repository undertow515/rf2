"""
Beta 분포 매개변수 분석
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats


def beta_cdf(x, a, b):
    """Beta 분포 CDF"""
    return stats.beta.cdf(x, a, b)


class BetaDistributionAnalyzer:
    """Beta 분포 매개변수 분석 클래스"""
    
    def __init__(self, ecdf_patterns, events):
        """
        Parameters
        ----------
        ecdf_patterns : np.ndarray
            ECDF 패턴 배열
        events : list
            강수 이벤트 리스트
        """
        self.ecdf_patterns = ecdf_patterns
        self.events = events
        self.params_df = None
        
    def run(self):
        """Beta 분포 적합 실행"""
        print("  > Fitting Beta Distribution Parameters...")
        self._fit_beta_parameters()
        self._calculate_indicators()
        return self.params_df
    
    def _fit_beta_parameters(self):
        """Beta 매개변수 적합"""
        years = [e['start_time'].year for e in self.events]
        months = [e['start_time'].month for e in self.events]
        days = [e['start_time'].day for e in self.events]
        durations = [e['duration_hours'] for e in self.events]
        total_precips = [e['total_precip_mm'] for e in self.events]
        
        x = self.ecdf_patterns
        params_dict = {}
        
        for i in range(x.shape[0]):
            try:
                params, _ = curve_fit(
                    beta_cdf, 
                    np.linspace(0, 1, x.shape[1]), 
                    x[i], 
                    p0=[2, 5], 
                    maxfev=2000
                )
                params_dict[i] = params
            except RuntimeError:
                continue
        
        self.params_df = pd.DataFrame(params_dict).T
        self.params_df.columns = ['a', 'b']
        
        valid_indices = self.params_df.index
        self.params_df["year"] = [years[i] for i in valid_indices]
        self.params_df["month"] = [months[i] for i in valid_indices]
        self.params_df["day"] = [days[i] for i in valid_indices]
        self.params_df["duration_hours"] = [durations[i] for i in valid_indices]
        self.params_df["total_precip_mm"] = [total_precips[i] for i in valid_indices]
        
    def _calculate_indicators(self):
        """지표 계산"""
        self.params_df["a/b"] = self.params_df["a"] / self.params_df["b"]
        self.params_df["a*b"] = self.params_df["a"] * self.params_df["b"]
        self.params_df["log_ab"] = np.log(self.params_df["a*b"] + 1e-6)
        self.params_df["loga"] = np.log(self.params_df["a"] + 1e-6)
        self.params_df["logb"] = np.log(self.params_df["b"] + 1e-6)
        self.params_df['intensity'] = (
            self.params_df['total_precip_mm'] / self.params_df['duration_hours']
        )