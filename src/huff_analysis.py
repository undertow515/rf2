"""
Huff Curve 분석
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


class HuffCurveAnalyzer:
    """Huff Curve 분석 클래스"""
    
    def __init__(self, ecdf_patterns):
        """
        Parameters
        ----------
        ecdf_patterns : np.ndarray
            ECDF 패턴 배열 (n_events, n_segments)
        """
        self.ecdf_patterns = ecdf_patterns
        self.pattern_labels = []
        self.huff_averages = {}
        self.polynomial_fits = {}
        self.isotonic_fits = {}

    def run(self):
        """Huff 분석 실행"""
        print("  > Analyzing Huff Curves...")
        if self.ecdf_patterns.size == 0:
            print("    ! Warning: No ECDF patterns to analyze.")
            return None, None
            
        self._label_patterns_by_huff_type()
        self._group_and_average_by_label()
        self._fit_models_to_averages()
        return self.pattern_labels, self.huff_averages

    def _label_patterns_by_huff_type(self):
        """Huff Quartile 라벨링"""
        time_points = np.linspace(0, 1, self.ecdf_patterns.shape[1])
        
        for pattern in self.ecdf_patterns:
            q1 = np.interp(0.25, time_points, pattern)
            q2 = np.interp(0.50, time_points, pattern) - q1
            q3 = np.interp(0.75, time_points, pattern) - (q1 + q2)
            q4 = 1.0 - (q1 + q2 + q3)
            
            quartile_precips = [q1, q2, q3, q4]
            max_quartile = np.argmax(quartile_precips) + 1
            self.pattern_labels.append(f"Q{max_quartile}")

    def _group_and_average_by_label(self):
        """라벨별 평균 ECDF 계산"""
        df = pd.DataFrame(self.ecdf_patterns)
        df['label'] = self.pattern_labels
        grouped_means = df.groupby('label').mean()
        
        for label, avg_pattern in grouped_means.iterrows():
            self.huff_averages[label] = avg_pattern.values

    def _fit_models_to_averages(self):
        """평균 ECDF에 모델 적합"""
        x = np.linspace(0, 1, self.ecdf_patterns.shape[1])
        
        for label, avg_pattern in self.huff_averages.items():
            # Polynomial Fit
            coeffs = np.polyfit(x, avg_pattern, 6)
            self.polynomial_fits[label] = np.poly1d(coeffs)
            
            # Isotonic Regression
            iso_reg = IsotonicRegression(y_min=0, y_max=1, increasing=True)
            self.isotonic_fits[label] = iso_reg.fit(x, avg_pattern)