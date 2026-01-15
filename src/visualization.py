"""
시각화 모듈
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymannkendall as mk

# 스타일 설정
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


class HuffVisualizer:
    """Huff Curve 시각화 클래스"""
    
    def __init__(self, analyzer, city_name="Unknown"):
        self.analyzer = analyzer
        self.city_name = city_name
        
    def plot_all_ecdfs_by_group(self, ecdf_patterns):
        """Huff 유형별 모든 ECDF 패턴 시각화"""
        huff_labels = sorted(self.analyzer.huff_averages.keys())
        n_groups = len(huff_labels)
        if n_groups == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        axes = axes.flatten()
        colors = sns.color_palette("viridis", n_groups)
        
        patterns_df = pd.DataFrame(ecdf_patterns)
        patterns_df['label'] = self.analyzer.pattern_labels

        for i, label in enumerate(huff_labels):
            if i >= len(axes):
                break
                
            group_patterns = patterns_df[
                patterns_df['label'] == label
            ].drop('label', axis=1).values
            
            # 개별 이벤트
            for pattern in group_patterns:
                axes[i].plot(pattern, color='gray', alpha=0.05, linewidth=1)
            
            # 평균
            avg_pattern = self.analyzer.huff_averages[label]
            axes[i].plot(avg_pattern, color=colors[i], linewidth=3, label='Mean ECDF')
            
            axes[i].set_title(f'{label} Type (n={len(group_patterns)})', 
                            fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            
            if i >= 2:
                axes[i].set_xlabel('Normalized Duration')
            if i % 2 == 0:
                axes[i].set_ylabel('Cumulative Precip. Ratio')

        fig.suptitle(f'ECDF Patterns by Huff Quartile - {self.city_name}', 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

    def plot_huff_curves_with_regression(self):
        """회귀 분석으로 생성된 Huff 곡선 시각화"""
        huff_labels = sorted(self.analyzer.huff_averages.keys())
        if not huff_labels:
            return
            
        plt.figure(figsize=(10, 7))
        x_smooth = np.linspace(0, 1, 200)
        colors = sns.color_palette("viridis", len(huff_labels))

        for i, label in enumerate(huff_labels):
            poly_func = self.analyzer.polynomial_fits[label]
            y_fit = poly_func(x_smooth)
            plt.plot(x_smooth, y_fit, color=colors[i], linewidth=3, 
                    label=f'{label} Curve')

        plt.title(f'Huff Curves (6th Degree Poly. Fit) - {self.city_name}', 
                 fontsize=16)
        plt.xlabel('Normalized Duration (0 to 1)')
        plt.ylabel('Cumulative Precipitation Ratio')
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.show()


class BetaVisualizer:
    """Beta 분포 분석 시각화 클래스"""
    
    def __init__(self, params_df, city_name="Unknown"):
        self.params_df = params_df
        self.city_name = city_name
        
    def plot_parameter_distributions(self):
        """Beta 매개변수 분포 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # a/b 분포
        sns.histplot(self.params_df["a/b"], bins=50, kde=True, 
                    color='skyblue', ax=axes[0])
        axes[0].set_title(f'Beta Parameter (a/b) Distribution - {self.city_name}')
        axes[0].set_xlabel('a/b (Time Asymmetry: <1 Early, >1 Late)')
        axes[0].set_ylabel('Frequency')

        # log(a*b) 분포
        sns.histplot(self.params_df["log_ab"], bins=50, kde=True, 
                    color='salmon', ax=axes[1])
        axes[1].set_title(f'Log(a*b) Distribution - {self.city_name}')
        axes[1].set_xlabel('log(a*b) (Concentration: Higher = More Peaked)')
        axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
    def plot_trend_analysis(self):
        """추세 분석 시각화"""
        params_df_yearly = self.params_df.groupby('year').mean()
        is_summer = self.params_df['month'].isin([6, 7, 8])
        params_df_summer = self.params_df[is_summer].groupby('year').mean()

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f"Long-term Trend Analysis (Sen's Slope) - {self.city_name}", 
                    fontsize=18, fontweight='bold', y=0.98)

        # 전체 기간
        self._plot_with_sens_slope(
            axes[0, 0], params_df_yearly['a/b'], 
            'Annual Mean: a/b (Pattern Shape)'
        )
        self._plot_with_sens_slope(
            axes[0, 1], params_df_yearly['log_ab'], 
            'Annual Mean: log(a*b) (Concentration)'
        )

        # 여름철
        self._plot_with_sens_slope(
            axes[1, 0], params_df_summer['a/b'], 
            'Summer(JJA): a/b (Pattern Shape)'
        )
        self._plot_with_sens_slope(
            axes[1, 1], params_df_summer['log_ab'], 
            'Summer(JJA): log(a*b) (Concentration)'
        )

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        plt.show()
        
    def _plot_with_sens_slope(self, ax, series, title, rolling_window=10):
        """Sen's Slope 추세선 플롯"""
        if series.empty:
            return

        try:
            result = mk.original_test(series.dropna())
            slope = result.slope
            p_value = result.p
            
            x_vals = series.index
            y_vals = series.values
            trend_line = slope * (x_vals - x_vals[0]) + (
                np.median(y_vals) - slope * np.median(x_vals - x_vals[0])
            )
            trend_label = f"Sen's Slope: {slope:.4f}/yr\n(p={p_value:.3f})"
        except:
            slope = 0
            trend_line = np.zeros_like(series)
            trend_label = "Trend calc failed"

        # 원본 데이터
        ax.scatter(series.index, series, color='skyblue', alpha=0.6, 
                  s=30, label='Annual Mean')
        
        # 이동 평균
        rolling_mean = series.rolling(window=rolling_window).mean()
        ax.plot(rolling_mean.index, rolling_mean, color='orange', 
               linewidth=2, label=f'{rolling_window}y Moving Avg')
        
        # 추세선
        ax.plot(series.index, trend_line, color='crimson', linestyle='--', 
               linewidth=2.5, label=trend_label)
                
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Year')
        ax.legend(loc='upper left', fontsize='small', frameon=True)
        ax.grid(True, linestyle=':', alpha=0.6)


class ComparisonVisualizer:
    """Huff vs Beta 비교 시각화 클래스"""
    
    def __init__(self, comparison_results, params_df, city_name="Unknown"):
        self.results = comparison_results
        self.params_df = params_df
        self.city_name = city_name
        
    def plot_trend_comparison(self):
        """추세 탐지 비교 시각화"""
        yearly_data = self.results['trend_detection']['yearly_data']
        mk_ab = self.results['trend_detection']['beta_ab']
        mk_logab = self.results['trend_detection']['beta_logab']
        mk_huff = self.results['trend_detection']['huff_q1']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Beta: a/b
        axes[0].scatter(yearly_data['year'], yearly_data['a/b'], 
                       alpha=0.6, s=50)
        z = np.polyfit(yearly_data['year'], yearly_data['a/b'], 1)
        p = np.poly1d(z)
        axes[0].plot(yearly_data['year'], p(yearly_data['year']), "r--", 
                    linewidth=2,
                    label=f'Trend: {mk_ab.slope:.4f}/yr (p={mk_ab.p:.3f})')
        axes[0].set_title('Beta: a/b 추세', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('a/b (시간 패턴)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Beta: log(a*b)
        axes[1].scatter(yearly_data['year'], yearly_data['log_ab'], 
                       alpha=0.6, s=50, color='orange')
        z = np.polyfit(yearly_data['year'], yearly_data['log_ab'], 1)
        p = np.poly1d(z)
        axes[1].plot(yearly_data['year'], p(yearly_data['year']), "r--", 
                    linewidth=2,
                    label=f'Trend: {mk_logab.slope:.4f}/yr (p={mk_logab.p:.3f})')
        axes[1].set_title('Beta: log(a*b) 추세', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('log(a*b) (집중도)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Huff: Q1 비율
        axes[2].scatter(yearly_data['year'], yearly_data['huff_quartile'], 
                       alpha=0.6, s=50, color='green')
        z = np.polyfit(yearly_data['year'], yearly_data['huff_quartile'], 1)
        p = np.poly1d(z)
        axes[2].plot(yearly_data['year'], p(yearly_data['year']), "r--", 
                    linewidth=2,
                    label=f'Trend: {mk_huff.slope:.4f}/yr (p={mk_huff.p:.3f})')
        axes[2].set_title('Huff: Quartile 1 비율 추세', fontsize=14, 
                         fontweight='bold')
        axes[2].set_xlabel('Year')
        axes[2].set_ylabel('Q1 Proportion')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Trend Detection Comparison - {self.city_name}', 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
        
    def plot_extreme_events(self):
        """극한 이벤트 비교 시각화"""
        threshold_90 = self.results['extreme_events']['threshold_90']
        extreme_beta = self.results['extreme_events']['extreme_beta']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # (1) 극한 이벤트 시간 분포
        year_min = extreme_beta['year'].min()
        year_max = extreme_beta['year'].max()
        year_range = year_max - year_min
        
        if year_range <= 10:
            bins = [year_min - 1, year_max + 1]
            labels = [f'{year_min}-{year_max}']
        elif year_range <= 20:
            mid_year = year_min + year_range // 2
            bins = [year_min - 1, mid_year, year_max + 1]
            labels = [f'{year_min}-{mid_year}', f'{mid_year+1}-{year_max}']
        else:
            third = year_range // 3
            bins = [year_min - 1, year_min + third, 
                   year_min + 2*third, year_max + 1]
            labels = [f'{year_min}-{year_min+third}', 
                     f'{year_min+third+1}-{year_min+2*third}',
                     f'{year_min+2*third+1}-{year_max}']
        
        extreme_beta_copy = extreme_beta.copy()
        extreme_beta_copy['period'] = pd.cut(extreme_beta_copy['year'], 
                                             bins=bins, labels=labels)
        period_counts = extreme_beta_copy['period'].value_counts().sort_index()
        
        axes[0, 0].bar(range(len(period_counts)), period_counts.values, 
                      color='crimson', alpha=0.7)
        axes[0, 0].set_xticks(range(len(period_counts)))
        axes[0, 0].set_xticklabels(period_counts.index, rotation=15, ha='right')
        axes[0, 0].set_title('Beta: 극한 집중도 이벤트의 시간 분포', 
                            fontsize=13, fontweight='bold')
        axes[0, 0].set_ylabel('Event Count')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # (2) log(a*b) vs 총 강수량
        axes[0, 1].scatter(self.params_df['total_precip_mm'], 
                          self.params_df['log_ab'], 
                          alpha=0.5, s=30, c='skyblue', 
                          edgecolors='black', linewidth=0.5)
        axes[0, 1].scatter(extreme_beta['total_precip_mm'], 
                          extreme_beta['log_ab'],
                          alpha=0.8, s=60, c='red', 
                          edgecolors='black', linewidth=1, 
                          label='Extreme (top 10%)')
        axes[0, 1].axhline(threshold_90, color='red', linestyle='--', 
                          linewidth=2, label=f'90th percentile')
        axes[0, 1].set_xlabel('Total Precipitation (mm)', fontsize=11)
        axes[0, 1].set_ylabel('log(a*b) - 집중도', fontsize=11)
        axes[0, 1].set_title('Beta: 집중도 vs 총강수량', 
                            fontsize=13, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # (3) Huff Quartile별 강수량 분포
        huff_data = [
            self.params_df[self.params_df['huff_quartile'] == q]['total_precip_mm'] 
            for q in range(1, 5)
        ]
        bp = axes[1, 0].boxplot(huff_data, labels=['Q1', 'Q2', 'Q3', 'Q4'], 
                                patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightgreen')
            patch.set_alpha(0.6)
        axes[1, 0].set_title('Huff: Quartile별 강수량 분포', 
                            fontsize=13, fontweight='bold')
        axes[1, 0].set_ylabel('Total Precipitation (mm)')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # (4) a/b vs log(a*b) scatter
        axes[1, 1].scatter(self.params_df['a/b'], self.params_df['log_ab'],
                          alpha=0.4, s=30, c='gray')
        axes[1, 1].scatter(extreme_beta['a/b'], extreme_beta['log_ab'],
                          alpha=0.8, s=60, c='red', 
                          edgecolors='black', linewidth=1)
        axes[1, 1].set_xlabel('a/b (시간 패턴)', fontsize=11)
        axes[1, 1].set_ylabel('log(a*b) (집중도)', fontsize=11)
        axes[1, 1].set_title('Beta: 2D 매개변수 공간에서 극한 이벤트 위치', 
                            fontsize=13, fontweight='bold')
        axes[1, 1].axhline(threshold_90, color='red', linestyle='--', 
                          linewidth=1.5, alpha=0.7)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Extreme Event Identification - {self.city_name}', 
                    fontsize=16, y=1.0)
        plt.tight_layout()
        plt.show()
        
    def plot_ml_comparison(self):
        """ML 분류 성능 비교 시각화"""
        score_beta = self.results['ml_application']['score_beta']
        score_huff = self.results['ml_application']['score_huff']
        feature_importance = self.results['ml_application']['feature_importance']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Feature Importance
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
        
        axes[0].barh(features, importances, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Importance', fontsize=11)
        axes[0].set_title('Beta 방법: Feature Importance', 
                         fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # 분류 정확도 비교
        methods = ['Beta\n(a/b, log(a*b))', 'Huff\n(Quartile)']
        scores = [score_beta, score_huff]
        colors_bar = ['steelblue', 'lightcoral']
        
        bars = axes[1].bar(methods, scores, color=colors_bar, alpha=0.7, 
                          edgecolor='black', linewidth=2)
        axes[1].set_ylabel('Accuracy', fontsize=11)
        axes[1].set_ylim([0, 1])
        axes[1].set_title('강수 유형 분류 정확도 비교', 
                         fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.3f}',
                        ha='center', va='bottom', fontsize=12, 
                        fontweight='bold')
        
        plt.suptitle(f'ML Application Comparison - {self.city_name}', 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()


class TrendVisualizer:
    """추세 분석 전용 시각화 클래스"""
    
    @staticmethod
    def plot_with_sens_slope(ax, series, title, rolling_window=10):
        """Sen's Slope 추세선 플롯 (정적 메서드)"""
        if series.empty:
            return

        try:
            result = mk.original_test(series.dropna())
            slope = result.slope
            p_value = result.p
            
            x_vals = series.index
            y_vals = series.values
            trend_line = slope * (x_vals - x_vals[0]) + (
                np.median(y_vals) - slope * np.median(x_vals - x_vals[0])
            )
            trend_label = f"Sen's Slope: {slope:.4f}/yr\n(p={p_value:.3f})"
        except:
            slope = 0
            trend_line = np.zeros_like(series)
            trend_label = "Trend calc failed"

        ax.scatter(series.index, series, color='skyblue', alpha=0.6, 
                  s=30, label='Annual Mean')
        
        rolling_mean = series.rolling(window=rolling_window).mean()
        ax.plot(rolling_mean.index, rolling_mean, color='orange', 
               linewidth=2, label=f'{rolling_window}y Moving Avg')
        
        ax.plot(series.index, trend_line, color='crimson', linestyle='--', 
               linewidth=2.5, label=trend_label)
                
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Year')
        ax.legend(loc='upper left', fontsize='small', frameon=True)
        ax.grid(True, linestyle=':', alpha=0.6)