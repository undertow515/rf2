"""
Huff Curve vs Beta Distribution 비교 분석
"""

import pandas as pd
import numpy as np
from scipy import stats
import pymannkendall as mk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class HuffBetaComparison:
    """Huff Curve와 Beta Distribution 비교 분석 클래스"""
    
    def __init__(self, params_df, pattern_labels):
        """
        Parameters
        ----------
        params_df : pd.DataFrame
            Beta 분포 매개변수 DataFrame
        pattern_labels : list
            Huff Curve 분류 결과
        """
        self.params_df = params_df.copy()
        self.params_df['huff_label'] = pattern_labels
        self.params_df['huff_quartile'] = (
            self.params_df['huff_label'].str.extract('(\d)').astype(int)
        )
        
        self.results = {}
        
    def run_all_comparisons(self):
        """모든 비교 분석 실행"""
        print("="*80)
        print("Huff Curve vs Beta Distribution 비교 분석 시작")
        print("="*80)
        
        self.compare_trend_detection()
        self.compare_extreme_event_identification()
        self.analyze_physical_interpretation()
        self.compare_seasonal_variation()
        self.demonstrate_ml_application()
        
        print("\n" + "="*80)
        print("비교 분석 완료")
        print("="*80)
        
        return self.results
    
    def compare_trend_detection(self):
        """추세 탐지 민감도 비교"""
        print("\n[1] 추세 탐지 민감도 비교")
        print("-" * 80)
        
        yearly_data = self.params_df.groupby('year').agg({
            'a/b': 'mean',
            'log_ab': 'mean',
            'huff_quartile': lambda x: (x == 1).sum() / len(x)
        }).reset_index()
        
        # Mann-Kendall 검정
        mk_ab = mk.original_test(yearly_data['a/b'])
        mk_logab = mk.original_test(yearly_data['log_ab'])
        mk_huff = mk.original_test(yearly_data['huff_quartile'])
        
        print("\n【Beta 분포 방법】")
        print(f"  a/b 추세:")
        print(f"    - Sen's Slope: {mk_ab.slope:.6f} /year")
        print(f"    - p-value: {mk_ab.p:.4f}")
        print(f"    - Trend: {mk_ab.trend}")
        
        print(f"\n  log(a*b) 추세:")
        print(f"    - Sen's Slope: {mk_logab.slope:.6f} /year")
        print(f"    - p-value: {mk_logab.p:.4f}")
        print(f"    - Trend: {mk_logab.trend}")
        
        print("\n【Huff Curve 방법】")
        print(f"  Q1 비율 추세:")
        print(f"    - Sen's Slope: {mk_huff.slope:.6f} /year")
        print(f"    - p-value: {mk_huff.p:.4f}")
        print(f"    - Trend: {mk_huff.trend}")
        
        self.results['trend_detection'] = {
            'yearly_data': yearly_data,
            'beta_ab': mk_ab,
            'beta_logab': mk_logab,
            'huff_q1': mk_huff
        }
        
    def compare_extreme_event_identification(self):
        """극한 이벤트 식별 능력 비교"""
        print("\n[2] 극한 이벤트 식별 능력 비교")
        print("-" * 80)
        
        threshold_90 = self.params_df['log_ab'].quantile(0.9)
        extreme_beta = self.params_df[self.params_df['log_ab'] > threshold_90].copy()
        
        print(f"\n【Beta 방법】")
        print(f"  극한 이벤트 정의: log(a*b) > {threshold_90:.3f} (상위 10%)")
        print(f"  극한 이벤트 수: {len(extreme_beta)}")
        print(f"  평균 강수량: {extreme_beta['total_precip_mm'].mean():.2f} mm")
        print(f"  평균 지속시간: {extreme_beta['duration_hours'].mean():.2f} hours")
        
        print(f"\n【Huff 방법】")
        print("  Quartile별 평균 강수량:")
        for q in range(1, 5):
            q_data = self.params_df[self.params_df['huff_quartile'] == q]
            print(f"    Q{q}: {q_data['total_precip_mm'].mean():.2f} mm (n={len(q_data)})")
        
        self.results['extreme_events'] = {
            'threshold_90': threshold_90,
            'extreme_beta': extreme_beta
        }
        
    def analyze_physical_interpretation(self):
        """물리적 해석력 분석"""
        print("\n[3] 물리적 해석력 분석")
        print("-" * 80)
        
        if 'intensity' not in self.params_df.columns:
            self.params_df['intensity'] = (
                self.params_df['total_precip_mm'] / self.params_df['duration_hours']
            )
        
        # 상관분석
        corr_ab_intensity = stats.spearmanr(
            self.params_df['a/b'], 
            self.params_df['intensity']
        )
        corr_logab_total = stats.spearmanr(
            self.params_df['log_ab'], 
            self.params_df['total_precip_mm']
        )
        corr_logab_intensity = stats.spearmanr(
            self.params_df['log_ab'], 
            self.params_df['intensity']
        )
        
        print("\n【Beta 매개변수와 물리량의 상관관계】")
        print(f"  a/b vs 강수강도: r={corr_ab_intensity.correlation:.3f}, "
              f"p={corr_ab_intensity.pvalue:.4f}")
        print(f"  log(a*b) vs 총강수량: r={corr_logab_total.correlation:.3f}, "
              f"p={corr_logab_total.pvalue:.4f}")
        print(f"  log(a*b) vs 강수강도: r={corr_logab_intensity.correlation:.3f}, "
              f"p={corr_logab_intensity.pvalue:.4f}")
        
        # 강수 유형 분류
        intensity_threshold = self.params_df['intensity'].quantile(0.75)
        self.params_df['precip_type'] = np.where(
            self.params_df['intensity'] > intensity_threshold,
            'Convective', 'Frontal'
        )
        
        print(f"\n【강수 유형 분류】 (강도 기준: {intensity_threshold:.2f} mm/hr)")
        convective_count = (self.params_df['precip_type'] == 'Convective').sum()
        frontal_count = (self.params_df['precip_type'] == 'Frontal').sum()
        
        print(f"  Convective (대류성): {convective_count} events")
        print(f"    - 평균 a/b: "
              f"{self.params_df[self.params_df['precip_type'] == 'Convective']['a/b'].mean():.3f}")
        print(f"  Frontal (전선성): {frontal_count} events")
        print(f"    - 평균 a/b: "
              f"{self.params_df[self.params_df['precip_type'] == 'Frontal']['a/b'].mean():.3f}")
        
        # t-test
        convective_ab = self.params_df[
            self.params_df['precip_type'] == 'Convective'
        ]['a/b']
        frontal_ab = self.params_df[
            self.params_df['precip_type'] == 'Frontal'
        ]['a/b']
        t_stat, p_val = stats.ttest_ind(convective_ab, frontal_ab)
        print(f"  → t-test: t={t_stat:.3f}, p={p_val:.4f}")
        
        self.results['physical_interpretation'] = {
            'correlations': {
                'ab_intensity': corr_ab_intensity,
                'logab_total': corr_logab_total,
                'logab_intensity': corr_logab_intensity
            },
            'intensity_threshold': intensity_threshold,
            't_test': (t_stat, p_val)
        }
        
    def compare_seasonal_variation(self):
        """계절별 패턴 변동성 비교"""
        print("\n[4] 계절별 패턴 변동성 비교")
        print("-" * 80)
        
        # 계절 정의
        self.params_df['season'] = pd.cut(
            self.params_df['month'], 
            bins=[0, 2, 5, 8, 11, 12],
            labels=['Winter', 'Spring', 'Summer', 'Fall', 'Winter']
        )
        self.params_df.loc[self.params_df['month'] == 12, 'season'] = 'Winter'
        self.params_df['season'] = self.params_df['season'].cat.remove_unused_categories()
        
        # Beta 방법: ANOVA
        seasons_unique = self.params_df['season'].unique()
        season_groups_ab = [
            self.params_df[self.params_df['season'] == s]['a/b'].dropna() 
            for s in seasons_unique
        ]
        season_groups_logab = [
            self.params_df[self.params_df['season'] == s]['log_ab'].dropna() 
            for s in seasons_unique
        ]
        
        f_stat_ab, p_val_ab = stats.f_oneway(*season_groups_ab)
        f_stat_logab, p_val_logab = stats.f_oneway(*season_groups_logab)
        
        print("\n【Beta 방법: 계절별 ANOVA】")
        print(f"  a/b: F={f_stat_ab:.3f}, p={p_val_ab:.4f}")
        print(f"  log(a*b): F={f_stat_logab:.3f}, p={p_val_logab:.4f}")
        
        # Huff 방법: Chi-square test
        contingency_table = pd.crosstab(
            self.params_df['season'], 
            self.params_df['huff_quartile']
        )
        chi2, p_chi, dof, expected = stats.chi2_contingency(contingency_table)
        
        print("\n【Huff 방법: Chi-square test】")
        print(f"  χ²={chi2:.3f}, p={p_chi:.4f}, dof={dof}")
        
        self.results['seasonal_variation'] = {
            'beta_anova': (f_stat_ab, p_val_ab, f_stat_logab, p_val_logab),
            'huff_chi2': (chi2, p_chi, dof),
            'contingency_table': contingency_table
        }
        
    def demonstrate_ml_application(self):
        """머신러닝 응용 예시"""
        print("\n[5] 머신러닝 응용: 강수 유형 분류")
        print("-" * 80)
        
        if 'precip_type' not in self.params_df.columns:
            self.analyze_physical_interpretation()
        
        # Feature 준비
        X_beta = self.params_df[['a/b', 'log_ab', 'duration_hours']].values
        X_huff = pd.get_dummies(
            self.params_df['huff_quartile'], 
            prefix='Q'
        ).values
        X_huff = np.column_stack([X_huff, self.params_df['duration_hours'].values])
        
        y = (self.params_df['precip_type'] == 'Convective').astype(int)
        
        # Train-test split
        X_beta_train, X_beta_test, y_train, y_test = train_test_split(
            X_beta, y, test_size=0.3, random_state=42
        )
        X_huff_train, X_huff_test, _, _ = train_test_split(
            X_huff, y, test_size=0.3, random_state=42
        )
        
        # Random Forest 학습
        rf_beta = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_huff = RandomForestClassifier(n_estimators=100, random_state=42)
        
        rf_beta.fit(X_beta_train, y_train)
        rf_huff.fit(X_huff_train, y_train)
        
        # 성능 평가
        score_beta = rf_beta.score(X_beta_test, y_test)
        score_huff = rf_huff.score(X_huff_test, y_test)
        
        print("\n【Random Forest 분류 성능】")
        print(f"  Beta 방법 (a/b, log(a*b), duration): {score_beta:.3f}")
        print(f"  Huff 방법 (Q1-Q4 dummy, duration): {score_huff:.3f}")
        
        # Feature importance
        feature_names_beta = ['a/b', 'log(a*b)', 'duration']
        importances_beta = rf_beta.feature_importances_
        
        print("\n【Beta 방법 Feature Importance】")
        for name, imp in zip(feature_names_beta, importances_beta):
            print(f"  {name}: {imp:.3f}")
        
        self.results['ml_application'] = {
            'score_beta': score_beta,
            'score_huff': score_huff,
            'feature_importance': dict(zip(feature_names_beta, importances_beta)),
            'rf_beta': rf_beta,
            'rf_huff': rf_huff
        }