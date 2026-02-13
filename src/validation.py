"""
Complete Experimental Validation Framework for A/B Testing

Implements ALL required validation checks:
1. Sample Ratio Mismatch (SRM) Detection 
2. Covariate Balance Verification 
3. Temporal Stability Checks 
4. Multiple Testing Correction

When run directly, validates all 5 A/B tests with comprehensive reporting.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional, Union
import warnings
from datetime import datetime
import os


class ExperimentValidator:
    """
    Complete validation framework for A/B tests.
    """
    
    def __init__(self, 
                 srm_threshold: float = 0.001,
                 balance_threshold: float = 0.2,
                 temporal_threshold: float = 0.2):
        self.srm_threshold = srm_threshold
        self.balance_threshold = balance_threshold
        self.temporal_threshold = temporal_threshold
    
    def sample_ratio_mismatch_test(self,
                                   df: pd.DataFrame,
                                   variant_col: str,
                                   expected_ratio: Optional[Dict[str, float]] = None) -> Dict:
        """Sample Ratio Mismatch detection."""
        
        observed = df[variant_col].value_counts().sort_index()
        total = len(df)
        n_variants = len(observed)
        
        if expected_ratio is None:
            expected = pd.Series([total / n_variants] * n_variants, index=observed.index)
        else:
            expected = pd.Series({k: v * total for k, v in expected_ratio.items()})
        
        chi2_stat = np.sum((observed - expected)**2 / expected)
        df_chi = n_variants - 1
        pvalue = 1 - stats.chi2.cdf(chi2_stat, df_chi)
        
        has_srm = pvalue < self.srm_threshold
        
        result = {
            'test': 'sample_ratio_mismatch',
            'chi2_statistic': chi2_stat,
            'degrees_of_freedom': df_chi,
            'pvalue': pvalue,
            'threshold': self.srm_threshold,
            'has_srm': has_srm,
            'observed_counts': observed.to_dict(),
            'expected_counts': expected.to_dict(),
            'observed_ratio': (observed / total).to_dict(),
            'expected_ratio': (expected / total).to_dict()
        }
        
        if has_srm:
            result['warning'] = f"CRITICAL: SRM detected (p={pvalue:.6f} < {self.srm_threshold}). Experiment is INVALID."
        else:
            result['message'] = f"No SRM detected (p={pvalue:.4f}). Allocation is as expected."
        
        return result
    
    def covariate_balance_check(self,
                                df: pd.DataFrame,
                                variant_col: str,
                                covariates: List[str],
                                threshold: Optional[float] = None) -> Dict:
        """Covariate balance verification using SMD."""
        
        if threshold is None:
            threshold = self.balance_threshold
        
        variants = df[variant_col].unique()
        
        if len(variants) < 2:
            return {'error': 'Need at least 2 variants for balance check'}
        
        balance_results = []
        imbalanced_covariates = []
        
        for covariate in covariates:
            if covariate not in df.columns:
                warnings.warn(f"Covariate '{covariate}' not found in dataframe")
                continue
            
            is_categorical = (
                df[covariate].dtype == 'object' or 
                df[covariate].dtype.name == 'category' or
                df[covariate].nunique() < 10
            )
            
            if is_categorical:
                for category in df[covariate].unique():
                    proportions = {}
                    for variant in variants:
                        variant_data = df[df[variant_col] == variant][covariate]
                        proportions[variant] = (variant_data == category).mean()
                    
                    variant_list = list(variants)
                    p1 = proportions[variant_list[0]]
                    p2 = proportions[variant_list[1]]
                    p_pooled = (p1 + p2) / 2
                    
                    if p_pooled > 0 and p_pooled < 1:
                        smd = abs(p1 - p2) / np.sqrt(p_pooled * (1 - p_pooled))
                    else:
                        smd = 0.0
                    
                    is_imbalanced = smd > threshold
                    
                    balance_results.append({
                        'covariate': f"{covariate}={category}",
                        'type': 'categorical',
                        'variant_1': variant_list[0],
                        'variant_2': variant_list[1],
                        'proportion_1': p1,
                        'proportion_2': p2,
                        'smd': smd,
                        'imbalanced': is_imbalanced
                    })
                    
                    if is_imbalanced:
                        imbalanced_covariates.append(f"{covariate}={category}")
            else:
                variant_stats = {}
                for variant in variants:
                    variant_data = df[df[variant_col] == variant][covariate]
                    variant_stats[variant] = {
                        'mean': variant_data.mean(),
                        'std': variant_data.std(),
                        'var': variant_data.var(),
                        'n': len(variant_data)
                    }
                
                variant_list = list(variants)
                v1, v2 = variant_list[0], variant_list[1]
                
                mean_diff = abs(variant_stats[v1]['mean'] - variant_stats[v2]['mean'])
                pooled_std = np.sqrt((variant_stats[v1]['var'] + variant_stats[v2]['var']) / 2)
                
                if pooled_std > 0:
                    smd = mean_diff / pooled_std
                else:
                    smd = 0.0
                
                is_imbalanced = smd > threshold
                
                balance_results.append({
                    'covariate': covariate,
                    'type': 'continuous',
                    'variant_1': variant_list[0],
                    'variant_2': variant_list[1],
                    'mean_1': variant_stats[v1]['mean'],
                    'mean_2': variant_stats[v2]['mean'],
                    'std_1': variant_stats[v1]['std'],
                    'std_2': variant_stats[v2]['std'],
                    'smd': smd,
                    'imbalanced': is_imbalanced
                })
                
                if is_imbalanced:
                    imbalanced_covariates.append(covariate)
        
        balance_df = pd.DataFrame(balance_results)
        max_smd = balance_df['smd'].max() if len(balance_df) > 0 else 0
        n_imbalanced = len(imbalanced_covariates)
        
        if max_smd < 0.1:
            message = f"Excellent balance (max SMD={max_smd:.3f} < 0.1)"
        elif max_smd < threshold:
            message = f"Good balance (max SMD={max_smd:.3f} < {threshold})"
        else:
            message = f"{n_imbalanced} covariate(s) imbalanced (max SMD={max_smd:.3f} ≥ {threshold})"
        
        return {
            'test': 'covariate_balance',
            'variants_compared': list(variants)[:2],
            'balance_results': balance_df,
            'imbalanced_covariates': imbalanced_covariates,
            'n_imbalanced': n_imbalanced,
            'max_smd': max_smd,
            'threshold': threshold,
            'message': message
        }
    
    def temporal_stability_check(self,
                                df: pd.DataFrame,
                                variant_col: str,
                                date_col: str,
                                threshold: Optional[float] = None) -> Dict:
        """Temporal stability verification."""
        
        if threshold is None:
            threshold = self.temporal_threshold
        
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        
        df['date'] = df[date_col].dt.date
        daily_counts = df.groupby(['date', variant_col]).size().unstack(fill_value=0)
        
        cv_results = {}
        for variant in daily_counts.columns:
            counts = daily_counts[variant]
            mean_count = counts.mean()
            std_count = counts.std()
            cv = std_count / mean_count if mean_count > 0 else 0.0
            cv_results[variant] = cv
        
        max_cv = max(cv_results.values())
        is_stable = max_cv < threshold
        
        message = (
            f"Stable allocation over time (max CV={max_cv:.3f} < {threshold})" if is_stable
            else f"Unstable allocation (max CV={max_cv:.3f} ≥ {threshold})"
        )
        
        return {
            'test': 'temporal_stability',
            'cv_by_variant': cv_results,
            'max_cv': max_cv,
            'threshold': threshold,
            'is_stable': is_stable,
            'daily_counts': daily_counts,
            'n_days': len(daily_counts),
            'message': message
        }
    
    def multiple_testing_correction(self,
                                    pvalues: List[float],
                                    method: str = 'holm',
                                    alpha: float = 0.05) -> Dict:
        """
        Multiple testing correction.
        
        Methods:
        - 'bonferroni': Most conservative (alpha/k)
        - 'holm': Holm-Bonferroni (recommended for 5-10 tests)
        - 'fdr_bh': Benjamini-Hochberg FDR (for >10 tests)
        
        References:
        - Bonferroni (1936)
        - Holm (1979)
        - Benjamini & Hochberg (1995)
        """
        
        pvalues_array = np.array(pvalues)
        n_tests = len(pvalues_array)
        
        # Apply correction
        reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(
            pvalues_array,
            alpha=alpha,
            method=method
        )
        
        method_names = {
            'bonferroni': 'Bonferroni',
            'holm': 'Holm-Bonferroni',
            'fdr_bh': 'Benjamini-Hochberg FDR'
        }
        
        return {
            'test': 'multiple_testing_correction',
            'method': method_names.get(method, method),
            'n_tests': n_tests,
            'alpha': alpha,
            'original_pvalues': pvalues_array.tolist(),
            'corrected_pvalues': pvals_corrected.tolist(),
            'reject': reject.tolist(),
            'n_significant_original': sum(pvalues_array < alpha),
            'n_significant_corrected': sum(reject),
            'message': (
                f"✓ Multiple testing correction applied: {method_names.get(method, method)}\n"
                f"  Original significant: {sum(pvalues_array < alpha)}/{n_tests}\n"
                f"  Corrected significant: {sum(reject)}/{n_tests}"
            )
        }
    
    def run_all_validations(self,
                           df: pd.DataFrame,
                           variant_col: str,
                           covariates: Optional[List[str]] = None,
                           date_col: Optional[str] = None,
                           metric_pvalues: Optional[List[float]] = None,
                           correction_method: str = 'holm') -> Dict:
        """
        Run complete validation suite including multiple testing correction.
        """
        
        results = {}
        
        print("=" * 80)
        print("EXPERIMENTAL VALIDATION SUITE")
        print("=" * 80)
        
        # 1. SRM Test
        print("\n1. Sample Ratio Mismatch Test")
        print("-" * 80)
        srm_result = self.sample_ratio_mismatch_test(df, variant_col)
        results['srm'] = srm_result
        print(srm_result.get('message', srm_result.get('warning', '')))
        
        if srm_result['has_srm']:
            print("\n" + "=" * 80)
            print("VALIDATION FAILED: SRM detected.")
            print("=" * 80)
            return results
        
        # 2. Covariate Balance
        if covariates:
            print("\n2. Covariate Balance Check")
            print("-" * 80)
            balance_result = self.covariate_balance_check(df, variant_col, covariates)
            results['balance'] = balance_result
            print(balance_result.get('message', ''))
        
        # 3. Temporal Stability
        if date_col:
            print("\n3. Temporal Stability Check")
            print("-" * 80)
            temporal_result = self.temporal_stability_check(df, variant_col, date_col)
            results['temporal'] = temporal_result
            print(temporal_result.get('message', ''))
        
        # 4. Multiple Testing Correction
        if metric_pvalues:
            print("\n4. Multiple Testing Correction")
            print("-" * 80)
            correction_result = self.multiple_testing_correction(
                metric_pvalues,
                method=correction_method,
                alpha=0.05
            )
            results['multiple_testing'] = correction_result
            print(correction_result.get('message', ''))
        
        # Summary
        print("\n" + "=" * 80)
        all_clear = (
            not srm_result['has_srm'] and
            (not covariates or balance_result.get('max_smd', 0) < self.balance_threshold) and
            (not date_col or temporal_result.get('is_stable', True))
        )
        
        if all_clear:
            print("ALL VALIDATION CHECKS PASSED")
        else:
            print("VALIDATION WARNINGS DETECTED")
        print("=" * 80)
        
        return results


# ============================================================================
# COMPREHENSIVE VALIDATION FOR ALL 5 A/B TESTS
# ============================================================================

def validate_test(test_name, csv_file, validator):
    """Validate a single test"""
    
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print('='*80)
    
    try:
        if os.path.exists(f'../src/{csv_file}'):
            df = pd.read_csv(f'../src/{csv_file}')
        elif os.path.exists(f'src/{csv_file}'):
            df = pd.read_csv(f'src/{csv_file}')
        else:
            raise FileNotFoundError(f"Cannot find {csv_file}")
    except FileNotFoundError:
        print(f"File not found: {csv_file}")
        print("   Please run data_generation.py first!")
        return None
    
    print(f"Loaded: {len(df):,} rows")
    
    # Variant split
    variant_counts = df['variant'].value_counts()
    print(f"Variants ({len(variant_counts)}):")
    for variant, count in variant_counts.items():
        pct = count / len(df) * 100
        print(f"   - {variant}: {count:,} ({pct:.1f}%)")
    
    # Quick validation
    srm = validator.sample_ratio_mismatch_test(df, 'variant')
    balance = validator.covariate_balance_check(
        df, 'variant', ['device_type', 'browser', 'region']
    )
    temporal = validator.temporal_stability_check(df, 'variant', 'timestamp')
    
    # Status
    srm_status = "PASS" if not srm['has_srm'] else "FAIL"
    balance_status = "OK" if balance['max_smd'] < 0.1 else "WARNING" if balance['max_smd'] < 0.2 else "FAIL"
    temporal_status = "OK" if temporal['is_stable'] else "WARNING"
    
    print(f"\nValidation Results:")
    print(f"  SRM Test:        {srm_status} (p={srm['pvalue']:.4f})")
    print(f"  Balance:         {balance_status} (SMD={balance['max_smd']:.3f})")
    print(f"  Temporal:        {temporal_status} (CV={temporal['max_cv']:.3f})")
    
    return {
        'test': test_name,
        'n': len(df),
        'n_variants': len(variant_counts),
        'srm_pvalue': srm['pvalue'],
        'srm_passed': not srm['has_srm'],
        'balance_smd': balance['max_smd'],
        'balance_ok': balance['max_smd'] < 0.2,
        'temporal_cv': temporal['max_cv'],
        'temporal_stable': temporal['is_stable'],
        'overall_valid': not srm['has_srm'] and balance['max_smd'] < 0.2
    }


def validate_all_tests():
    """Run comprehensive validation on all 5 A/B tests"""
    
    print("="*80)
    print("COMPREHENSIVE VALIDATION SUITE")
    print("Validating All 5 A/B Tests")
    print("="*80)
    
    validator = ExperimentValidator(
        srm_threshold=0.001,
        balance_threshold=0.2,
        temporal_threshold=0.2
    )
    
    tests = [
        ('Test 1: Menu Design', 'test1_menu.csv'),
        ('Test 2: Novelty Slider', 'test2_novelty_slider.csv'),
        ('Test 3: Product Sliders', 'test3_product_sliders.csv'),
        ('Test 4: Customer Reviews', 'test4_reviews.csv'),
        ('Test 5: Search Engine', 'test5_search_engine.csv')
    ]
    
    results = []
    for test_name, csv_file in tests:
        result = validate_test(test_name, csv_file, validator)
        if result:
            results.append(result)
    
    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY TABLE")
    print('='*80)
    
    if results:
        summary_df = pd.DataFrame(results)
        
        print(f"\n{'Test':<30} {'N':>8} {'SRM':>8} {'Balance':>10} {'Temporal':>10} {'Valid':>8}")
        print('-'*80)
        
        for _, row in summary_df.iterrows():
            test = row['test'][:28]
            n = f"{int(row['n']):,}"
            srm = "PASS" if row['srm_passed'] else "FAIL"
            balance = "Good" if row['balance_ok'] else "Warning"
            temporal = "Stable" if row['temporal_stable'] else "Unstable"
            valid = "YES" if row['overall_valid'] else "CHECK"
            
            print(f"{test:<30} {n:>8} {srm:>8} {balance:>10} {temporal:>10} {valid:>8}")
        
        # Overall stats
        print('\n' + '='*80)
        print("OVERALL STATISTICS")
        print('='*80)
        
        n_total = summary_df['n'].sum()
        n_passed_srm = summary_df['srm_passed'].sum()
        n_valid = summary_df['overall_valid'].sum()
        
        print(f"\nTotal samples across all tests: {n_total:,}")
        print(f"Tests passed SRM check: {n_passed_srm}/{len(results)}")
        print(f"Tests fully valid: {n_valid}/{len(results)}")
        
        if n_passed_srm == len(results) and n_valid == len(results):
            print("\n ALL TESTS ARE VALID")
            print("\nAll experiments passed validation checks!")
            print("You can proceed with statistical analysis with full confidence.")
        elif n_passed_srm < len(results):
            print("\n CRITICAL ISSUES DETECTED")
            print("\nSome tests failed SRM check - DO NOT analyze those tests!")
        else:
            print("\nMINOR WARNINGS DETECTED")
            print("\nTests passed critical checks but have minor balance/temporal issues.")
            print("Proceed with caution and consider causal adjustment methods.")
        
        # Detailed recommendations
        print("\n" + "="*80)
        print("RECOMMENDATIONS BY TEST")
        print("="*80)
        
        for _, row in summary_df.iterrows():
            print(f"\n{row['test']}:")
            if row['overall_valid']:
                print("All checks passed - proceed with analysis")
            else:
                if not row['srm_passed']:
                    print("    SRM FAILED - DO NOT ANALYZE")
                    print("     → Investigate randomization bug")
                    print("     → Restart experiment after fix")
                elif not row['balance_ok']:
                    print("    Balance issue detected")
                    print("     → Use regression adjustment or CUPED")
                    print("     → Check for selection bias")
                if not row['temporal_stable']:
                    print("    Temporal instability detected")
                    print("     → Check for system changes during test")
                    print("     → Consider excluding unstable periods")
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80 + "\n")



# MAIN EXECUTION
if __name__ == "__main__":
    validate_all_tests()
