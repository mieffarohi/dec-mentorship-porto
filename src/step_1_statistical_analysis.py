import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional
import warnings

# Try to import validation module
try:
    from validation import ExperimentValidator
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    warnings.warn("Validation module not available. Skipping validation checks.")


class ABTestAnalyzer:
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        if VALIDATION_AVAILABLE:
            self.validator = ExperimentValidator(srm_threshold=0.001)  # Stricter for SRM
        else:
            self.validator = None
    
    def calculate_sample_size(self,
                            baseline_rate: float,
                            mde: float,
                            alpha: float = 0.05,
                            power: float = 0.80,
                            two_tailed: bool = True) -> int:
        
        if two_tailed:
            z_alpha = stats.norm.ppf(1 - alpha/2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        
        z_beta = stats.norm.ppf(power)
  
        p1 = baseline_rate
        p2 = baseline_rate * (1 + mde)
        
        
        p2 = min(p2, 0.999)
        
        numerator = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
        denominator = (p2 - p1) ** 2
        
        n = numerator / denominator
        
        return int(np.ceil(n))
    
    def two_sample_ttest(self,
                        control: np.ndarray,
                        treatment: np.ndarray,
                        metric_name: str,
                        equal_var: bool = False) -> Dict:
        
        control = control[~np.isnan(control)]
        treatment = treatment[~np.isnan(treatment)]
        
        control_mean = control.mean()
        treatment_mean = treatment.mean()
        control_std = control.std(ddof=1)
        treatment_std = treatment.std(ddof=1)
        n_control = len(control)
        n_treatment = len(treatment)
        
        statistic, pvalue = stats.ttest_ind(treatment, control, equal_var=equal_var)
        
        pooled_std = np.sqrt((control_std**2 + treatment_std**2) / 2)
        cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        se_diff = np.sqrt(control_std**2/n_control + treatment_std**2/n_treatment)
        
        if not equal_var:
            num = (control_std**2/n_control + treatment_std**2/n_treatment)**2
            denom = ((control_std**2/n_control)**2/(n_control-1) + 
                    (treatment_std**2/n_treatment)**2/(n_treatment-1))
            df = num / denom if denom > 0 else n_control + n_treatment - 2
        else:
            df = n_control + n_treatment - 2
        
        t_crit = stats.t.ppf(1 - self.alpha/2, df)
        diff = treatment_mean - control_mean
        ci_lower = diff - t_crit * se_diff
        ci_upper = diff + t_crit * se_diff
        
        relative_lift_pct = (diff / control_mean * 100) if control_mean != 0 else 0
        
        return {
            'metric': metric_name,
            'test_type': 't-test',
            'statistic': statistic,
            'pvalue': pvalue,
            'significant': pvalue < self.alpha,
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'control_std': control_std,
            'treatment_std': treatment_std,
            'absolute_diff': diff,
            'relative_lift_pct': relative_lift_pct,
            'cohens_d': cohens_d,
            'effect_interpretation': self._interpret_cohens_d(cohens_d),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_control': n_control,
            'n_treatment': n_treatment,
            'degrees_of_freedom': df
        }
    
    def proportion_test(self,
                       control_successes: int,
                       control_total: int,
                       treatment_successes: int,
                       treatment_total: int,
                       metric_name: str) -> Dict:
        
        p_control = control_successes / control_total
        p_treatment = treatment_successes / treatment_total
        
        p_pooled = (control_successes + treatment_successes) / (control_total + treatment_total)
        
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))
        
        z_stat = (p_treatment - p_control) / se if se > 0 else 0
        
        pvalue = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        se_diff = np.sqrt(p_control*(1-p_control)/control_total + 
                         p_treatment*(1-p_treatment)/treatment_total)
        z_crit = stats.norm.ppf(1 - self.alpha/2)
        diff = p_treatment - p_control
        ci_lower = diff - z_crit * se_diff
        ci_upper = diff + z_crit * se_diff
        
        relative_lift_pct = (diff / p_control * 100) if p_control > 0 else 0
        
        return {
            'metric': metric_name,
            'test_type': 'proportion_test',
            'statistic': z_stat,
            'pvalue': pvalue,
            'significant': pvalue < self.alpha,
            'control_rate': p_control,
            'treatment_rate': p_treatment,
            'absolute_diff': diff,
            'relative_lift_pct': relative_lift_pct,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_control': control_total,
            'n_treatment': treatment_total
        }