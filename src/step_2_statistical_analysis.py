def chi_square_test(self,
                       control: np.ndarray,
                       treatment: np.ndarray,
                       metric_name: str) -> Dict:
        
        combined = np.concatenate([control, treatment])
        labels = np.concatenate([np.zeros(len(control)), np.ones(len(treatment))])
        
        contingency_table = pd.crosstab(combined, labels)
        
        chi2, pvalue, dof, expected = chi2_contingency(contingency_table)

        n = len(combined)
        min_dim = min(contingency_table.shape[0], contingency_table.shape[1]) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        return {
            'metric': metric_name,
            'test_type': 'chi_square',
            'statistic': chi2,
            'pvalue': pvalue,
            'significant': pvalue < self.alpha,
            'degrees_of_freedom': dof,
            'cramers_v': cramers_v,
            'effect_interpretation': self._interpret_cramers_v(cramers_v),
            'n_control': len(control),
            'n_treatment': len(treatment)
        }
    
    def mann_whitney_u_test(self,
                           control: np.ndarray,
                           treatment: np.ndarray,
                           metric_name: str) -> Dict:

        

        control = control[~np.isnan(control)]
        treatment = treatment[~np.isnan(treatment)]
        

        statistic, pvalue = stats.mannwhitneyu(treatment, control, alternative='two-sided')
        

        n1 = len(control)
        n2 = len(treatment)
        rank_biserial = 1 - (2*statistic) / (n1 * n2)
        

        control_median = np.median(control)
        treatment_median = np.median(treatment)
        
        return {
            'metric': metric_name,
            'test_type': 'mann_whitney',
            'statistic': statistic,
            'pvalue': pvalue,
            'significant': pvalue < self.alpha,
            'control_median': control_median,
            'treatment_median': treatment_median,
            'rank_biserial': rank_biserial,
            'n_control': n1,
            'n_treatment': n2
        }
    
    def bootstrap_confidence_interval(self,
                                     control: np.ndarray,
                                     treatment: np.ndarray,
                                     metric_name: str,
                                     n_bootstrap: int = 10000,
                                     confidence_level: float = 0.95) -> Dict:
        
        np.random.seed(42)
        

        control = control[~np.isnan(control)]
        treatment = treatment[~np.isnan(treatment)]
        
 
        boot_diffs = []
        for _ in range(n_bootstrap):
            control_boot = np.random.choice(control, size=len(control), replace=True)
            treatment_boot = np.random.choice(treatment, size=len(treatment), replace=True)
            boot_diffs.append(treatment_boot.mean() - control_boot.mean())
        
        boot_diffs = np.array(boot_diffs)
        

        alpha_bootstrap = 1 - confidence_level
        ci_lower = np.percentile(boot_diffs, alpha_bootstrap/2 * 100)
        ci_upper = np.percentile(boot_diffs, (1 - alpha_bootstrap/2) * 100)
        
 
        observed_diff = treatment.mean() - control.mean()

        significant = not (ci_lower <= 0 <= ci_upper)
        
        return {
            'metric': metric_name,
            'test_type': 'bootstrap',
            'observed_diff': observed_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': significant,
            'confidence_level': confidence_level,
            'n_bootstrap': n_bootstrap
        }
    
    def multiple_testing_correction(self,
                                   p_values: List[float],
                                   method: str = 'holm') -> Dict:


        reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(
            p_values, 
            alpha=self.alpha, 
            method=method
        )
        
        fwer_uncorrected = 1 - (1 - self.alpha) ** len(p_values)
        
        return {
            'method': method,
            'original_pvalues': p_values,
            'corrected_pvalues': pvals_corrected.tolist(),
            'reject': reject.tolist(),
            'fwer_uncorrected': fwer_uncorrected,
            'num_tests': len(p_values),
            'num_significant_uncorrected': sum(p < self.alpha for p in p_values),
            'num_significant_corrected': sum(reject)
        }