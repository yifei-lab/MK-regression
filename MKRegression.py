import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import optimize, stats
from collections import OrderedDict
import argparse
import math


# likelihood function for MK test
class SimpleMK(object):
    def __init__(self,
                 neutral_div,
                 neutral_poly,
                 foreground_div,
                 foreground_poly):
        self.foreground_div = tf.convert_to_tensor(foreground_div, dtype='float64')
        self.foreground_poly = tf.convert_to_tensor(foreground_poly, dtype='float64')
        
        # neutral data
        self.neutral_div = tf.convert_to_tensor(neutral_div, dtype='float64')
        self.neutral_poly = tf.convert_to_tensor(neutral_poly, dtype='float64')
        
    @staticmethod
    def jukes_cantor(rate, obs):
        lambda_time = rate / 3
        
        # jitter term for numerical stability
        # this should be as small as possible
        lambda_time = lambda_time + 1e-12
        
        prob = 3./4. - 3./4. * tf.exp(-4. * lambda_time)
        nll = SimpleMK.cross_entropy(prob, obs)
        return nll
    
    @staticmethod
    def cross_entropy(prob, obs):
        return -tf.reduce_sum(obs * tf.math.log(prob) + (1. - obs) * tf.math.log(1. - prob))

    @tf.function
    def negative_log_likelihood(self,
                                neutral_div_rate,
                                neutral_poly_logit,
                                omega_alpha,
                                foreground_poly_logit):
        # neutral divergence likelihood
        neutral_div_nll = SimpleMK.jukes_cantor(neutral_div_rate, self.neutral_div)
        
        # neutral polymorphism likelihood
        neutral_poly_nll = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(self.neutral_poly,
                                                                                 tf.fill(self.neutral_poly.shape,
                                                                                         neutral_poly_logit)))
        
        # foreground polymorphism likelihood
        foreground_poly_nll = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(self.foreground_poly,
                                                                                    foreground_poly_logit))
        
        # foreground divergence likelihood
        foreground_div_rate = (neutral_div_rate * omega_alpha
                               + neutral_div_rate * tf.math.sigmoid(foreground_poly_logit)
                               / tf.math.sigmoid(neutral_poly_logit))
        
        foreground_div_nll = SimpleMK.jukes_cantor(foreground_div_rate, self.foreground_div)
         
        ### debug mode
        # tf.debugging.check_numerics(neutral_div_nll, 'neutral div')
        # tf.debugging.check_numerics(neutral_poly_nll, 'neutral poly')
        # tf.debugging.check_numerics(foreground_poly_nll, 'foreground poly')
        # tf.debugging.check_numerics(foreground_div_nll, 'foreground div')

        return (neutral_div_nll
                + neutral_poly_nll
                + foreground_div_nll
                + foreground_poly_nll)


class SimpleMKRegression(object):
    def __init__(self,
                 neutral_div,
                 neutral_poly,
                 foreground_div,
                 foreground_poly,
                 feature,):
        self.mk_model = SimpleMK(neutral_div,
                                 neutral_poly,
                                 foreground_div,
                                 foreground_poly)
        assert neutral_div.shape == neutral_poly.shape, 'Unmatched shape for neutral data'
        assert foreground_div.shape == foreground_poly.shape, 'Unmatched shape for foreground data'
        assert foreground_div.shape[0] == feature.shape[0], 'Unmatched shape for features'
        
        feature = np.concatenate([np.ones(shape=(feature.shape[0], 1), dtype='float64'),
                                  feature], axis=1)
        
        self.feature = tf.convert_to_tensor(feature, dtype='float64')

    @tf.function
    def negative_log_likelihood(self,
                                neutral_div_log,
                                neutral_poly_logit,
                                omega_alpha_coeff,
                                foreground_poly_coeff):
        omega_alpha = tf.math.exp(tf.linalg.matvec(self.feature, omega_alpha_coeff))
        foreground_poly_logit = tf.linalg.matvec(self.feature, foreground_poly_coeff)
        neutral_div_rate = tf.math.exp(neutral_div_log)
        
        nll = self.mk_model.negative_log_likelihood(neutral_div_rate,
                                                    neutral_poly_logit,
                                                    omega_alpha,
                                                    foreground_poly_logit)
        
        return nll

    def compute_omega_alpha(self, omega_alpha_coeff):
        omega_alpha = tf.math.exp(tf.linalg.matvec(self.feature, omega_alpha_coeff))
        return omega_alpha.numpy()
    
    def fit(self, initial_para=None):
        n_coeff = self.feature.shape[1]
        
        if initial_para is None:
            initial_para = np.zeros(n_coeff * 2 + 2, dtype='float64')
            # mean neutral divergence
            dS = tf.reduce_mean(self.mk_model.neutral_div).numpy()
            # mean neutral polymorphism
            pS = tf.reduce_mean(self.mk_model.neutral_poly).numpy()
            # mean foreground polymorphism
            pN = tf.reduce_mean(self.mk_model.foreground_poly).numpy()
            # mean foreground divergence
            dN = tf.reduce_mean(self.mk_model.foreground_div).numpy()
            
            initial_para[0] = np.log(dS)
            initial_para[1] = np.log(pS / (1 - pS))

            if dN/dS - pN/pS > 0:
                initial_para[2] = np.log(dN/dS - pN/pS)
            else:
                initial_para[2] = -5

            initial_para[2+n_coeff] = np.log(pN/(1 - pN))
        else:
            initial_para = np.array(initial_para)
        
        assert initial_para.shape[0] == n_coeff * 2 + 2, 'Invalid initial values'
        
        # batched calculation of log likelihood
        def loss(x):
            x = tf.Variable(x, dtype='float64')
            
            with tf.GradientTape() as g:
                nll = self.negative_log_likelihood(x[0], x[1], x[2:2+n_coeff], x[2+n_coeff:])
            dy_dx = g.gradient(nll, x)

            return nll.numpy(), dy_dx.numpy()
        
        # optimize parameters
        # factr controls the accuracy (check doc for details)
        # maxfun and maxiter control the maximum number of iterations
        est, func, info = optimize.fmin_l_bfgs_b(loss, initial_para, pgtol=0,
                iprint=-1, factr=100., maxfun=100000, maxiter=100000)

        if (info['warnflag'] == 0):
            print('The MK regression is converged properly.')
        else:
            print('Warning: The MK regression is NOT converged!')

        print('Log likelihood = {}'.format(-1. * func))
        
        # calculate confidence intervals and p-values
        # TensorFlow 2.x solution for calculating hessian matrix
        # modified from:
        # https://github.com/tensorflow/tensorflow/issues/29781
        def hessian_func(x):
            x = tf.Variable(x, dtype='float64')
            n_coeff = self.feature.shape[1]
            
            with tf.GradientTape(persistent=True) as h:
                with tf.GradientTape() as g:
                    y = self.negative_log_likelihood(x[0], x[1], x[2:2+n_coeff], x[2+n_coeff:])
                # gradient of negative log likelihood w.r.t parameters
                dy_dx = g.gradient(y, x)

            # the jacobian of gradient is the transpose of hessian.
            # the transpose if irrevlevant because our matrix is
            # symetric (assume log likelihood function is smooth).
            # experimental_use_pfor=False disable parallization
            # and may save memory.
            hessian = h.jacobian(dy_dx, x, experimental_use_pfor=False)
                
            return hessian.numpy()

        # calculate hessian of negative log likelihood
        hessian = hessian_func(est)

        # check if Hessian matrix is positive definite
        if not np.all(np.linalg.eigvals(hessian) > 0):
            print('Warning: Hessian matrix is not positive definite!')

        # Wald test for significance
        standard_error = np.sqrt(np.linalg.inv(hessian).diagonal())
        z_score = est/standard_error
        pvalue = 2 * stats.norm.cdf(-np.abs(z_score))

        # output data structure
        class ModelResult(object):
            def __init__(self, x, fun, se, z, pvalue):
                # estimated parameters
                self.x = x
                # final loss (negative log likelihood)
                self.fun = fun
                # standard error
                self.se = se
                # z score
                self.z = z
                # p value
                self.pvalue = pvalue
        
        output = ModelResult(est, func, standard_error, z_score, pvalue)
        est_omega_alpha = self.compute_omega_alpha(est[2:2+n_coeff])
        return output, est_omega_alpha


def check_binary(x, name):
    assert np.sum(np.logical_or(x == 1, x == 0)) == x.shape[0], name + ' is not binary'
        
       
if __name__ == '__main__': 
    # parse cmd arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", dest="neutral_file", type=str, required=True,
                        help="input file of neutral sites")

    parser.add_argument("-f", dest="foreground_file", type=str, required=True,
                        help="input file of functional sites")

    parser.add_argument("-p", dest="parameter_file", type=str, required=True,
                        help="output file of estimated coefficients for omega_a")

    parser.add_argument("-o", dest="omega_a_file", type=str, required=False,
                        help="output file of site-wise omega_a (optional)")

    parser.add_argument("-g", dest="gamma_file", type=str, required=False,
                        help="output file of estimated coefficients for polymorphic rate (optional)")

    parser.add_argument("-m", dest="model", type=str, required=False, default='all',
                        help="model variables, as a list of column names, 'all' (default), or an empty list for the null model (intercept only)")

    parser.add_argument("-r", dest="compute_r2", required=False, default=False, action='store_true',
                        help="compute lielihood-based partial R2")

    args = parser.parse_args()

    neutral_input_file = args.neutral_file
    foreground_input_file = args.foreground_file
    neutral_data = pd.read_csv(neutral_input_file, sep='\t')
    foreground_data = pd.read_csv(foreground_input_file, sep='\t')

    assert neutral_data.shape[1] == 2, 'The neutral file must have two columns.'
    assert foreground_data.shape[1] >= 2, 'The foreground file must have more than two columns.'

    neutral_div = neutral_data.iloc[:, 0].values
    neutral_poly = neutral_data.iloc[:, 1].values
    foreground_div = foreground_data.iloc[:, 0].values
    foreground_poly = foreground_data.iloc[:, 1].values
    if args.model == 'null':
        variables = []
        feature = foreground_data[[]].values
    elif args.model == 'all':
        variables = foreground_data.columns[2:] 
        feature = foreground_data.iloc[:, 2:].values
    else:
        variables = args.model.split(',')
        feature = foreground_data[variables].values

    check_binary(neutral_div, 'neutral div')
    check_binary(neutral_poly, 'neutral poly')
    check_binary(foreground_div, 'foreground div')
    check_binary(foreground_poly, 'foreground poly')
   

    regression_model = SimpleMKRegression(neutral_div,
                                          neutral_poly,
                                          foreground_div,
                                          foreground_poly,
                                          feature)

    res, est_omega_alpha = regression_model.fit()

    # regression coefficient (beta)
    para_names = ['intercept']
    para_names += [x + '_coeff' for x in variables]
    df = pd.DataFrame.from_dict(OrderedDict([('parameter', para_names),
                                             ('estimate', res.x[2:feature.shape[1] + 3]),
                                             ('se', res.se[2:feature.shape[1] + 3]),
                                             ('z-score', res.z[2:feature.shape[1] + 3]),
                                             ('p-value', res.pvalue[2:feature.shape[1] + 3]),
                                            ]))

    df.to_csv(args.parameter_file, sep='\t', index=False)

    # regression coefficient (gamma)
    if args.gamma_file is not None:
        para_names = ['gamma_intercept']
        para_names += [x + '_gamma_coeff' for x in variables]
        df = pd.DataFrame.from_dict(OrderedDict([('parameter', para_names),
                                                 ('estimate', res.x[feature.shape[1] + 3:]),
                                                 ('se', res.se[feature.shape[1] + 3:]),
                                                 ('z-score', res.z[feature.shape[1] + 3:]),
                                                 ('p-value', res.pvalue[feature.shape[1] + 3:]),
                                                ]))

        df.to_csv(args.gamma_file, sep='\t', index=False)

    if args.omega_a_file is not None:
        df = pd.DataFrame.from_dict({'omega_a': est_omega_alpha})
        df.to_csv(args.omega_a_file, sep='\t', index=False)

    # compute partial R2
    if args.compute_r2:
        print("Fitting null model to estimate R2...")
        assert args.model != 'null', 'partial R2 cannot be computed for the null model.'
        logLk1 = res.fun

        feature0 = foreground_data[[]].values
        regression_model0 = SimpleMKRegression(neutral_div,
                                              neutral_poly,
                                              foreground_div,
                                              foreground_poly,
                                              feature0)

        res0, est_omega_alpha0 = regression_model0.fit()
        logLk0 = res0.fun
        
        n = float(foreground_data.shape[0])
        logLkMax = 1. - math.exp((2./n) * logLk0)
        r2 = 1. - math.exp(-(2./n) * (logLk1 - logLk0))
        r2n = r2 / logLkMax

        print('Partial (normalized) R2 = {}'.format(r2n))


