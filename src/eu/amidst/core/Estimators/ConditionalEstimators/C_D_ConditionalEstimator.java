package eu.amidst.core.Estimators.ConditionalEstimators;

import eu.amidst.core.Estimators.UnConditionalEstimators.GaussianEstimator;
import eu.amidst.core.Potential.Potential;
import eu.amidst.core.StaticDataBase.DataInstance;

/**
 * Created by afa on 03/07/14.
 */
public class C_D_ConditionalEstimator implements ConditionalEstimator {
    private GaussianEstimator[] estim;

    @Override
    public Potential getRestrictedPotential(DataInstance instance) {
        return null;
    }

    @Override
    public double[] getSufficientStatistics(DataInstance instance) {
        return new double[0];
    }

    @Override
    public double[] getExpectedSufficientStatistics(DataInstance instance, Potential pot) {
        return new double[0];
    }

    @Override
    public void setExpectationParameters(double[] ss) {

    }

    @Override
    public double[] getExpectationParameters() {
        return new double[0];
    }

    @Override
    public double getProbability(DataInstance data) {
        return 0;
    }
}
