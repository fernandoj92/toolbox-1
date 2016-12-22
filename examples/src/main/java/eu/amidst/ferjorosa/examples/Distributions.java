package eu.amidst.ferjorosa.examples;

import eu.amidst.core.distribution.Multinomial;
import eu.amidst.core.distribution.Normal;
import eu.amidst.core.exponentialfamily.EF_Multinomial;
import eu.amidst.core.exponentialfamily.EF_Normal;
import eu.amidst.core.exponentialfamily.NaturalParameters;
import eu.amidst.core.utils.ArrayVector;
import eu.amidst.core.variables.Assignment;
import eu.amidst.core.variables.HashMapAssignment;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;

/**
 * Created by fer on 23/11/16.
 */
public class Distributions {

    public static void main(String[] args) throws Exception {

        multinomialTest();
    }

    private static void normalTest(){

        Variables variables = new Variables();
        Variable ef_gaussian = variables.newGaussianVariable("ef_gaussian");
        Variable gaussian = variables.newGaussianVariable("gaussian");

        EF_Normal ef_normal = new EF_Normal(ef_gaussian);
        ef_normal.setNaturalWithMeanPrecision(0, 1);
        ef_normal.updateMomentFromNaturalParameters();

        Normal normal = new Normal(gaussian);

        Assignment ef_assignment = new HashMapAssignment(1);
        ef_assignment.setValue(ef_gaussian,0.5);
        Assignment assignment = new HashMapAssignment(1);
        assignment.setValue(gaussian,0.5);

        System.out.println(Math.exp(normal.getLogProbability(assignment)));
        System.out.println(Math.exp(ef_normal.computeLogProbabilityOf(ef_assignment)));
    }

    private static void multinomialTest(){

        Variables variables = new Variables();
        Variable ef_multinomial = variables.newMultinomialVariable("ef_multinomial", 3);
        Variable multinomial = variables.newMultinomialVariable("multinomial", 3);

        EF_Multinomial ef_multinomialDist = new EF_Multinomial(ef_multinomial);

        NaturalParameters naturalParameters = new ArrayVector(ef_multinomial.getNumberOfStates());
        naturalParameters.set(0, 1/3);
        naturalParameters.set(1, 1/3);
        naturalParameters.set(2, 1/3);
        ef_multinomialDist.setNaturalParameters(naturalParameters);
        ef_multinomialDist.updateMomentFromNaturalParameters();

        Multinomial multinomialDist = new Multinomial(multinomial);

        Assignment ef_assignment = new HashMapAssignment(1);
        ef_assignment.setValue(ef_multinomial,1);
        Assignment assignment = new HashMapAssignment(1);
        assignment.setValue(multinomial,1);

        System.out.println(Math.exp(multinomialDist.getLogProbability(assignment)));
        System.out.println(Math.exp(ef_multinomialDist.computeLogProbabilityOf(ef_assignment)));
    }
}
