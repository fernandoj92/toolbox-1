package eu.amidst.ferjorosa.examples;

import eu.amidst.core.distribution.Multinomial;
import eu.amidst.core.distribution.Multinomial_MultinomialParents;
import eu.amidst.core.inference.InferenceAlgorithm;
import eu.amidst.core.inference.messagepassing.VMP;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.HashMapAssignment;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.huginlink.inference.HuginInference;

/**
 * Created by Fernando on 29/12/2016.
 */
public class Inference_Examples {

    public static void main(String[] args) throws Exception {
        //vmpExample();
        System.loadLibrary("hgapi83_amidst-64.jar");
        huginExample();
    }

    private static void vmpExample(){
        Variables variables = new Variables();
        Variable varA = variables.newMultinomialVariable("A", 2);
        Variable varB = variables.newMultinomialVariable("B", 2);

        DAG dag = new DAG(variables);

        dag.getParentSet(varB).addParent(varA);

        BayesianNetwork bn = new BayesianNetwork(dag);

        Multinomial distA = bn.getConditionalDistribution(varA);
        Multinomial_MultinomialParents distB = bn.getConditionalDistribution(varB);

        //distA.setProbabilities(new double[]{0.8, 0.2});
        distB.getMultinomial(0).setProbabilities(new double[]{0.70, 0.30});
        distB.getMultinomial(1).setProbabilities(new double[]{0.40, 0.60});

        System.out.println(bn.toString());


        HashMapAssignment assignment = new HashMapAssignment(1);
        assignment.setValue(varB, 1.0);

        VMP vmp = new VMP();
        vmp.setTestELBO(true);
        vmp.setMaxIter(100);
        vmp.setThreshold(0.0001);

        vmp.setModel(bn);

        vmp.setEvidence(assignment);
        vmp.runInference();

        Multinomial postA = vmp.getPosterior(varA);
        System.out.println("P(A) = " + postA.toString());
    }

    private static void huginExample(){

        Variables variables = new Variables();
        Variable varA = variables.newMultinomialVariable("A", 2);
        Variable varB = variables.newMultinomialVariable("B", 2);

        DAG dag = new DAG(variables);

        dag.getParentSet(varB).addParent(varA);

        BayesianNetwork bn = new BayesianNetwork(dag);

        Multinomial distA = bn.getConditionalDistribution(varA);
        Multinomial_MultinomialParents distB = bn.getConditionalDistribution(varB);

        distA.setProbabilities(new double[]{0.5, 0.5});
        distB.getMultinomial(0).setProbabilities(new double[]{0.70, 0.30});
        distB.getMultinomial(1).setProbabilities(new double[]{0.40, 0.60});

        System.out.println(bn.toString());


        //Create an instance of a inference algorithm.
        // In this case, a HUGIN exact algorithm for inference is used.
        InferenceAlgorithm inferenceAlgorithm = new HuginInference();
        //Then, we set the BN model
        inferenceAlgorithm.setModel(bn);

        HashMapAssignment assignment = new HashMapAssignment(1);
        assignment.setValue(varB, 1.0);

        inferenceAlgorithm.setEvidence(assignment);

        //Run inference.
        inferenceAlgorithm.runInference();

        Multinomial postA = inferenceAlgorithm.getPosterior(varA);
        System.out.println("P(A) = " + postA.toString());
    }
}
