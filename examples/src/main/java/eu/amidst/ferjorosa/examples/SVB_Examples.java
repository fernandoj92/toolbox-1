package eu.amidst.ferjorosa.examples;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.datastream.DataStream;
import eu.amidst.core.inference.messagepassing.VMP;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.core.learning.parametric.ParallelMaximumLikelihood;
import eu.amidst.core.learning.parametric.ParameterLearningAlgorithm;
import eu.amidst.core.learning.parametric.bayesian.SVB;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;

/**
 * Created by fer on 22/12/16.
 */
public class SVB_Examples {

    public static void main(String[] args) throws Exception {
        onlyRainWithLatentCloudy();
        threeObservedLatentcloudy();
        sprinklerMaximumLikelihood();

    }

    private static void onlyRainWithLatentCloudy(){

        DataStream<DataInstance> data = DataStreamLoader.open("datasets/ferjorosa/sprinklerData300_rain.arff");

        SVB svb = new SVB();
        svb.setWindowsSize(1000);
        svb.setSeed(5);
        VMP vmp = svb.getPlateuStructure().getVMP();
        vmp.setTestELBO(true);
        vmp.setMaxIter(1000);
        vmp.setThreshold(0.0001);

        Variables variables = new Variables(data.getAttributes());

        Variable latentVar = variables.newMultinomialVariable("latentCloudy", 2);
        Variable rain = variables.getVariableByName("rain");

        DAG dag = new DAG(variables);

        dag.getParentSet(rain).addParent(latentVar);

        svb.setDAG(dag);
        //svb.initLearning();
        svb.setDataStream(data);
        svb.runLearning();

        BayesianNetwork learnBN = svb.getLearntBayesianNetwork();

        System.out.println(learnBN.toString());
    }

    private static void threeObservedLatentcloudy(){

        DataStream<DataInstance> data = DataStreamLoader.open("datasets/ferjorosa/sprinklerDataHidden.arff");

        SVB svb = new SVB();
        svb.setWindowsSize(1000);
        svb.setSeed(5);
        VMP vmp = svb.getPlateuStructure().getVMP();
        vmp.setTestELBO(true);
        vmp.setMaxIter(1000);
        vmp.setThreshold(0.0001);

        Variables variables = new Variables(data.getAttributes());

        Variable latentCloudy = variables.newMultinomialVariable("latentCloudy", 2);
        Variable rain = variables.getVariableByName("rain");
        Variable sprinkler = variables.getVariableByName("sprinkler");
        Variable wetGrass = variables.getVariableByName("wetGrass");

        DAG dag = new DAG(variables);

        dag.getParentSet(rain).addParent(latentCloudy);
        dag.getParentSet(sprinkler).addParent(latentCloudy);
        dag.getParentSet(wetGrass).addParent(sprinkler);
        dag.getParentSet(wetGrass).addParent(rain);

        svb.setDAG(dag);
        //svb.initLearning();
        svb.setDataStream(data);
        svb.runLearning();

        BayesianNetwork learnBN = svb.getLearntBayesianNetwork();

        System.out.println(learnBN.toString());
    }

    public static void sprinklerMaximumLikelihood(){

        DataStream<DataInstance> data = DataStreamLoader.open("datasets/ferjorosa/sprinklerData300.arff");

        //We create MaximumLikelihood object
        ParameterLearningAlgorithm parameterLearningAlgorithm = new ParallelMaximumLikelihood();

        Variables variables = new Variables(data.getAttributes());

        DAG dag = new DAG(variables);

        dag.getParentSet(variables.getVariableByName("sprinkler")).addParent(variables.getVariableByName("cloudy"));
        dag.getParentSet(variables.getVariableByName("rain")).addParent(variables.getVariableByName("cloudy"));
        dag.getParentSet(variables.getVariableByName("wetGrass")).addParent(variables.getVariableByName("sprinkler"));
        dag.getParentSet(variables.getVariableByName("wetGrass")).addParent(variables.getVariableByName("rain"));

        //We fix the DAG structure
        parameterLearningAlgorithm.setDAG(dag);

        //We should invoke this method before processing any data
        parameterLearningAlgorithm.initLearning();

        //Then we show how we can perform parameter learnig by a sequential updating of data batches.
        for (DataOnMemory<DataInstance> batch : data.iterableOverBatches(1000)){
            parameterLearningAlgorithm.updateModel(batch);
        }

        //And we get the model
        BayesianNetwork bnModel = parameterLearningAlgorithm.getLearntBayesianNetwork();

        //We print the model
        System.out.println(bnModel.toString());
    }
}
