package eu.amidst.ferjorosa.examples;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.datastream.DataStream;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.core.learning.parametric.ParallelMaximumLikelihood;
import eu.amidst.core.learning.parametric.ParameterLearningAlgorithm;
import eu.amidst.core.learning.parametric.bayesian.SVB;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.utils.DAGGenerator;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;

/**
 * Created by Fernando on 12/4/2016.
 */
public class EF_Examples {

    public static void main(String[] args) throws Exception {
        //sprinklerOneObservedVar();
        //sprinklerOneLatentVar();
        //oneGaussianTwoMultinomial();
        sprinklerMaximumLikelihood();
    }

    public static void oneGaussianTwoMultinomial(){

        DataStream<DataInstance> data = DataStreamLoader.open("datasets/ferjorosa/oneGaussianTwoMultinomial.arff");

        //We create MaximumLikelihood object
        ParameterLearningAlgorithm parameterLearningAlgorithm = new ParallelMaximumLikelihood();

        Variables variables = new Variables(data.getAttributes());

        //We fix the DAG structure
        parameterLearningAlgorithm.setDAG(new DAG(variables));

        //We should invoke this method before processing any data
        parameterLearningAlgorithm.initLearning();

        //Then we show how we can perform parameter learnig by a sequential updating of data batches.
        for (DataOnMemory<DataInstance> batch : data.iterableOverBatches(400)){
            System.out.println(parameterLearningAlgorithm.updateModel(batch));
        }

        //And we get the model
        BayesianNetwork bnModel = parameterLearningAlgorithm.getLearntBayesianNetwork();

        //We print the model
        System.out.println(bnModel.toString());
    }

    public static void sprinklerOneObservedVar(){

        DataStream<DataInstance> data = DataStreamLoader.open("datasets/ferjorosa/sprinklerData300_oneVar.arff");

        //We create MaximumLikelihood object
        ParameterLearningAlgorithm parameterLearningAlgorithm = new ParallelMaximumLikelihood();

        Variables variables = new Variables(data.getAttributes());

        //We fix the DAG structure
        parameterLearningAlgorithm.setDAG(new DAG(variables));

        //We should invoke this method before processing any data
        parameterLearningAlgorithm.initLearning();

        //Then we show how we can perform parameter learnig by a sequential updating of data batches.
        for (DataOnMemory<DataInstance> batch : data.iterableOverBatches(300)){
            System.out.println(parameterLearningAlgorithm.updateModel(batch));
        }

        //And we get the model
        BayesianNetwork bnModel = parameterLearningAlgorithm.getLearntBayesianNetwork();

        //We print the model
        System.out.println(bnModel.toString());
    }

    public static void sprinklerMaximumLikelihood(){

        DataStream<DataInstance> data = DataStreamLoader.open("datasets/ferjorosa/sprinklerData300.arff");

        //We create MaximumLikelihood object
        ParameterLearningAlgorithm parameterLearningAlgorithm = new ParallelMaximumLikelihood();

        Variables variables = new Variables(data.getAttributes());

        DAG dag = new DAG(variables);

        dag.getParentSet(variables.getVariableByName("sprinkler")).addParent(variables.getVariableByName("cloudy"));

        //We fix the DAG structure
        parameterLearningAlgorithm.setDAG(dag);

        //We should invoke this method before processing any data
        parameterLearningAlgorithm.initLearning();

        //Then we show how we can perform parameter learnig by a sequential updating of data batches.
        for (DataOnMemory<DataInstance> batch : data.iterableOverBatches(400)){
            parameterLearningAlgorithm.updateModel(batch);
        }

        //And we get the model
        BayesianNetwork bnModel = parameterLearningAlgorithm.getLearntBayesianNetwork();

        //We print the model
        System.out.println(bnModel.toString());
    }

    public static void sprinklerOneLatentVar(){

        DataStream<DataInstance> data = DataStreamLoader.open("datasets/ferjorosa/sprinklerData300_oneVar.arff");

        //We create MaximumLikelihood object
        ParameterLearningAlgorithm parameterLearningAlgorithm = new SVB();

        Variables variables = new Variables(data.getAttributes());

        Variable latentVar = variables.newMultinomialVariable("latentVar", 3);
        Variable cloudy = variables.getVariableByName("cloudy");

        DAG dag = new DAG(variables);

        dag.getParentSet(cloudy).addParent(latentVar);

        parameterLearningAlgorithm.setDAG(dag);

        parameterLearningAlgorithm.initLearning();

        //Then we show how we can perform parameter learnig by a sequential updating of data batches.
        for (DataOnMemory<DataInstance> batch : data.iterableOverBatches(100)){
            parameterLearningAlgorithm.updateModel(batch);
        }

        //And we get the model
        BayesianNetwork bnModel = parameterLearningAlgorithm.getLearntBayesianNetwork();

        //We print the model
        System.out.println(bnModel.toString());
    }

    public static void sprinklerNaiveBayes(){

        //We can open the data stream using the static class DataStreamLoader
        DataStream<DataInstance> data = DataStreamLoader.open("datasets/ferjorosa/spinklerData300.arff");

        //We create a SVB object
        SVB parameterLearningAlgorithm = new SVB();

        //We fix the DAG structure
        parameterLearningAlgorithm.setDAG(DAGGenerator.getHiddenNaiveBayesStructure(data.getAttributes(),"GlobalHidden", 2));

        //We fix the size of the window
        parameterLearningAlgorithm.setWindowsSize(100);

        //We can activate the output
        parameterLearningAlgorithm.setOutput(true);

        //We set the data which is going to be used for leaning the parameters
        parameterLearningAlgorithm.setDataStream(data);

        //We perform the learning
        parameterLearningAlgorithm.runLearning();

        //And we get the model
        BayesianNetwork bnModel = parameterLearningAlgorithm.getLearntBayesianNetwork();

        //We print the model
        System.out.println(bnModel.toString());
    }
}