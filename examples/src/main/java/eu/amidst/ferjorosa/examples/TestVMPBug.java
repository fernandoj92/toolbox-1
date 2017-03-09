package eu.amidst.ferjorosa.examples;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.datastream.DataStream;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.core.learning.parametric.ParallelMaximumLikelihood;
import eu.amidst.core.learning.parametric.bayesian.SVB;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variables;

/**
 * Created by equipo on 09/03/2017.
 */
public class TestVMPBug {

    public static void main(String[] args) throws Exception {

        DataStream<DataInstance> dataSprinkler = DataStreamLoader.open("datasets/ferjorosa/sprinklerData300.arff");
        DataStream<DataInstance> dataSprinklerHiddenCloudy = DataStreamLoader.open("datasets/ferjorosa/sprinklerDataHidden.arff");
        DataStream<DataInstance> dataAsia = DataStreamLoader.open("datasets/ferjorosa/Asia_train.arff");
        DataStream<DataInstance> dataAsiaHiddenVisit = DataStreamLoader.open("datasets/ferjorosa/sprinklerData300.arff");
        DataStream<DataInstance> dataAsiaHiddenVisitSmoking = DataStreamLoader.open("datasets/ferjorosa/sprinklerData300.arff");

    }

    /**
     * Creamos la estructura de la red bayesiana Asia tradicional sin variables latentes (ocultas).
     *
     * @return la red bayesiana Asia sin variables latentes.
     */
    private static DAG defineAsiaDAG(DataStream<DataInstance> data){

        Variables variables = new Variables(data.getAttributes());

        DAG dag = new DAG(variables);

        dag.getParentSet(variables.getVariableByName("vXRay")).addParent(variables.getVariableByName("vTbOrCa"));
        dag.getParentSet(variables.getVariableByName("vDyspnea")).addParent(variables.getVariableByName("vTbOrCa"));
        dag.getParentSet(variables.getVariableByName("vDyspnea")).addParent(variables.getVariableByName("vBronchitis"));
        dag.getParentSet(variables.getVariableByName("vTbOrCa")).addParent(variables.getVariableByName("vTuberculosis"));
        dag.getParentSet(variables.getVariableByName("vTbOrCa")).addParent(variables.getVariableByName("vLungCancer"));
        dag.getParentSet(variables.getVariableByName("vTuberculosis")).addParent(variables.getVariableByName("vVisitToAsia"));
        dag.getParentSet(variables.getVariableByName("vLungCancer")).addParent(variables.getVariableByName("vSmoking"));
        dag.getParentSet(variables.getVariableByName("vBronchitis")).addParent(variables.getVariableByName("vSmoking"));

        return dag;
    }

    /**
     * Creamos la estructura de la red bayesiana "Sprinkler" tradicional sin variables latentes (ocultas)
     *
     * @return la red bayesiana sprinkler sin variables latentes.
     */
    private static DAG defineSprinklerDAG(DataStream<DataInstance> data){

        Variables variables = new Variables(data.getAttributes());

        DAG dag = new DAG(variables);

        dag.getParentSet(variables.getVariableByName("sprinkler")).addParent(variables.getVariableByName("cloudy"));
        dag.getParentSet(variables.getVariableByName("rain")).addParent(variables.getVariableByName("cloudy"));
        dag.getParentSet(variables.getVariableByName("wetGrass")).addParent(variables.getVariableByName("sprinkler"));
        dag.getParentSet(variables.getVariableByName("wetGrass")).addParent(variables.getVariableByName("rain"));

        return dag;
    }

    /**
     * Aprende los parametros de la red utilizando MLE. Con cada llamada se crea una nueva instancia, asi que no hay
     * problemas de referencias o de parametros mal inicializados.
     */
    private static BayesianNetwork learnWithMLE(DAG dag, DataStream<DataInstance> data){

        //We create MaximumLikelihood object
        ParallelMaximumLikelihood mle = new ParallelMaximumLikelihood();

        //We fix the DAG structure
        mle.setDAG(dag);

        //We should invoke this method before processing any data
        mle.initLearning();

        //Then we show how we can perform parameter learnig by a sequential updating of data batches.
        for (DataOnMemory<DataInstance> batch : data.iterableOverBatches(1000)){
            mle.updateModel(batch);
        }

        //And we get the model
        BayesianNetwork bnModel = mle.getLearntBayesianNetwork();

        return bnModel;
    }

    /**
     * Aprende los parametros de la red utilizando el Variational Bayes. Con cada llamada se crea una nueva instancia,
     * asi que no hay problemas de referencias o de parametros mal inicializados.
     */
    private static BayesianNetwork learnWithSVB(DAG dag, DataStream<DataInstance> data){

        //We create MaximumLikelihood object
        SVB svb = new SVB();

        //We fix the DAG structure
        svb.setDAG(dag);

        //We should invoke this method before processing any data
        svb.initLearning();

        //Then we show how we can perform parameter learnig by a sequential updating of data batches.
        for (DataOnMemory<DataInstance> batch : data.iterableOverBatches(1000)){
            svb.updateModel(batch);
        }

        //And we get the model
        BayesianNetwork bnModel = svb.getLearntBayesianNetwork();

        return bnModel;
    }
}
