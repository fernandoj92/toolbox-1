package eu.amidst.ferjorosa.examples;

import eu.amidst.core.datastream.*;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.core.learning.parametric.ParallelMaximumLikelihood;
import eu.amidst.core.learning.parametric.bayesian.SVB;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.utils.DAGGenerator;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by equipo on 09/03/2017.
 */
public class TestVMPBug {

    public static void main(String[] args) throws Exception {
        //asiaHiddenVisitComparison();
        //asiaLCM();
        sprinklerHiddenCloudyComparison();
        //sprinklerLCM();
    }

    /******************************************************************************************************************/
    /******************************************************************************************************************/
    /******************************************************************************************************************/

    // Este caso compara la red de Asia tradicional en la que el nodo vVisitAsia se oculta (LV)
    private static void asiaHiddenVisitComparison(){
        System.out.println("\n\n/*******************************************************************************/");
        System.out.println("/******************************** ASIA MLE *************************************/");
        System.out.println("/*******************************************************************************/");
        DataStream<DataInstance> dataAsia = DataStreamLoader.open("datasets/ferjorosa/data/Asia_train.arff");
        DAG asiaDAG = defineAsiaDAG(dataAsia);
        BayesianNetwork asiaMLE = learnWithMLE(asiaDAG, dataAsia);
        System.out.println(asiaMLE.toString());

        System.out.println("\n\n/*******************************************************************************/");
        System.out.println("/******************************** ASIA SVB *************************************/");
        System.out.println("/*******************************************************************************/");
        DataStream<DataInstance> dataAsiaHiddenVisit = DataStreamLoader.open("datasets/ferjorosa/data/Asia_trainHidden.arff");
        DAG asiaDAGHiddenVisit = defineAsiaDAGHiddenVisit(dataAsiaHiddenVisit);
        BayesianNetwork asiaSVB = learnWithSVB(asiaDAGHiddenVisit, dataAsiaHiddenVisit);
        System.out.println(asiaSVB.toString());
    }

    // Este caso aprende un LCM con todas las variables de Asia y poniendo un root oculto
    private static void asiaLCM(){
        System.out.println("\n\n/*******************************************************************************/");
        System.out.println("/******************************** ASIA LCM *************************************/");
        System.out.println("/*******************************************************************************/");
        DataStream<DataInstance> dataAsiaLCM = DataStreamLoader.open("datasets/ferjorosa/data/Asia_train.arff");
        DAG asiaLCM = defineAsiaDAGLCM(dataAsiaLCM);
        BayesianNetwork asiaLCMSVB = learnWithSVB(asiaLCM, dataAsiaLCM);
        System.out.println(asiaLCMSVB.toString());
    }

    private static void sprinklerHiddenCloudyComparison(){

        System.out.println("\n\n/*******************************************************************************/");
        System.out.println("/******************************** Sprinkler MLE *************************************/");
        System.out.println("/*******************************************************************************/");
        DataStream<DataInstance> dataSprinkler = DataStreamLoader.open("datasets/ferjorosa/data/sprinklerData300.arff");
        DAG sprinklerDAG = defineSprinklerDAG(dataSprinkler);
        BayesianNetwork sprinklerMLE = learnWithMLE(sprinklerDAG, dataSprinkler);
        System.out.println(sprinklerMLE.toString());

        System.out.println("\n\n/*******************************************************************************/");
        System.out.println("/******************************** Sprinkler SVB *************************************/");
        System.out.println("/*******************************************************************************/");
        DataStream<DataInstance> dataHiddenCloudy = DataStreamLoader.open("datasets/ferjorosa/data/sprinklerDataHidden.arff");
        DAG sprinklerDAGHiddenCloudy = defineSprinklerDAGHiddenCloudy(dataHiddenCloudy);
        BayesianNetwork sprinklerSVB = learnWithSVB(sprinklerDAGHiddenCloudy, dataHiddenCloudy);
        System.out.println(sprinklerSVB.toString());
    }

    private static void sprinklerLCM(){
        System.out.println("\n\n/*******************************************************************************/");
        System.out.println("/******************************** Sprinkler LCM *************************************/");
        System.out.println("/*******************************************************************************/");
        DataStream<DataInstance> dataSprinklerLCM = DataStreamLoader.open("datasets/ferjorosa/data/sprinklerData300.arff");
        DAG sprinklerLCM = defineSprinklerLCM(dataSprinklerLCM);
        BayesianNetwork sprinklerLCMSVB = learnWithSVB(sprinklerLCM, dataSprinklerLCM);
        System.out.println(sprinklerLCMSVB.toString());
    }

    /******************************************************************************************************************/
    /******************************************************************************************************************/
    /******************************************************************************************************************/

    // Creamos el DAG tradicional de Asia sin variables latentes (ocultas).
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

    // Creamos el DAG tradicional de Asia con vVisitAsia como variable latente.
    private static DAG defineAsiaDAGHiddenVisit(DataStream<DataInstance> data){

        List<Attribute> attributes = new ArrayList<>();
        attributes.add(data.getAttributes().getAttributeByName("vTuberculosis"));
        attributes.add(data.getAttributes().getAttributeByName("vSmoking"));
        attributes.add(data.getAttributes().getAttributeByName("vLungCancer"));
        attributes.add(data.getAttributes().getAttributeByName("vTbOrCa"));
        attributes.add(data.getAttributes().getAttributeByName("vXRay"));
        attributes.add(data.getAttributes().getAttributeByName("vBronchitis"));
        attributes.add(data.getAttributes().getAttributeByName("vDyspnea"));

        Attributes attributeCollection = new Attributes(attributes);

        Variables variables = new Variables(attributeCollection);

        Variable hiddenVisitAsia = variables.newMultinomialVariable("hiddenVisitAsia", 2);

        DAG dag = new DAG(variables);

        dag.getParentSet(variables.getVariableByName("vXRay")).addParent(variables.getVariableByName("vTbOrCa"));
        dag.getParentSet(variables.getVariableByName("vDyspnea")).addParent(variables.getVariableByName("vTbOrCa"));
        dag.getParentSet(variables.getVariableByName("vDyspnea")).addParent(variables.getVariableByName("vBronchitis"));
        dag.getParentSet(variables.getVariableByName("vTbOrCa")).addParent(variables.getVariableByName("vTuberculosis"));
        dag.getParentSet(variables.getVariableByName("vTbOrCa")).addParent(variables.getVariableByName("vLungCancer"));
        dag.getParentSet(variables.getVariableByName("vTuberculosis")).addParent(hiddenVisitAsia);
        dag.getParentSet(variables.getVariableByName("vLungCancer")).addParent(variables.getVariableByName("vSmoking"));
        dag.getParentSet(variables.getVariableByName("vBronchitis")).addParent(variables.getVariableByName("vSmoking"));

        return dag;
    }

    // Creamos un LCM con todas los atributos de Asia y una LV como raiz
    private static DAG defineAsiaDAGLCM(DataStream<DataInstance> data){
        return  DAGGenerator.getHiddenNaiveBayesStructure(data.getAttributes(),"GlobalHidden", 3);
    }

    /******************************************************************************************************************/

    // Creamos la estructura de la red bayesiana "Sprinkler" tradicional sin variables latentes (ocultas)
    private static DAG defineSprinklerDAG(DataStream<DataInstance> data){

        Variables variables = new Variables(data.getAttributes());

        DAG dag = new DAG(variables);

        dag.getParentSet(variables.getVariableByName("sprinkler")).addParent(variables.getVariableByName("cloudy"));
        dag.getParentSet(variables.getVariableByName("rain")).addParent(variables.getVariableByName("cloudy"));
        dag.getParentSet(variables.getVariableByName("wetGrass")).addParent(variables.getVariableByName("sprinkler"));
        dag.getParentSet(variables.getVariableByName("wetGrass")).addParent(variables.getVariableByName("rain"));

        return dag;
    }

    // Creamos la estructura de la red bayesiana Sprinkler tradicional con cloudy como LV
    private static DAG defineSprinklerDAGHiddenCloudy(DataStream<DataInstance> data){

        Variables variables = new Variables(data.getAttributes());

        Variable latentCloudy = variables.newMultinomialVariable("latentCloudy", 2);

        DAG dag = new DAG(variables);

        dag.getParentSet(variables.getVariableByName("sprinkler")).addParent(latentCloudy);
        dag.getParentSet(variables.getVariableByName("rain")).addParent(latentCloudy);
        dag.getParentSet(variables.getVariableByName("wetGrass")).addParent(variables.getVariableByName("sprinkler"));
        dag.getParentSet(variables.getVariableByName("wetGrass")).addParent(variables.getVariableByName("rain"));

        return dag;
    }

    // Creamos un LCM con todos los atributos de Sprinkler y una LV como raiz
    private static DAG defineSprinklerLCM(DataStream<DataInstance> data){
        return  DAGGenerator.getHiddenNaiveBayesStructure(data.getAttributes(),"GlobalHidden", 2);
    }

    /******************************************************************************************************************/
    /******************************************************************************************************************/
    /******************************************************************************************************************/

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
        for (DataOnMemory<DataInstance> batch : data.iterableOverBatches(100)){
            svb.updateModel(batch);
        }

        //And we get the model
        BayesianNetwork bnModel = svb.getLearntBayesianNetwork();

        return bnModel;
    }
}
