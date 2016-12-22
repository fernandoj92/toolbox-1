package eu.amidst.core.examples.variables;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataStream;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;

/**
 * Created by fer on 3/11/16.
 */
public class DistributionTest {

    public static void main(String[] args) throws Exception {

        Variables variables = new Variables();

        Variable child_gaussian = variables.newGaussianVariable("child_gaussian");
        Variable parent_gaussian = variables.newGaussianVariable("parent_gaussian");
        Variable parent_gaussian2 = variables.newGaussianVariable("parent_gaussian2");
        Variable parent_gaussian3 = variables.newGaussianVariable("parent_gaussian3");

        DAG dag = new DAG(variables);

        //dag.getParentSet(child_multinomial).addParent(parent_gaussian);
        dag.getParentSet(child_gaussian).addParent(parent_gaussian);
        dag.getParentSet(child_gaussian).addParent(parent_gaussian2);
        dag.getParentSet(child_gaussian).addParent(parent_gaussian3);

        BayesianNetwork bn = new BayesianNetwork(dag);

        System.out.println(bn.toString());
    }
}
