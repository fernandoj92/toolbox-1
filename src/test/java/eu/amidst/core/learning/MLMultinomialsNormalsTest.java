package eu.amidst.core.learning;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataStream;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.io.BayesianNetworkLoader;
import eu.amidst.core.utils.BayesianNetworkSampler;
import eu.amidst.core.variables.Variable;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.assertTrue;

/**
 * Created by Hanen on 27/01/15.
 */
public class MLMultinomialsNormalsTest {

        @Test
        public void testingML2() throws  IOException, ClassNotFoundException {

            // load the true WasteIncinerator hugin Bayesian network containing 3 Multinomial and 6 Gaussian variables

            BayesianNetwork trueBN = BayesianNetworkLoader.loadFromFile("./networks/WasteIncinerator.bn");

            System.out.println("\nWasteIncinerator network \n ");
            System.out.println(trueBN.getDAG().toString());
            System.out.println(trueBN.toString());

            //Sampling from trueBN
            BayesianNetworkSampler sampler = new BayesianNetworkSampler(trueBN);
            sampler.setSeed(0);

            //Load the sampled data
            DataStream<DataInstance> data = sampler.sampleToDataBase(100000);

            //try{
            //    sampler.sampleToAnARFFFile("./data/WasteIncineratorSamples.arff", 10000);
            //} catch (IOException ex){
            //}
            //DataStream data = new StaticDataOnDiskFromFile(new ARFFDataReader(new String("data/WasteIncineratorSamples.arff")));

            //Structure learning is excluded from the test, i.e., we use directly the initial Asia network structure
            // and just learn then test the parameter learning

            //Parameter Learning
            MaximumLikelihoodForBN.setBatchSize(1000);
            MaximumLikelihoodForBN.setParallelMode(true);
            BayesianNetwork bnet = MaximumLikelihoodForBN.learnParametersStaticModel(trueBN.getDAG(), data);

            //Check if the probability distributions of each node
            for (Variable var : trueBN.getStaticVariables()) {
                System.out.println("\n------ Variable " + var.getName() + " ------");
                System.out.println("\nTrue distribution:\n"+ trueBN.getDistribution(var));
                System.out.println("\nLearned distribution:\n"+ bnet.getDistribution(var));
                assertTrue(bnet.getDistribution(var).equalDist(trueBN.getDistribution(var), 0.05));
            }

            //Or check directly if the true and learned networks are equals
            assertTrue(bnet.equalBNs(trueBN,0.05));
        }
}

