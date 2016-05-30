package eu.amidst.tutorials.huginsa2016.examples;

import COM.hugin.HAPI.ExceptionHugin;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataStream;
import eu.amidst.core.io.BayesianNetworkWriter;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.huginlink.io.BNWriterToHugin;
import eu.amidst.latentvariablemodels.staticmodels.FactorAnalysis;

import java.io.IOException;

/**
 * Created by rcabanas on 23/05/16.
 */
public class StaticModelSaveToDisk {
    public static void main(String[] args) throws ExceptionHugin, IOException {

        //Load the datastream
        String filename = "datasets/simulated/exampleDS_d0_c5.arff";
        DataStream<DataInstance> data = DataStreamLoader.openFromFile(filename);

        //Learn the model
        FactorAnalysis model = new FactorAnalysis(data.getAttributes());
        model.setNumberOfLatentVariables(3);
        model.setWindowSize(200);
        model.updateModel(data);
        BayesianNetwork bn = model.getModel();

        System.out.println(bn);

        // Save with .bn format
        BayesianNetworkWriter.saveToFile(bn, "networks/simulated/exampleBN.bn");

        // Save with hugin format
        BNWriterToHugin.saveToHuginFile(bn, "networks/simulated/exampleBN.net");
    }

}