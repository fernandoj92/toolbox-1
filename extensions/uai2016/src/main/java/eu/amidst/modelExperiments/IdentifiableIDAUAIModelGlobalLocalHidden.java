
/*
* Licensed to the Apache Software Foundation (ASF) under one or more contributor
license agreements. See the NOTICE file distributed with this work for
additional information regarding copyright ownership. The ASF licenses this file
to You under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*
* See the License for the specific language governing permissions and
limitations under the License.
*
*/
package eu.amidst.modelExperiments;

import eu.amidst.core.variables.Variable;
import eu.amidst.flinklink.core.learning.parametric.IdenitifableModelling;

import java.io.Serializable;

/**
 * Created by ana@cs.aau.dk <mailto:ana@cs.aau.dk> on 15/02/16.
 */
public class IdentifiableIDAUAIModelGlobalLocalHidden implements IdenitifableModelling, Serializable {

    private int numLocalHiddenVariables;

    public IdentifiableIDAUAIModelGlobalLocalHidden(int numLocalHiddenVariables){
        this.numLocalHiddenVariables = numLocalHiddenVariables;
    }

    /**
     * Represents the serial version ID for serializing the object.
     */
    private static final long serialVersionUID =4107783324901370839L;

    @Override
    public int getNumberOfEpochs() {
        return numLocalHiddenVariables+4;
    }


    @Override
    public boolean isActiveAtEpoch(Variable variable,int epoch) {

        if (epoch==0)
            return true;

        if (variable.getName().startsWith("GlobalHidden"))
            return epoch%getNumberOfEpochs() ==0;
        else if (variable.getName().contains("Beta0"))
            return epoch%getNumberOfEpochs() ==1;
        else if (variable.getName().contains("Beta_GlobalHidden"))
            return epoch%getNumberOfEpochs() ==2;
        else if (variable.getName().contains("Gamma"))
            return epoch%getNumberOfEpochs() ==3;
        else if (variable.getName().contains("Beta_LocalHidden")) {
            int length = variable.getName().length();
            int index = variable.getName().charAt(length);
            return epoch % getNumberOfEpochs() == index+4;
        }else if (variable.getName().startsWith("LocalHidden_Mean_Parameter_")){
            int length = variable.getName().length();
            int index = variable.getName().charAt(length);
            return epoch % getNumberOfEpochs() == index*2;
        }else if (variable.getName().startsWith("LocalHidden_Gamma_Parameter_")){
            int length = variable.getName().length();
            int index = variable.getName().charAt(length);
            return epoch % getNumberOfEpochs() == index*2+1;
        }
        else
            return true;
    }

}

