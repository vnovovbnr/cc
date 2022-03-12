package com.example.ftpipehd_mobile.model;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.weightinit.impl.XavierInitScheme;

import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

public class MNISTCNN {
    private long inputSize[][] = {{256, 784}, {256, 4, 26, 26}, {256, 4, 13, 13}, {256, 8, 11, 11}, {256, 8, 5, 5}};
    private SameDiff model;
    private SameDiff subModel;
    private Map<String, GradientUpdater> updaterMap;
    private int iteration = 0;
    private int epoch = 0;

    public SameDiff makeMNISTNet() {
        SameDiff sd = SameDiff.create();

        //Properties for MNIST dataset:
        int nIn = 28 * 28;
        int nOut = 10;

        //Create input and label variables
        SDVariable in = sd.placeHolder("input", DataType.FLOAT, -1, nIn);                 //Shape: [?, 784] - i.e., minibatch x 784 for MNIST
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, nOut);             //Shape: [?, 10] - i.e., minibatch x 10 for MNIST

        SDVariable reshaped = in.reshape(-1, 1, 28, 28);

        Pooling2DConfig poolConfig = Pooling2DConfig.builder().kH(2).kW(2).sH(2).sW(2).build();

        Conv2DConfig convConfig = Conv2DConfig.builder().kH(3).kW(3).build();

        // layer 1: Conv2D with a 3x3 kernel and 4 output channels
        SDVariable w0 = sd.var("w0", new XavierInitScheme('c', 28 * 28, 26 * 26 * 4), DataType.FLOAT, 3, 3, 1, 4);
        SDVariable b0 = sd.zero("b0", 4);

        SDVariable conv1 = sd.cnn().conv2d(reshaped, w0, b0, convConfig);

        // layer 2: MaxPooling2D with a 2x2 kernel and stride, and ReLU activation
        SDVariable pool1 = sd.cnn().maxPooling2d(conv1, poolConfig);

        SDVariable relu1 = sd.nn().relu(pool1, 0);

        // layer 3: Conv2D with a 3x3 kernel and 8 output channels
        SDVariable w1 = sd.var("w1", new XavierInitScheme('c', 13 * 13 * 4, 11 * 11 * 8), DataType.FLOAT, 3, 3, 4, 8);
        SDVariable b1 = sd.zero("b1", 8);

        SDVariable conv2 = sd.cnn().conv2d(relu1, w1, b1, convConfig);

        // layer 4: MaxPooling2D with a 2x2 kernel and stride, and ReLU activation
        SDVariable pool2 = sd.cnn().maxPooling2d(conv2, poolConfig);

        SDVariable relu2 = sd.nn().relu(pool2, 0);

        SDVariable flat = relu2.reshape(-1, 5 * 5 * 8);

        // layer 5: Output layer on flattened input
        SDVariable wOut = sd.var("wOut", new XavierInitScheme('c', 5 * 5 * 8, 10), DataType.FLOAT, 5 * 5 * 8, 10);
        SDVariable bOut = sd.zero("bOut", 10);

        SDVariable z = sd.nn().linear("z", flat, wOut, bOut);

        // softmax crossentropy loss function
        SDVariable out = sd.nn().softmax("out", z, 1);
        SDVariable loss = sd.loss().softmaxCrossEntropy("loss", label, out, null);

        sd.setLossVariables(loss);

        this.model = sd;
        return sd;
    }

    /**
     * This is a simple construction of sub model to test the feasibility of the Java NN
     * @param start
     * @param end
     * @return
     */
    public SameDiff simpleMakeSubModel(int start, int end) {
        SameDiff sd = SameDiff.create();

        long[] curInput = inputSize[start];

//        int nIn = -1;
//        for (int i : curInput) {
//            nIn *= i;
//        }


        //SDVariable inReshaped = sd.placeHolder("input", DataType.FLOAT, curInput);
        SDVariable inReshaped = sd.var("input", DataType.FLOAT, curInput);

        // SDVariable inReshaped = in.reshape(curInput);

        Pooling2DConfig poolConfig = Pooling2DConfig.builder().kH(2).kW(2).sH(2).sW(2).build();

        Conv2DConfig convConfig = Conv2DConfig.builder().kH(3).kW(3).build();


        for (int i = start; i <= end; i++) {
            String name = "inter_" + i;
            if (i == end) {
                name = "output";
            }
            switch (i) {
                case 0:
                    SDVariable w0 = sd.var("w0", new XavierInitScheme('c', 28 * 28, 26 * 26 * 4), DataType.FLOAT, 3, 3, 1, 4);
                    SDVariable b0 = sd.zero("b0", 4);

                    inReshaped = sd.reshape("reshapedInput", inReshaped, -1, 1, 28, 28);
                    // inReshaped = inReshaped.reshape(-1, 1, 28, 28);
                    inReshaped = sd.cnn().conv2d(name, inReshaped, w0, b0, convConfig);
                    break;
                case 1:
                    SDVariable pool1 = sd.cnn().maxPooling2d(inReshaped, poolConfig);

                    inReshaped = sd.nn().relu(name, pool1, 0);
                    break;
                case 2:
                    SDVariable w1 = sd.var("w1", new XavierInitScheme('c', 13 * 13 * 4, 11 * 11 * 8), DataType.FLOAT, 3, 3, 4, 8);
                    SDVariable b1 = sd.zero("b1", 8);

                    inReshaped = sd.cnn().conv2d(name, inReshaped, w1, b1, convConfig);
                    break;
                case 3:
                    SDVariable pool2 = sd.cnn().maxPooling2d(inReshaped, poolConfig);
                    inReshaped = sd.nn().relu(name, pool2, 0);
                    break;
                case 4:
                    int nOut = 10;
                    SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, nOut);

                    SDVariable wOut = sd.var("wOut", new XavierInitScheme('c', 5 * 5 * 8, 10), DataType.FLOAT, 5 * 5 * 8, 10);
                    SDVariable bOut = sd.zero("bOut", 10);

                    inReshaped = sd.reshape("reshapedInput", inReshaped, -1, 5 * 5 * 8);
                    //inReshaped = inReshaped.reshape(-1, 5 * 5 * 8);
                    SDVariable z = sd.nn().linear("z", inReshaped, wOut, bOut);

                    // softmax crossentropy loss function
                    SDVariable out = sd.nn().softmax("output", z, 1);
                    SDVariable loss = sd.loss().softmaxCrossEntropy("loss", label, out, null);

                    sd.setLossVariables(loss);
                    break;
                default:
                    System.out.print("Unknown layer: " + i);
            }
        }
        this.subModel = sd;
        return sd;
    }

    /**
     * An external initialize training method extracted from the source code, in which it is a protected method,
     * it initializes the updater and return an updater map of trainable variables
     * return Map<String, GradientUpdater>
     */
    public Map<String, GradientUpdater> externalInitializeTraining() {
        SameDiff sd = this.subModel;
        if (!sd.isInitializedTraining()) {
            if (sd.getTrainingConfig() == null) {
                throw new ND4JIllegalStateException("No training config specified!");
            }

            updaterMap = new HashMap<String, GradientUpdater>();
            for (Variable v : sd.getVariables().values()) {
                if (v.getVariable().getVariableType() != VariableType.VARIABLE || !v.getVariable().dataType().isFPType()) {
                    //Skip non-trainable parameters
                    continue;
                }

                INDArray arr = v.getVariable().getArr();
                long stateSize = sd.getTrainingConfig().getUpdater().stateSize(arr.length());
                INDArray view = stateSize == 0 ? null : Nd4j.createUninitialized(arr.dataType(), 1, stateSize);
                GradientUpdater gu = sd.getTrainingConfig().getUpdater().instantiate(view, false);
                gu.setStateViewArray(view, arr.shape(), arr.ordering(), true);
                updaterMap.put(v.getName(), gu);
            }
            return updaterMap;
        }
        return null;
    }

    /**
     * Similar to the step() in Pytorch, which updates the weights according to the gradient
     */
    public void step() {
        Set<String> paramsToTrain = new LinkedHashSet<String>();
        Map<String, String> gradVarToVarMap = new HashMap<String, String>(); // 梯度变量和变量名字之间的映射
        for (Variable v : subModel.getVariables().values()) {
            if (v.getVariable().getVariableType() == VariableType.VARIABLE) {
                paramsToTrain.add(v.getName());
            }
        }
        // TODO: TrainingSession 里面的 reg 相关的代码要不要加
        for (String s : paramsToTrain) {
            SDVariable grad = this.subModel.getVariable(s).getGradient();
            INDArray paramArr = this.subModel.getVariable(s).getArr();
            INDArray gradArr = grad.getArr();
            if (grad == null) {
                continue;
            }
            GradientUpdater u = updaterMap.get(s);
            u.applyUpdater(gradArr, this.iteration, this.epoch); // in-place update

            if (this.subModel.getTrainingConfig().isMinimize()) {
                paramArr.subi(gradArr);
            } else {
                paramArr.addi(gradArr);
            }
        }
    }
}
