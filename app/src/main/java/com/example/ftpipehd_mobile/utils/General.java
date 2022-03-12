package com.example.ftpipehd_mobile.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class General {
    /**
     * This method reuses some codes in Evaluation.java in dl4j
     * @param predicted the prediction array
     * @param labels the labels
     * @return the correct number of the evaluation
     */
    public static int getCorrectNum(INDArray predicted, INDArray labels) {
        if (!Arrays.equals(predicted.shape(), labels.shape())) {
            throw new IllegalArgumentException("Unable to evaluate. Predictions and labels arrays are not same shape." +
                    " Predictions shape: " + Arrays.toString(predicted.shape()) + ", Labels shape: " + Arrays.toString(labels.shape()));
        }
        INDArray guessIndex = Nd4j.argMax(predicted, 1);
        INDArray groundTruthIndex = Nd4j.argMax(labels, 1);
        int ret = 0;

        int batchNum = (int) guessIndex.length();
        for (int i = 0; i < batchNum; i++) {
            int truthIdx = (int) guessIndex.getDouble(i);
            int predictIdx = (int) groundTruthIndex.getDouble(i);
            if (truthIdx == predictIdx) {
                ret++;
            }
        }
        return ret;
    }
}
