package com.example.ftpipehd_mobile;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import com.example.ftpipehd_mobile.model.MNISTCNN;
import com.example.ftpipehd_mobile.utils.General;

import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.common.resources.ResourceType;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.autodiff.listeners.impl.ScoreListener;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;

import org.nd4j.autodiff.util.SameDiffUtils;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.ExternalErrorsFunction;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.weightinit.impl.XavierInitScheme;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MainActivity extends AppCompatActivity {
    private Thread trainThread;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        DL4JResources.setBaseDirectory(new File(getExternalFilesDir(null), ""));

        this.trainThread = new Thread(() -> {
            TextView msg = findViewById(R.id.train_message);
            msg.setText("Train Start");
            try {
                train();
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
        trainThread.start();
    }

    private void train() throws Exception {
        MNISTCNN testSubModel = new MNISTCNN();
        MNISTCNN testSubModel2 = new MNISTCNN();
        MNISTCNN testSubModel3 = new MNISTCNN();
        SameDiff model = testSubModel.makeMNISTNet();
        SameDiff subModel = testSubModel.simpleMakeSubModel(0, 1);
        SameDiff subModel2 = testSubModel2.simpleMakeSubModel(2, 3);
        SameDiff subModel3 = testSubModel3.simpleMakeSubModel(4, 4);

        int batchSize = 256;
        //File file = new File(getExternalFilesDir(null), "");

        DataSetIterator trainData = new MnistDataSetIterator(batchSize, true, 12345);

        int iteration = 1;
        int epoch = 3;

        //Create and set the training configuration
        double learningRate = 1e-3;
        TrainingConfig config = new TrainingConfig.Builder()
                .l2(1e-4)                               //L2 regularization
                .updater(new Adam(learningRate))        //Adam optimizer with specified learning rate
                .dataSetFeatureMapping("input")         //DataSet features array should be associated with variable "input"
                .dataSetLabelMapping("label")           //DataSet label array should be associated with variable "label"
                .build();
        model.setTrainingConfig(config);

        subModel.setTrainingConfig(config);
        subModel2.setTrainingConfig(config);
        subModel3.setTrainingConfig(config);

        Map<String, GradientUpdater> gu1 = testSubModel.externalInitializeTraining();
        Map<String, GradientUpdater> gu2 = testSubModel2.externalInitializeTraining();
        Map<String, GradientUpdater> gu3 = testSubModel3.externalInitializeTraining();

        int cnt = 0;
        int totalCorrectNum = 0;
        int totalNum = 0;
        Log.e("MainActivityTrain", "Start Training ...");
        for (int i = 0; i < epoch; i++) {
            while (trainData.hasNext()) {
                DataSet curData = trainData.next();
                long curBatchSize = curData.getFeatures().shape()[0];
                long[] curInputShape = subModel.getVariable("input").getArr().shape();
                if (curBatchSize != curInputShape[0]) {
                    continue;
                }

                // subModel 1
                INDArray input = curData.getFeatures();
                INDArray label = curData.getLabels();

                subModel.getVariable("input").setArray(input);
                INDArray output = subModel.getVariable("output").eval();

                // subModel 2
                subModel2.getVariable("input").setArray(output);
                INDArray output2 = subModel2.getVariable("output").eval();

                // subModel 3;
                subModel3.getVariable("label").setArray(label);
                subModel3.getVariable("input").setArray(output2);
                INDArray loss = subModel3.getVariable("loss").eval();

                // get correctNum
                int correctNum = General.getCorrectNum(subModel3.getVariable("output").getArr(), subModel3.getVariable("label").getArr());
                totalCorrectNum += correctNum;
                totalNum += curBatchSize;

                Log.e("MainActivityTrain", "correctNum: " + correctNum);

                // backward test
                Map<String, INDArray> grads3 = subModel3.calculateGradients(null, subModel3.getVariables().keySet());
                testSubModel3.step();
                Log.e("MainActivityTrain", "subModel3 backward finish ");

                ExternalErrorsFunction fn2 = SameDiffUtils.externalErrors(subModel2, null, subModel2.getVariable("output"));
                INDArray externalGrad2 = grads3.get("reshapedInput").reshape(-1, 8, 5, 5);
                Map<String, INDArray> externalGradMap2 = new HashMap<String, INDArray>();
                externalGradMap2.put("output-grad", externalGrad2);
                Map<String, INDArray> grads2 = subModel2.calculateGradients(externalGradMap2, subModel2.getVariables().keySet());
                testSubModel2.step();
                Log.e("MainActivityTrain", "subModel2 backward finish ");

                ExternalErrorsFunction fn1 = SameDiffUtils.externalErrors(subModel, null, subModel.getVariable("output"));
                INDArray externalGrad = grads2.get("input");
                Map<String, INDArray> externalGradMap = new HashMap<String, INDArray>();
                externalGradMap.put("output-grad", externalGrad);
                Map<String, INDArray> grad = subModel.calculateGradients(externalGradMap, subModel.getVariables().keySet());
                testSubModel.step();


                Log.e("MainActivityTrain", "Batch " + cnt + " Loss: " + loss.getFloat());
                cnt++;
            }
            Log.e("MainActivityTrain", "Epoch " + i + " Acc: " + totalCorrectNum * 1.0 / totalNum * 100 + "%");
            totalCorrectNum = 0;
            totalNum = 0;
            cnt = 0;
            trainData.reset();
        }
    }
}