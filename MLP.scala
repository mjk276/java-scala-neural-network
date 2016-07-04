//package com.example.neuralnetwork

import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction


object MLP {

  def main(args: Array[String]) {
  
    val numRows = 28
    val numColumns = 28
    val outputNum = 10
    val batchSize = 128
    val rngSeed = 123
    val numEpochs = 15
    val mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed)
    val mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed)
    
    // setup the conf object that defines the model
    val conf = new NeuralNetConfiguration.Builder().seed(rngSeed).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .learningRate(0.006)
      .updater(Updater.NESTEROVS)
      .momentum(0.9)
      .regularization(true)
      .l2(1e-4)
      .list()
      .layer(0, new DenseLayer.Builder().nIn(numRows * numColumns).nOut(1000)
      .activation("relu")
      .weightInit(WeightInit.XAVIER)
      .build())
      .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
      .nIn(1000)
      .nOut(outputNum)
      .activation("softmax")
      .weightInit(WeightInit.XAVIER)
      .build())
      .pretrain(false)
      .backprop(true)
      .build()
      
    // initializing the model with the conf object above  
    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(1))

    // train the model with MNIST data for 15 Epochs
    for (i <- 0 until numEpochs) {
      model.fit(mnistTrain)
    }
    
    // quick eval of data
    val eval = new Evaluation(outputNum)
    while (mnistTest.hasNext) {
      val next = mnistTest.next()
      val output = model.output(next.getFeatureMatrix)
      eval.eval(next.getLabels, output)
    }
    
    println(eval.stats())
    println("**************** Done ********************")
  }
}
