package io.sugo.bigdl

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.keras.{Dense, Sequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, Shape}
import org.apache.spark.{SparkConf, SparkContext}

object IrisExample {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("IrisExample")
    val sc = new SparkContext(Engine.createSparkConf(sparkConf))
    Engine.init
    val sampleRDD = sc.textFile("data/iris.data", 1).filter(!"".equals(_)).map(line => {
      val subs = line.split(",") // "," may exist in content.
      val feature = Tensor(subs.slice(0, 4).map(_.toFloat), Array(4))
      val getLabel: String => Float = {
        case "Iris-setosa" => 1.0f
        case "Iris-versicolor" => 2.0f
        case "Iris-virginica" => 3.0f
      }
      Sample[Float](feature, Tensor(Array(getLabel(subs(4))), Array(1)))
    })

    val Array(trainingRDD, valRDD) = sampleRDD.randomSplit(
      Array(0.9, 0.1))

    val model = Sequential[Float]()
    model.add(Dense(40, inputShape = Shape(4), activation = "relu"))
    model.add(Dense(20, activation = "relu"))
    model.add(Dense(3, activation = "softmax"))
    model.compile("adam", "sparse_categorical_crossentropy", Array("accuracy"))

    model.fit(trainingRDD, batchSize=50, nbEpoch=50, validationData = valRDD)

    model.predict(sampleRDD, 50)
  }

}

