package com.databricks.spark.sql.perf.mllib.feature

import com.databricks.spark.sql.perf.mllib.{BenchmarkAlgorithm, MLBenchContext, TestFromTraining}
import com.databricks.spark.sql.perf.mllib.data.DataGenerator
import org.apache.spark.ml
import org.apache.spark.ml.PipelineStage
import org.apache.spark.sql.DataFrame

/** Object for testing StandardScaler performance */
object StandardScaler extends BenchmarkAlgorithm with TestFromTraining with UnaryTransformer {

  override def trainingDataSet(ctx: MLBenchContext): DataFrame = {
    import ctx.params._
    import ctx.sqlContext.implicits._

    DataGenerator.generateContinuousFeatures(
      ctx.sqlContext,
      numExamples.get,
      ctx.seed(),
      numPartitions.get,
      numFeatures.get).toDF(inputCol)

  }

  override def getPipelineStage(ctx: MLBenchContext): PipelineStage = {
    import ctx.params._
    import ctx.sqlContext.implicits._

    new ml.feature.StandardScaler()
      .setWithMean(true)
      .setWithStd(true)
      .setInputCol(inputCol)
  }
}
