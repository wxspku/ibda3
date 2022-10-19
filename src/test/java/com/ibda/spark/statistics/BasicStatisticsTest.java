package com.ibda.spark.statistics;

import com.ibda.spark.SparkAnalysis;
import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.linalg.DenseMatrix;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.stat.Summarizer;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static com.ibda.spark.SparkAnalysis.DESCRIPTIVE_TARGET_ALL;

public class BasicStatisticsTest {

    static BasicStatistics stat = null;
    static{
        stat = new BasicStatistics("test for BasicStatistics");
    }

    /**
     * 使用KolmogorovSmirnovTest检测是否符合正态分布
     * @return
     * @throws AnalysisException
     */
    private boolean isNormalDistribution()  {
        String path = FilePathUtil.getClassRoot() + "/data/stattest.xlsx";
        Dataset<Row> dataset = stat.readExcel( path, 0, "A2",
                "id INT NOT NULL,score DOUBLE NOT NULL");
        HypothesisTestResults results = stat.normalDistTest(dataset, "score");
        System.out.println(results);
        return results.nullHypothesisAccepted(0);
    }

    private void sparkSummarizeDemo(){
        List<Row> data = Arrays.asList(
                RowFactory.create(Vectors.dense(2.0, 4.5, 5.0), 1.0),
                RowFactory.create(Vectors.dense(4.0, 6.0, 8.0), 2.0)
        );

        StructType schema = new StructType(new StructField[]{
                new StructField("features", new VectorUDT(), false, Metadata.empty()),
                new StructField("weight", DataTypes.DoubleType, false, Metadata.empty())
        });

        Dataset<Row> df = stat.getSpark().createDataFrame(data, schema);

        Row result1 = df.select(Summarizer.metrics("mean", "variance","normL1","normL2","sum")
                        .summary(new Column("features"), new Column("weight")).as("summary"))
                .select("summary.mean", "summary.variance","summary.normL1", "summary.normL2","summary.sum").first();
        System.out.println("with weight: mean = " + result1.<Vector>getAs(0).toString() +
                ", variance = " + result1.<Vector>getAs(1).toString()+
                ", normL1 = " + result1.<Vector>getAs(2).toString() +
                ", normL2 = " + result1.<Vector>getAs(3).toString() +
                ", sum = " + result1.<Vector>getAs(4).toString());
        //mean = [3.333333333333333,5.5,7.0], variance = [2.000000000000001,1.125,4.5],
        //normL1 = [10.0,16.5,21.0], normL2 = [6.0,9.604686356149273,12.36931687685298], sum = [10.0,16.5,21.0]
        Row result2 = df.select(
                Summarizer.mean(new Column("features")),
                Summarizer.variance(new Column("features"))
        ).first();
        System.out.println("without weight: mean = " + result2.<Vector>getAs(0).toString() +
                ", variance = " + result2.<Vector>getAs(1).toString());
    }

    private void descriptiveStatisticsDemo(){
        /*"chi_score DOUBLE NOT NULL," +
                "math_score DOUBLE NOT NULL," +
                "eng_score DOUBLE NOT NULL");*/
        Dataset<Row> dataset = getExcelDataset();
        DescriptiveStatistics statistics = stat.getDescriptiveStatistics(dataset, new String[]{"chi_score", "math_score", "eng_score"}, null, DESCRIPTIVE_TARGET_ALL);
        System.out.println("描述性统计结果:" + statistics);
    }

    private void calcDescriptiveStatistics() throws AnalysisException {
        String path = FilePathUtil.getClassRoot() + "/data/stattest.xlsx";
        Dataset<Row> dataset = stat.readExcel(path, 0, "A2",
                "id INT NOT NULL,score DOUBLE NOT NULL");
        //使用sql查询汇总数据
        dataset.createOrReplaceTempView("scores");
        Dataset<Row> summary = dataset.sparkSession().sql("select count(*) _count,min(score) _min,max(score) _max,mean(score) _mean,std(score) _std from scores");
        summary.show();
        /*单列描述性统计
        +-------+-----------------+
                |summary|            score|
                +-------+-----------------+
                |  count|               30|
                |   mean|75.21666666666667|
                | stddev|14.78097760131496|
                |    min|             35.0|
                |    max|             99.0|
                +-------+-----------------+
         */
        Dataset<Row> score_summary = dataset.describe("score");
        score_summary.show();
        //colum to vector, then Summarizer vector
        Dataset<Row> vectorDataset = SparkAnalysis.transVectorColumns(dataset, new String[]{"score"}, "score_vector");
        vectorDataset.show();
        //向量的normL1 = sum , normL2 = sqrt(Σx^2*w)
        Row result1 = vectorDataset.select(Summarizer.metrics("mean", "variance","normL1","normL2","sum")
                        .summary(new Column("score_vector")).as("summary"))
                .select("summary.mean", "summary.variance","summary.normL1", "summary.normL2","summary.sum").first();
        System.out.println("with weight: mean = " + result1.<Vector>getAs(0).toString() +
                ", variance = " + result1.<Vector>getAs(1).toString() +
                ", normL1 = " + result1.<Vector>getAs(2).toString() +
                ", normL2 = " + result1.<Vector>getAs(3).toString() +
                ", sum = " + result1.<Vector>getAs(4).toString());

        Row result2 = vectorDataset.select(
                Summarizer.mean(new Column("score_vector")),
                Summarizer.variance(new Column("score_vector"))
        ).first();
        System.out.println("without weight: mean = " + result2.<Vector>getAs(0).toString() +
                ", variance = " + result2.<Vector>getAs(1).toString());
    }

    private void calcCorrelation(){
        Dataset<Row> dataset = getExcelDataset();
        DenseMatrix pearson = stat.getCorrelationMatrix(dataset, new String[]{"chi_score", "math_score", "eng_score"});
        System.out.println("Pearson correlation matrix:\n" + pearson.toString());

        DenseMatrix spearman = stat.getCorrelationMatrix(dataset, new String[]{"chi_score", "math_score", "eng_score"}, SparkAnalysis.CorrelationMethod.spearman);
        System.out.println("Spearman correlation matrix:\n" + spearman.toString());

    }

    private Dataset<Row> getExcelDataset() {
        String path = FilePathUtil.getClassRoot() + "/data/stattest.xlsx";
        Dataset<Row> dataset = stat.readExcel(path, 4, "A2:D31",
                "id INT NOT NULL," +
                        "chi_score DOUBLE NOT NULL," +
                        "math_score DOUBLE NOT NULL," +
                        "eng_score DOUBLE NOT NULL");
        return dataset;
    }

    /**
     * 卡方分布检验示例，检验每个分类属性是否与标签属性独立，在逻辑回归前可以用于确定自变量
     * 每一行数据为一个样本，记录的是该样本的各个分类属性值，样本分label列和features列，features列中的每个列均为分类属性，
     * 每个单独的取值（可能为实数）均视为一个分类，实际应用中，定距或定比数据需要进行离散化处理（QuantileDiscretizer/Bucketizer）
     * 将label列、features中的每一个列进行一次多项卡方检验，基本过程如下：
     * 1：计算label列、当前features列的每个取值组合的样本数，得到分类的实际频数行列表
     * 2：计算每个取值组合期望频数：行和*列和/总数
     * 3：计算卡方统计量及伴随概率p
     * 4：比较：卡方统计量＞临界值（或者：伴随概率p＜显著性水平α），则分类变量不独立
     */
    private void ChiSquareTestDemo(){
        List<Row> data = Arrays.asList(
                RowFactory.create(0.0, Vectors.dense(0.5, 10.0)),
                RowFactory.create(0.0, Vectors.dense(1.5, 20.0)),
                RowFactory.create(1.0, Vectors.dense(1.5, 30.0)),
                RowFactory.create(0.0, Vectors.dense(3.5, 30.0)),
                RowFactory.create(0.0, Vectors.dense(3.5, 40.0)),
                RowFactory.create(1.0, Vectors.dense(3.5, 40.0))
        );

        StructType schema = new StructType(new StructField[]{
                new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("features", new VectorUDT(), false, Metadata.empty()),
        });

        Dataset<Row> df = stat.getSpark().createDataFrame(data, schema);
        HypothesisTestResults results = stat.chiSquareTest(df, "features", "label");
        System.out.println("chiSquareTest results:" + results);

    }

    private void FValueAndANOVATestDemo(){
        String path = FilePathUtil.getClassRoot() + "data/anova_dvdplayers.csv";
        Dataset<Row> dataset = stat.readFile(path, null);
        dataset.show();
        Dataset<Row> dvdscore_vector = SparkAnalysis.transVectorColumns(dataset, new String[]{"dvdscore"}, "dvdscore_vector");
        dvdscore_vector.show();

        System.out.println("FValue Test --------------------------");
        //pValues: Vector - degreesOfFreedom: Array[Long] - fValues: Vector Each of these fields has one value per feature.
        HypothesisTestResults results = stat.FValueTest(dataset, new String[]{"dvdscore"}, "agegroup");
        System.out.println(results);


        System.out.println("Anova Test --------------------------");
        //pValues: Vector - degreesOfFreedom: Array[Long] - fValues: Vector Each of these fields has one value per feature.
        HypothesisTestResults results2 = stat.ANOVATest(dvdscore_vector, "dvdscore_vector", "agegroup");
        System.out.println(results2);
    }

    @Test
    public  void testBasicStat() throws AnalysisException {
        sparkSummarizeDemo();
        descriptiveStatisticsDemo();
        ChiSquareTestDemo();
        FValueAndANOVATestDemo();
        System.out.println("正态性检验：" + isNormalDistribution());
        calcCorrelation();

    }

}