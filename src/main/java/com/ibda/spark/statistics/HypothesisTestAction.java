package com.ibda.spark.statistics;

import org.apache.spark.ml.stat.ANOVATest;
import org.apache.spark.ml.stat.ChiSquareTest;
import org.apache.spark.ml.stat.FValueTest;
import org.apache.spark.ml.stat.KolmogorovSmirnovTest;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public abstract class HypothesisTestAction extends SparkStatAction {

    public static final String NORM_DIST = "norm";

    /**
     * 在数据列平铺数据集上执行统计动作
     * @param dataset  平铺数据集
     * @param featureColumns  需要处理的数据列列表
     * @param labelColumn 分类标签数据列，没有时直接传空
     * @return
     */
    public HypothesisTestResults doTest(Dataset<Row> dataset, String[] featureColumns, String labelColumn){
        Row row = doAction(dataset,featureColumns,labelColumn).head();
        return HypothesisTestResults.buildTestResults(row);
    }

    /**
     * 在特征向量类型VectorUDT数据集上执行假设检验统计动作
     * @param dataset
     * @param featureVectorColumn   VectorUDT类型数据列
     * @param labelColumn
     * @return
     */
    public  HypothesisTestResults doTest(Dataset<Row> dataset, String featureVectorColumn, String labelColumn) {
        Row row = doAction(dataset,featureVectorColumn,labelColumn).head();
        return HypothesisTestResults.buildTestResults(row);
    }

    /**
     * 卡方检验
     */
    public static final HypothesisTestAction ChiSquareTestAction = new HypothesisTestAction(){

        @Override
        public Dataset<Row> doAction(Dataset<Row> dataset, String featureVectorColumn, String specialColumn) {
            return ChiSquareTest.test(dataset,featureVectorColumn, specialColumn);
        }
    };

    /**
     * KolmogorovSmirnov检验，用于数据字段的正态性检测
     */
    public static final HypothesisTestAction KSTestAction = new HypothesisTestAction(){

        @Override
        public Dataset<Row> doAction(Dataset<Row> dataset, String featureVectorColumn, String specialColumn) {
            final String view_name = "temp_view_ks_test";
            //获取均值和标准差
            SparkSession spark = dataset.sparkSession();
            dataset.createOrReplaceTempView(view_name);
            Dataset<Row> summary = spark.sql(String.format(
                    "select mean(%1$s) ,stddev_samp(%1$s)  from %2$s",featureVectorColumn,view_name));
            Row row = summary.head();
            //KS TEST
            double mean = row.getDouble(0);
            double std_dev = row.getDouble(1);

            return KolmogorovSmirnovTest.test(dataset, "score", NORM_DIST, mean, std_dev);
        }
    };

    /**
     * F检验
     */
    public static final HypothesisTestAction FValueTestAction = new HypothesisTestAction(){

        @Override
        public Dataset<Row> doAction(Dataset<Row> dataset, String featureVectorColumn, String specialColumn) {
            return FValueTest.test(dataset,featureVectorColumn,specialColumn);
        }
    };

    /**
     * 单因素方差检验
     */
    public static final HypothesisTestAction ANOVATestAction = new HypothesisTestAction(){

        @Override
        public Dataset<Row> doAction(Dataset<Row> dataset, String featureVectorColumn, String specialColumn) {
            return ANOVATest.test(dataset,featureVectorColumn,specialColumn);
        }
    };
}
