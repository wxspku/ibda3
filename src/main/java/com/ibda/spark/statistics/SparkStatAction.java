package com.ibda.spark.statistics;

import com.ibda.spark.util.SparkUtil;
import org.apache.spark.ml.stat.Correlation;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public abstract class SparkStatAction {

    public static final String FEATURES_VECTOR_COL = "features_vector_col";

    /**
     * 在数据列平铺数据集上执行统计动作
     * @param dataset  平铺数据集
     * @param featureColumns  需要处理的数据列列表
     * @param specialColumn 特殊数据列：假设检验的分类标签数据列，描述性统计的权重列
     * @return
     */
    public  Dataset<Row> doAction(Dataset<Row> dataset, String[] featureColumns, String specialColumn){
        Dataset<Row> vectorDataset = SparkUtil.assembleVector(dataset,featureColumns, FEATURES_VECTOR_COL);
        return doAction(vectorDataset,FEATURES_VECTOR_COL,specialColumn);
    }
    /**
     * 统计动作抽象方法，返回数据集
     * @param dataset
     * @param featureVectorColumn
     * @param specialColumn
     * @return
     */
    public  abstract Dataset<Row> doAction(Dataset<Row> dataset, String featureVectorColumn, String specialColumn);

    /**
     * 计算相关系数矩阵
     */
    public static final SparkStatAction CorrelationAction = new SparkStatAction(){

        @Override
        public Dataset<Row> doAction(Dataset<Row> dataset, String featureVectorColumn, String method) {
            return Correlation.corr(dataset, featureVectorColumn,method);
        }
    };
}
