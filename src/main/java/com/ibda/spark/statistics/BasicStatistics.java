package com.ibda.spark.statistics;

import com.ibda.spark.SparkAnalysis;
import org.apache.spark.ml.linalg.DenseMatrix;
import org.apache.spark.ml.stat.Summarizer;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema;

import java.util.Arrays;
import java.util.stream.Collectors;

public class BasicStatistics extends SparkAnalysis {

    public static final String FEATURES_VECTOR_COL = "features_vector_col";


    public BasicStatistics(String appName) {
        super(appName);
    }

    /**
     * 计算数据集指定字段的描述性统计结果
     * @param dataset
     * @param featureColumns   数据列名称表
     * @param weightColumn     权重列 
     * @param targets
     * @return
     */
    public DescriptiveStatistics getDescriptiveStatistics(Dataset<Row> dataset, String[] featureColumns, String weightColumn, DescriptiveTarget[] targets){
        Dataset<Row> vectorDataset = assembleVector(dataset,featureColumns, FEATURES_VECTOR_COL);
        return getDescriptiveStatistics(vectorDataset,FEATURES_VECTOR_COL,weightColumn , targets);
    }

    /**
     * 计算数据集指定字段的描述性统计结果
     * @param dataset
     * @param featureVectorColumn  组装成VectorUDT的数据列
     * @param weightColumn
     * @param targets
     * @return
     */
    public DescriptiveStatistics getDescriptiveStatistics(Dataset<Row> dataset, String featureVectorColumn, String weightColumn, DescriptiveTarget[] targets){
        String[] metrics = new String[targets.length];
        Arrays.stream(targets).map(target->target.toString()).collect(Collectors.toList()).toArray(metrics);
        Dataset<Row> summary = null;
        if (weightColumn == null){
            summary = dataset.select(Summarizer.metrics(metrics).summary(new Column(featureVectorColumn)));
        }
        else{
            summary = dataset.select(Summarizer.metrics(metrics).summary(new Column(featureVectorColumn), new Column(weightColumn)));
        }
        //两层向量[[统计指标1结果，统计指标2结果，...]],除count外，每个结果又是一个向量,
        // [[30,[35.0,61.0,60.0],[99.0,100.0,99.0],[75.21666666666667,81.93333333333334,81.86666666666667],[2256.5,2458.0,2456.0]]]
        Row row = summary.head();
        //解耦Row首层
        GenericRowWithSchema genericRow = (GenericRowWithSchema)row.get(0);
        DescriptiveStatistics statistics = DescriptiveStatistics.buildStatistics(genericRow, metrics);

        /*for (int i=0;i<genericRow.size();i++){
            Object object = genericRow.get(i);
            System.out.println(metrics[i] + "[" + object.getClass() + "]:" + object.toString());
        }
        */
        return statistics;
    }

    /**
     * 计算数据集指定字段之间的pearson相关系数矩阵
     * @param dataset
     * @param featureColumns
     * @return
     */
    public DenseMatrix getCorrelationMatrix(Dataset<Row> dataset,String[] featureColumns){
        return getCorrelationMatrix(dataset,featureColumns,CorrelationMethod.pearson);
    }

    /**
     * 计算数据集指定字段之间的相关系数矩阵
     * @param dataset
     * @param featureColumns
     * @param method
     * @return
     */
    public DenseMatrix getCorrelationMatrix(Dataset<Row> dataset,String[] featureColumns,CorrelationMethod method){
        Dataset<Row> vectorDataset = assembleVector(dataset,featureColumns, FEATURES_VECTOR_COL);
        return getCorrelationMatrix(vectorDataset,FEATURES_VECTOR_COL,method);
    }

    /**
     *
     * @param dataset
     * @param featureVectorColumn
     * @param method
     * @return
     */
    public DenseMatrix getCorrelationMatrix(Dataset<Row> dataset,String featureVectorColumn,CorrelationMethod method){
        Dataset<Row> corr = SparkStatAction.CorrelationAction.doAction(dataset,featureVectorColumn,method.toString());
        //DenseMatrix
        DenseMatrix matrix = (DenseMatrix) corr.head().get(0);
        return matrix;
    }
    /**
     * 获取指定字段正态分布检验结果，自行计算均值和标准差
     * @param dataset
     * @param columnName
     * @return
     */
    public HypothesisTestResults normalDistTest(Dataset<Row> dataset, String columnName) {
        return HypothesisTestAction.KSTestAction.doTest(dataset,columnName,null);
    }



    /**
     * 卡方分布检验，检验每个分类属性是否与标签属性独立，在逻辑回归前可以用于确定自变量
     * 每一行数据为一个样本，记录的是该样本的各个分类属性值，样本分label列和features列，features列中的每个列均为分类属性，
     * 每个单独的取值（可能为实数）均视为一个分类，实际应用中，定距或定比数据需要进行离散化处理（QuantileDiscretizer/Bucketizer）
     * 将label列、features中的每一个列进行一次多项卡方检验，基本过程如下：
     * 1：计算label列、当前features列的每个取值组合的样本数，得到分类的实际频数行列表
     * 2：计算每个取值组合期望频数：行和*列和/总数
     * 3：计算卡方统计量及伴随概率p
     * 4：比较：卡方统计量＞临界值（或者：伴随概率p＜显著性水平α），则分类变量不独立
     * @param dataset
     * @param featureColumns
     * @param labelColumn
     * @return
     */
    public HypothesisTestResults chiSquareTest(Dataset<Row> dataset,String[] featureColumns,String labelColumn){
        return HypothesisTestAction.ChiSquareTestAction.doTest(dataset,featureColumns,labelColumn);
    }

    /**
     *
     * @param dataset
     * @param featureVectorColumn
     * @param labelColumn
     * @return
     */
    public HypothesisTestResults chiSquareTest(Dataset<Row> dataset,String featureVectorColumn,String labelColumn){
        return HypothesisTestAction.ChiSquareTestAction.doTest(dataset,featureVectorColumn,labelColumn);
    }
    /**
     * FValue检验
     * @param dataset
     * @param featureColumns
     * @param labelColumn
     * @return
     */
    public HypothesisTestResults FValueTest(Dataset<Row> dataset,String[] featureColumns,String labelColumn){
        return HypothesisTestAction.FValueTestAction.doTest(dataset,featureColumns,labelColumn);
    }

    public HypothesisTestResults FValueTest(Dataset<Row> dataset,String featureVectorColumn,String labelColumn){
        return HypothesisTestAction.FValueTestAction.doTest(dataset,featureVectorColumn,labelColumn);
    }
    /**
     * 单因素方差分析测试
     * @param dataset
     * @param featureColumns
     * @param labelColumn
     * @return
     */
    public HypothesisTestResults ANOVATest(Dataset<Row> dataset,String[] featureColumns,String labelColumn){
        return HypothesisTestAction.ANOVATestAction.doTest(dataset,featureColumns,labelColumn);
    }

    /**
     *
     * @param dataset
     * @param featureVectorColumn
     * @param labelColumn
     * @return
     */
    public HypothesisTestResults ANOVATest(Dataset<Row> dataset,String featureVectorColumn,String labelColumn){
        return HypothesisTestAction.ANOVATestAction.doTest(dataset,featureVectorColumn,labelColumn);
    }


}