package com.ibda.spark;

import com.ibda.spark.util.SparkUtil;
import com.ibda.util.AnalysisConst;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.UUID;

public class SparkAnalysis extends AnalysisConst {


    protected SparkSession spark = null;

    public SparkSession getSpark() {
        return spark;
    }

    protected SparkAnalysis(String appName){
        if (appName == null){
            appName = this.getClass().getSimpleName() + "_" + UUID.randomUUID();
        }
        this.spark = SparkUtil.buildSparkSession(appName);
    }

    public Dataset<Row> readExcel(String path){
        return readExcel(path,0,null,null);
    }

    public Dataset<Row> readExcel(String path, int sheetIndex, String ddl){
        return readExcel(path,sheetIndex,null,ddl);
    }

    public Dataset<Row> readExcel(String path, int sheetIndex, String range, String ddl){
        return SparkUtil.sparkExcelRead(spark,path,sheetIndex,range,ddl);
    }

    public Dataset<Row> readFile(String path, String ddl){
        return SparkUtil.sparkReadFile(spark,path,ddl);
    }
    /**
     *
     * @param source
     * @param inputNames
     * @param outputName
     * @return
     */
    public static Dataset<Row> transVectorColumns(Dataset<Row> source, String[] inputNames, String outputName) {
        return SparkUtil.transVectorColumns(source,inputNames,outputName);
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        spark.stop();
    }

    /**
     * 描述性统计指标，不要改变拼写及大小写，名称需与DescriptiveStatistics的字段一致
     */
    public enum DescriptiveTarget{
        count,
        min,
        max,
        mean,
        normL1, //Σ(|x|)
        normL2, //sqrt(Σx^2*w)
        numNonZeros,
        std,    //样本标准差
        sum,
        variance //方差
    }

    /**
     * 相关矩阵计算方法
     */
    public enum CorrelationMethod
    {
        pearson, //default
        spearman
    }

    /**
     * 所有描述性统计指标
     */
    public static final DescriptiveTarget[] DESCRIPTIVE_TARGET_ALL = DescriptiveTarget.values();

    /**
     * 常用描述性统计指标
     */
    public static final DescriptiveTarget[] DESCRIPTIVE_TARGET_COMMON = new DescriptiveTarget[]{
            DescriptiveTarget.count,
            DescriptiveTarget.min,
            DescriptiveTarget.max,
            DescriptiveTarget.mean,
            DescriptiveTarget.std,
            DescriptiveTarget.sum
    };
}
