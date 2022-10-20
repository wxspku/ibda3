package com.ibda.spark;

import com.ibda.spark.util.SparkUtil;
import com.ibda.util.AnalysisConst;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Map;
import java.util.UUID;

public class SparkAnalysis extends AnalysisConst {
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
     * 离群点、缺失值处理方法
     */
    public enum ProcessMethod{
        REMOVE,             //简单去除样本或变量，离群或缺失样本比例较低，或该变量的样本缺失值过多
        REPLACE_WITH_MEAN,  //使用均值、中位数、众数、离群临界值、最邻近正常值替换，仅用于数值类型
        REPLACE_WITH_MEDIAN,
        REPLACE_WITH_MODE,
        REPLACE_WITH_THRESHOLD,
        REPLACE_WITH_NEAREST
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

    /**
     *
     * @param path
     * @return
     */
    public Dataset<Row> loadData(String path){
        return loadData(path,null);
    }

    /**
     * 简版读取文件数据，如需完整的参数控制
     * @param path
     * @param ddl
     * @return
     */
    public Dataset<Row> loadData(String path, String ddl){
        return SparkUtil.loadData(spark,path,ddl);
    }

    /**
     *
     * @param path
     * @param format
     * @param options
     * @param ddl
     * @return
     */
    public Dataset<Row> loadData(String path, String format, Map<String,String> options, String ddl){
        return SparkUtil.loadData(spark,path,format,options,ddl);
    }
    /**
     * 读取Excel数据
     * @param path
     * @param sheetIndex
     * @param range
     * @param ddl
     * @return
     */
    public Dataset<Row> loadExcel(String path, int sheetIndex, String range, String ddl){
        return SparkUtil.loadExcel(spark,path,ddl,sheetIndex,range);
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

    /**
     * 清除重复记录操作
     * @param source
     * @return
     */
    public static Dataset<Row> removeDuplicates(Dataset<Row> source){
        return null;
    }

    /**
     * 处理离群点
     * @param source
     * @param method
     * @return
     */
    public static Dataset<Row> processOutlier(Dataset<Row> source,ProcessMethod method){
        return null;
    }

    /**
     * 处理缺失值
     * @param source
     * @param method
     * @return
     */
    public static Dataset<Row> processMissing(Dataset<Row> source,ProcessMethod method){
        return null;
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        spark.stop();
    }


}
