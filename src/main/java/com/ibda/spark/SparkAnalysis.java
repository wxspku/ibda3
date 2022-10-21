package com.ibda.spark;

import com.ibda.spark.util.SparkUtil;
import com.ibda.util.AnalysisConst;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Map;
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
    public static Dataset<Row> assembleVector(Dataset<Row> source, String[] inputNames, String outputName) {
        return SparkUtil.assembleVector(source,inputNames,outputName);
    }

    /**
     * TODO 清除重复记录操作
     * @param source   原始数据集
     * @param columns  需处理的列表，即判断重复样本时不考虑列表之外的数据列，null代表所有列
     * @return
     */
    public static Dataset<Row> removeDuplicates(Dataset<Row> source, String[] columns){
        return source;
    }

    /**
     * 处理离群点,低于SAMPLE_REMOVE_THRESHOLD(0.05)比例时，离群点样本删除
     * @param source
     * @param columns
     * @param method
     * @return
     */
    public static Dataset<Row> processOutlier(Dataset<Row> source, String[] columns, OutlierProcessMethod method){
        return processOutlier(source,columns,method, OUTLIER_SAMPLE_REMOVE_THRESHOLD);
    }
    /**
     * TODO 处理离群点,原则上离群点记录数只占样本数的数据较低时，可以简单去除，否则可以使用替代法,如果不需要删除，则将阈值直接设为0
     * @param source         原始数据集
     * @param columns        需处理的字段列表，只考虑本列表字段的离群点，null代表所有列
     * @param method         离群点比例超出truncateRatio时，离群点的处理方法
     * @param removeRatioThreshold  样本删除比例阈值，离群点低于该样本比例时，直接删除离群点，离群点的比例需考虑关注字段的所有离群点比例
     * @return
     */
    public static Dataset<Row> processOutlier(Dataset<Row> source, String[] columns, OutlierProcessMethod method, double removeRatioThreshold){
        return source;
    }

    /**
     * 使用缺省阈值处理缺失值
     * @param source
     * @param columns
     * @param method
     * @return
     */
    public static Dataset<Row> processMissing(Dataset<Row> source,String[] columns, OutlierProcessMethod method){
        return processMissing(source,columns,method, OUTLIER_SAMPLE_REMOVE_THRESHOLD, MISSING_VALUE_FEATURE_REMOVE_THRESHOLD);
    }

    /**
     * TODO 处理缺失值，样本缺失值可以参照离群点处理，包含大量缺失值的字段，需要删除,如果不需要删除，则将前一个阈值设为0，后一个阈值设为1
     * @param source
     * @param columns        需处理的字段列表，只考虑本列表字段的离群点，null代表所有列
     * @param method
     * @param sampleRemoveRatioThreshold
     * @param featureRemoveRatioThreshold
     * @return
     */
    public static Dataset<Row> processMissing(Dataset<Row> source, String[] columns, OutlierProcessMethod method, double sampleRemoveRatioThreshold, double featureRemoveRatioThreshold){
        return source;
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        spark.stop();
    }


}
