package com.ibda.spark.util;

import org.apache.commons.io.FilenameUtils;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.*;
import scala.collection.mutable.HashTable;

import java.util.HashMap;
import java.util.Map;

public class SparkUtil {
    /**
     * 特殊格式标示，后缀为txt，libsvm
     */
    public static final String LIBSVM_FORMAT = "libsvm";
    /**
     * 特殊格式标示，excel文件
     */
    public static final String EXCEL_FORMAT = "com.crealytics.spark.excel";
    /**
     * 特殊格式标示，图像文件，读取图像文件目录
     */
    public static final String IMAGE_FORMAT =  "image";

    /**
     * 根据文件后缀名读取特殊类型
     */
    private static final Map<String,String> FILE_EXT_FORMATS = new HashMap<String,String>();
    static{
        FILE_EXT_FORMATS.put("xlsx",EXCEL_FORMAT);
        FILE_EXT_FORMATS.put("xls",EXCEL_FORMAT);
        //FILE_EXT_FORMATS.put("jpg",IMAGE_FORMAT);
    }

    /**
     * CSV文件缺省读取选项
     */
    private static final Map<String, String> CSV_DEFAULT_OPTIONS = new HashMap<>();
    static
    {
        CSV_DEFAULT_OPTIONS.put("header","true");
        CSV_DEFAULT_OPTIONS.put("inferSchema","true");
        CSV_DEFAULT_OPTIONS.put("ignoreLeadingWhiteSpace","true");
        CSV_DEFAULT_OPTIONS.put("ignoreTrailingWhiteSpace","true");
    }

    /**
     * EXCEL文件缺省读取选项
     */
    private static final Map<String, String> EXCEL_DEFAULT_OPTIONS = new HashMap<>();
    static
    {
        EXCEL_DEFAULT_OPTIONS.put("header","true");
        EXCEL_DEFAULT_OPTIONS.put("inferSchema","true");
        EXCEL_DEFAULT_OPTIONS.put("dataAddress","0!A1");
    }

    private static final Map<String,Map<String,String>> FILE_EXT_OPTIONS = new HashMap<>();
    static{
        FILE_EXT_OPTIONS.put("xlsx",EXCEL_DEFAULT_OPTIONS);
        FILE_EXT_OPTIONS.put("xls",EXCEL_DEFAULT_OPTIONS);
        FILE_EXT_OPTIONS.put("csv",CSV_DEFAULT_OPTIONS);
    }

    /**
     * 构建SparkSession
     *
     * @param appName
     * @return
     *
     */
    public static SparkSession buildSparkSession(String appName) {
        // 添加spark运行必须的系统参数
        /*for (Map.Entry entry:System.getProperties().entrySet()){
            System.out.println(entry.getKey() + "=" + entry.getValue());
        }
        //-Dspark.master=local  -Dspark.submit.deployMode=client  -Dspark.executor.instances=3
        if (System.getProperty("spark.master") == null){
            System.out.println("未检测到spark.master参数，缺省使用local[*]-------------");
            System.setProperty("spark.master","local[*]");
        }
        if (System.getProperty("spark.submit.deployMode") == null){
            System.out.println("未检测到spark.submit.deployMode参数，缺省使用client-------------");
            System.setProperty("spark.submit.deployMode","client");
        }*/
        SparkSession spark = SparkSession
                .builder()
                .appName(appName)
                .getOrCreate();
        return spark;
    }

    /**
     *
     * @param spark
     * @param path
     * @return
     */
    public static Dataset<Row> loadData(SparkSession spark, String path){
        return loadData(spark, path, null, null, null);
    }

    /**
     *
     * @param spark
     * @param path
     * @param format
     * @return
     */
    public static Dataset<Row> loadData(SparkSession spark, String path, String format){
        return loadData(spark, path, format, null, null);
    }

    /**
     *
     * @param spark
     * @param path
     * @param format
     * @param options
     * @return
     */
    public static Dataset<Row> loadData(SparkSession spark, String path, String format, Map<String, String> options){
        return loadData(spark, path, format, options, null);
    }
    /**
     *
     * @param spark
     * @param path     文件路径
     * @param format   特殊格式标识，如不能通过文件扩展名进行推断，必须提供本参数
     * @param options  读取选项，根据文件格式不同而有所区别，Excel文件需要读取指定区域时使用key-value对("dataAddress","1!A2:D30")指定
     *                 也可以调用buildExcelOptions直接生成
     * @param ddl      字段定义
     * @return
     */
    public static Dataset<Row> loadData(SparkSession spark, String path, String format, Map<String, String> options, String ddl){
        DataFrameReader reader = buildReader(spark, path, format, ddl, options);
        return reader.load(path);
    }

    /**
     * 读取Excel首个sheet
     * @param spark
     * @param path
     * @return
     */
    public static Dataset<Row> loadExcel(SparkSession spark, String path){
        return loadData(spark, path, EXCEL_FORMAT);
    }

    /**
     * 读取Excel首个sheet，使用ddl定义
     * @param spark
     * @param path
     * @param ddl
     * @return
     */
    public static Dataset<Row> loadExcel(SparkSession spark, String path, String ddl){
        return loadData(spark, path, EXCEL_FORMAT,null,ddl);
    }

    /**
     * 读取Excel指定区域数据
     * @param spark
     * @param path
     * @param ddl
     * @param sheetIndex
     * @param range
     * @return
     */
    public static Dataset<Row> loadExcel(SparkSession spark, String path, String ddl, int sheetIndex, String range){
        Map<String, String> options = buildExcelOptions(sheetIndex,range);
        return loadData(spark, path, EXCEL_FORMAT, options, ddl);
    }

    /**
     * 获取Reader，便于自行添加特殊的读取选项
     * @param spark
     * @param path
     * @param format
     * @param ddl
     * @param options
     * @return
     */
    public static DataFrameReader buildReader(SparkSession spark, String path, String format, String ddl, Map<String, String> options) {
        DataFrameReader reader = spark.read();
        String extension = FilenameUtils.getExtension(path.toLowerCase());
        //格式为null时，根据扩展名推断文件格式，否则会按照缺省文件格式parquet处理
        if (format == null){
            format = FILE_EXT_FORMATS.getOrDefault(extension,extension);
        }
        if (format != null){
            reader.format(format);
        }
        Map<String,String> mergedOptions = new HashMap<>();
        if (FILE_EXT_OPTIONS.containsKey(extension)){
            mergedOptions.putAll(FILE_EXT_OPTIONS.get(extension));
        }
        if (options != null && !options.isEmpty()){
            mergedOptions.putAll(options);
        }
        reader.options(mergedOptions);
        if (ddl != null){
            reader.schema(ddl);
        }

        return reader;
    }

    /**
     *
     * @param source
     * @param inputNames
     * @param outputName
     * @return
     */
    public static Dataset<Row> assembleVector(Dataset<Row> source, String[] inputNames, String outputName) {
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(inputNames)//new String[]{"score"}
                .setOutputCol(outputName);//"score_vector"
        Dataset<Row> vectorDataset = assembler.transform(source);
        return vectorDataset;
    }

    /**
     * Excel设置指定sheet和区域选项
     * @param sheetIndex
     * @param range
     * @return
     */
    private static Map<String,String> buildExcelOptions(int sheetIndex,String range){
        Map<String,String> result = new HashMap<>(EXCEL_DEFAULT_OPTIONS);
        String dataAddress = (range == null) ? sheetIndex + "!A1" : sheetIndex + "!" + range;
        result.put("dataAddress", dataAddress);
        return result;
    }
}
