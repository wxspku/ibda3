package com.ibda.spark.util;

import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.types.DataTypes.DoubleType;
import static org.apache.spark.sql.types.DataTypes.IntegerType;

public class SparkUtil {

    private static final String SPARK_EXCEL_FORMAT = "com.crealytics.spark.excel";

    /**
     * 构建SparkSession
     *
     * @param appName
     * @return
     */
    public static SparkSession buildSparkSession(String appName) {
        // 添加spark运行必须的系统参数
        //-Dspark.master=local  -Dspark.submit.deployMode=client  -Dspark.executor.instances=3 */
        if (System.getProperty("spark.master") == null){
            System.setProperty("spark.master","local[4]");
        }
        if (System.getProperty("spark.submit.deployMode") == null){
            System.setProperty("spark.submit.deployMode","client");
        }
        SparkSession spark = SparkSession
                .builder()
                .appName(appName)
                .getOrCreate();
        return spark;
    }

    /**
     * 读取Excel首个Sheet,使用缺省的ddl
     * @param spark
     * @param path
     * @return
     */
    public static Dataset<Row> sparkExcelRead(SparkSession spark, String path){
        return sparkExcelRead(spark,path,0,null,null);
    }

    public static Dataset<Row> transVectorColumns(Dataset<Row> source, String[] inputNames, String outputName) {
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(inputNames)//new String[]{"score"}
                .setOutputCol(outputName);//"score_vector"
        Dataset<Row> vectorDataset = assembler.transform(source);
        return vectorDataset;
    }
    /**
     * 读取Excel首个Sheet,使用指定的ddl
     * @param spark
     * @param path
     * @return
     */
    public static Dataset<Row> sparkExcelRead(SparkSession spark, String path,String ddl){
        return sparkExcelRead(spark,path,0,null,ddl);
    }
    /**
     *
     * @param spark
     * @param path
     * @param sheetIndex
     * @param range
     * @param ddl 字段定义
     * @return
     */
    public static Dataset<Row> sparkExcelRead(SparkSession spark, String path, int sheetIndex, String range, String ddl) {
        String dataAddress = (range == null) ? sheetIndex + "!A1" : sheetIndex + "!" + range;
        DataFrameReader reader = spark.read().format(SPARK_EXCEL_FORMAT)
                .option("header", "false")
                .option("dataAddress", dataAddress)
                .option("inferSchema", "true");
        if (ddl != null){
            reader.schema(ddl);
        }
        Dataset<Row> dataset = reader.load(path);
        return dataset;
    }

    public static Dataset<Row> sparkReadFile(SparkSession spark, String path, String ddl) {
        if (path.toLowerCase().endsWith(".csv")){
            DataFrameReader reader = spark.read()
                    .option("header","true")
                    .option("inferSchema","true")
                    .option("ignoreLeadingWhiteSpace","true")
                    .option("ignoreTrailingWhiteSpace","true");
            if (ddl != null){
                reader.schema(ddl);
            }
            Dataset<Row> dataset = reader.csv(path);
            return  dataset;
        }
        return null;
    }


}
