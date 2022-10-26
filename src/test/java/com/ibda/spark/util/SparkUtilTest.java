package com.ibda.spark.util;

import com.ibda.util.FilePathUtil;
import org.apache.spark.sql.AnalysisException;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.apache.spark.sql.types.DataTypes.DoubleType;
import static org.apache.spark.sql.types.DataTypes.IntegerType;

public class SparkUtilTest {
    @Test
    public void testLoadData() throws AnalysisException {
        SparkSession spark = SparkUtil.buildSparkSession("sparkReadExcel");
        String path = FilePathUtil.getClassRoot() + "/data/stattest.xlsx";
        //add是创建新的StructType，所以不能用StructType.add
        //StructField初始化MetaData需使用Metadata.empty(),否则报错
        StructType schema = new StructType(new StructField[]{
                new StructField("id",IntegerType,false, Metadata.empty()),
                new StructField("score",DoubleType,false,Metadata.empty())
        });
        //"id INT NOT NULL,score DOUBLE NOT NULL"
        System.out.println(schema.toDDL());
        Dataset<Row> dataset = SparkUtil.loadExcel(spark, path, schema.toDDL(),0, "A2");
        dataset.show();
        //输出数据统计信息
        Dataset<Row> score1 = dataset.select("score").sort("score");
        score1.show();

        //使用sql查询数据
        dataset.createOrReplaceTempView("scores");
        Dataset<Row> score = spark.sql("select score from scores where score>=70");
        score.show();

        Map<String,String> options = new HashMap<>();
        //读取csv文件
        Dataset<Row> csv = SparkUtil.loadData(spark, FilePathUtil.getClassRoot() + "/data/anova_dvdplayers.csv", null, null, null);
        csv.show();

        //读取libsvm文件
        //option("numFeatures", "780").load("data/mllib/sample_libsvm_data.txt");
        options.clear();
        options.put("numFeatures", "780");
        Dataset<Row> libsvm = SparkUtil.loadData(spark, FilePathUtil.getWorkingDirectory() + "/data/mllib/sample_libsvm_data.txt", SparkUtil.LIBSVM_FORMAT, options, null);
        libsvm.show();

        //读取图像目录 .option("dropInvalid", true).load("data/mllib/images/origin/kittens")
        options.clear();
        options.put("dropInvalid", "true");
        Dataset<Row> images = SparkUtil.loadData(spark, FilePathUtil.getWorkingDirectory() + "data/mllib/images/origin/kittens", SparkUtil.IMAGE_FORMAT, options, null);
        images.show();
        // parquet file
        // output/bankloan_logistic.model/data/part-00000-beef3ccc-176c-4476-993e-141707928d86-c000.snappy.parquet
        Dataset<Row> models = SparkUtil.loadData(spark, FilePathUtil.getWorkingDirectory() + "output/bankloan_logistic.model/data/part-00000-beef3ccc-176c-4476-993e-141707928d86-c000.snappy.parquet", "parquet");
        models.show();
    }
}