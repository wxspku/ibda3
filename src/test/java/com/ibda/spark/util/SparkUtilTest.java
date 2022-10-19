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

import static org.apache.spark.sql.types.DataTypes.DoubleType;
import static org.apache.spark.sql.types.DataTypes.IntegerType;
import static org.junit.Assert.*;

public class SparkUtilTest {
    @Test
    public static void main() throws AnalysisException {
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
        Dataset<Row> dataset = SparkUtil.sparkExcelRead(spark, path, 0, "A2",
                "id INT NOT NULL,score DOUBLE NOT NULL");
        dataset.show();
        //输出数据统计信息
        Dataset<Row> score1 = dataset.select("score").sort("score");
        score1.show();
        /*Row result1 = dataset.select(Summarizer.metrics("mean", "variance")
                        .summary(new Column("score")).as("summary"))
                .select("summary.mean", "summary.variance").first();
        System.out.println("mean = " + result1.<Vector>getAs(0).toString() +
                ", variance = " + result1.<Vector>getAs(1).toString());

        Row result2 = dataset.select(
                Summarizer.mean(new Column("score")),
                Summarizer.variance(new Column("score"))
        ).first();
        System.out.println("without weight: mean = " + result2.<Vector>getAs(0).toString() +
                ", variance = " + result2.<Vector>getAs(1).toString());*/

        //使用sql查询数据
        dataset.createTempView("scores");
        Dataset<Row> score = spark.sql("select score from scores where score>=70");
        score.show();
    }
}