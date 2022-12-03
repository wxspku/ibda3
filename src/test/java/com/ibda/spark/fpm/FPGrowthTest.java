package com.ibda.spark.fpm;

import com.ibda.spark.SparkMLTest;
import com.ibda.spark.regression.ModelColumns;
import com.ibda.spark.regression.SparkHyperModel;
import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.fpm.FPGrowth;
import org.apache.spark.ml.fpm.FPGrowthModel;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.RelationalGroupedDataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;
import org.junit.Test;

import java.io.IOException;

public class FPGrowthTest extends SparkMLTest<FPGrowth, FPGrowthModel> {

    //加载test01的测试数据
    protected void loadTest01Data() {
        modelColumns = new ModelColumns(null,null,"invoiceCode");
        modelColumns.setAdditionCols(new String[]{"stock_count","stock_list"});
        //加载mysql数据
        Dataset<Row> jdbcData = spark.read()
                .format("jdbc")
                .option("url", "jdbc:mysql://localhost:3308/employees?useUnicode=true&autoReconnect=true&failOverReadOnly=false")
                .option("dbtable", "retail")
                .option("user", "root")
                .option("password", "root")
                .load();
        jdbcData.show();
        jdbcData.createOrReplaceTempView("retail");
        Dataset<Row> rowDataset = spark.sql("SELECT invoiceCode, count(distinct stockCode) stock_count, array_agg(distinct stockCode) stock_list FROM retail GROUP BY invoiceCode");
        splitDataset(rowDataset);
        datasets[0].show();
    }

    protected void loadTest02Data() {

    }

    @Override
    public void initTrainingParams() {
        //minSupport需要根据数据调整，必须小于单条目的最大支持度，
        trainingParams.put("itemsCol","stock_list");
        trainingParams.put("minSupport",0.02);
        trainingParams.put("minConfidence",0.5);
    }

    @Override
    public void test01LearningEvaluatingPredicting() throws IOException {
        super.test01LearningEvaluatingPredicting();
        String modelPath = FilePathUtil.getAbsolutePath("output/" + modelClass.getSimpleName() + ".model", true);
        SparkHyperModel<FPGrowthModel> loadedModel = SparkHyperModel.loadFromModelFile(modelPath, modelClass);
        FPGrowthModel model = loadedModel.getModel();
        // Display frequent itemsets.
        Dataset<Row> freqItems = model.freqItemsets().orderBy("freq");
        System.out.println("频繁项集数：" + freqItems.count());
        freqItems.show();

        // Display generated association rules.
        Dataset<Row> associationRules = model.associationRules().sort(functions.col("lift").desc());
        System.out.println("关联规则数：" + associationRules.count());
        associationRules.show();
        // transform examines the input items against all the association rules and summarize the
        // consequents as prediction
        /*
        * transform: For each transaction in itemsCol, the transform method will compare its items against the antecedents
        * of each association rule. If the record contains all the antecedents of a specific association rule, the rule will
        * be considered as applicable and its consequents will be added to the prediction result. The transform method will
        * summarize the consequents from all the applicable rules as prediction.
        * The prediction column has the same data type as itemsCol and does not contain existing items in the itemsCol.*/
        model.transform(datasets[1]).show();
    }
    @Override
    public void test02MachineLearning() throws IOException {

    }
}
