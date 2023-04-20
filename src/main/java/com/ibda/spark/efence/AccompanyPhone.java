package com.ibda.spark.efence;

import com.ibda.spark.regression.ModelColumns;
import com.ibda.spark.regression.SparkHyperModel;
import com.ibda.spark.regression.SparkML;
import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.fpm.FPGrowth;
import org.apache.spark.ml.fpm.FPGrowthModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class AccompanyPhone {
    static SparkML<FPGrowth, FPGrowthModel> fpm = null;

    public static void main(String[] args) throws IOException {
        System.out.println("参数列表：--------------------");
        for (String arg:args){
            System.out.print(arg + " ");
        }
        System.out.println("\n参数列表。--------------------");
        fpm = new SparkML<>(FPGrowth.class);
        //加载及转换数据
        Map<String,String> params = new HashMap<String,String>();
        params.put("url", "jdbc:mysql://rm-8vbn04qji2xh0k86r.mysql.zhangbei.rds.aliyuncs.com:3306/efence?useUnicode=true&autoReconnect=true&failOverReadOnly=false");
        params.put("dbtable", "card_acc_hd");
        params.put("user", "efence_user");
        params.put("password", "EFence!@#456");
        params.put("partitionColumn", "sort_no");
        params.put("numPartitions", "4");
        params.put("lowerBound", "1");
        params.put("upperBound", "13989");
        Dataset<Row> dataset = fpm.loadJdbc(params);
        //string转换为数组
        Dataset<Row> transfer = dataset.withColumn("card_list", functions.split(functions.col("card_list"), ",").cast("array<string>"));
        transfer.show();
        ModelColumns modelColumns = new ModelColumns(null,null,"id");
        modelColumns.setAdditionCols(new String[]{"card_count","card_list"});
        //transfer.
        //训练参数
        Map<String,Object> trainingParams = new HashMap<String,Object>();
        trainingParams.put("itemsCol","card_list");
        trainingParams.put("minSupport",0.05);
        trainingParams.put("minConfidence",0.5);
        trainingParams.put("numPartitions",4);
        System.out.println("训练模型：" + FPGrowth.class.getSimpleName() + "/" + FPGrowthModel.class.getSimpleName());
        SparkHyperModel<FPGrowthModel> hyperModel = fpm.fit(transfer, modelColumns,  trainingParams);

        //模型读写
        System.out.println("测试读写模型---------------");
        String modelPath = FilePathUtil.getAbsolutePath("output/" + FPGrowthModel.class.getSimpleName() + ".model", true);
        System.out.println("保存及加载模型：" + modelPath);
        hyperModel.saveModel(modelPath);

        FPGrowthModel model = hyperModel.getModel();
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
        model.transform(transfer).show();
    }
}
