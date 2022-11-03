package com.ibda.spark.regression;

import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.classification.ClassificationModel;
import org.apache.spark.ml.classification.OneVsRestModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema;
import org.junit.Before;
import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runners.MethodSorters;

import java.io.IOException;
import java.lang.reflect.ParameterizedType;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public abstract class SparkMLTest<E extends Estimator,M extends Model> {
    /**
     * 训练类
     */
    protected Class<E> estimatorClass = (Class<E>) ((ParameterizedType)this.getClass().getGenericSuperclass()).getActualTypeArguments()[0];
    /**
     * 模型类
     */
    protected Class<M> modelClass = (Class<M>) ((ParameterizedType)this.getClass().getGenericSuperclass()).getActualTypeArguments()[1];

    protected SparkML<E,M> sparkLearning = new SparkML<>(estimatorClass);

    protected Map<String, Object> trainingParams = new HashMap<String, Object>();

    protected PipelineModel pipelineModel = null;

    protected ModelColumns modelColumns = null;

    //数组：训练集、测试集、预测集
    protected Dataset<Row>[] datasets = new Dataset[3];

    protected void loadDataSet(String datafile,String format) {
        System.out.println("加载数据集:" + datafile );
        Dataset allData = sparkLearning.loadData(datafile,format);
        Dataset<Row> trainAndTest = allData.filter(modelColumns.getLabelCol() + " is not null");
        //划分训练集、测试集
        Dataset<Row>[] trainAndTests = trainAndTest.randomSplit(new double[]{0.7d, 0.3d},System.currentTimeMillis());
        datasets[0] = trainAndTests[0];
        datasets[1] = trainAndTests[1];
        datasets[2] = allData.filter(modelColumns.getLabelCol()  + " is null");
        System.out.println(String.format("记录总数：%1$s,训练集大小：%2$s,测试集大小：%3$s,预测集大小：%4$s",allData.count(),datasets[0].count(),datasets[1].count(),datasets[2].count()));
        pipelineModel = modelColumns.fit(datasets[0]);
    }

    /**
     * 准备数据、设置训练参数
     */
    @Before
    public void prepareData(){
        loadBinomialData();
        initTrainingParams();
    }

    @Test
    public void test01LearningEvaluatingPredicting() throws IOException {
        //训练
        System.out.println("训练模型：" + estimatorClass.getSimpleName() + "/" + modelClass.getSimpleName());
        SparkHyperModel<M> hyperModel = sparkLearning.fit(datasets[0], modelColumns,pipelineModel, trainingParams);
        System.out.println("训练模型结果及性能\n:" + hyperModel);
        //评估
        Map<String, Object> metrics = hyperModel.evaluate(datasets[1]);
        System.out.println("评估模型性能\n:" + metrics);
        Dataset<Row> tested = SparkHyperModel.getEvaluatePredictions(metrics);
        tested.show();
        //预测
        Dataset<Row> predicting = datasets[2];//carSales.filter("lnsales is null");
        System.out.println("预测数据集:" + predicting.count());
        predicting.show();
        Dataset<Row> predicted = hyperModel.predict(predicting);
        System.out.println("预测结果集:" + predicted.count());
        predicted.show();

        //预测单个数据
        Row[] rows = (Row[])predicted.select("regression_features_vector").head(20);
        Arrays.stream(rows).forEach(row->{
            GenericRowWithSchema gRow = (GenericRowWithSchema)row;
            Vector data = (Vector)gRow.values()[0];
            double label = sparkLearning.predict((PredictionModel)hyperModel.getModel(),data);
            System.out.println(data.toString() + ":" + label);
        });

        //模型读写
        System.out.println("测试读写模型---------------");
        String modelPath = FilePathUtil.getAbsolutePath("output/" + modelClass.getSimpleName() + ".model", true);
        System.out.println("保存及加载模型："  + modelPath);
        hyperModel.saveModel(modelPath);

        SparkHyperModel<M> loadedModel = SparkHyperModel.loadFromModelFile(modelPath, modelClass);
        Map<String, Object> metrics2 = loadedModel.evaluate(datasets[1], modelColumns,pipelineModel );
        System.out.println("评估存储模型性能\n:" + metrics2);

        System.out.println("使用存储模型进行预测\n:");
        Dataset<Row> predicted2 = loadedModel.predict(datasets[2],modelColumns,pipelineModel);
        predicted2.show();
    }
    @Test
    public void test02LearningMultinomialRegression() throws IOException {
        if (ClassificationModel.class.isAssignableFrom(modelClass) ||
                modelClass.equals(OneVsRestModel.class)){
            loadMultinomialData();
            test01LearningEvaluatingPredicting();
        }
        else{
            System.err.println("Not a Multinomial Regression-----------------");
        }
    }

    protected void loadBinomialData() {
        modelColumns = new ModelColumns(
                new String[]{"Age", "Income", "Credit_cards", "Education", "Car_loans"},
                null,
                "Credit_rating");
        loadDataSet(FilePathUtil.getAbsolutePath("data/credit_decision_tree.csv", false),"csv");
    }

    protected void loadMultinomialData() {
        //agecat,gender,marital,active,bfast，缺省加载，不做OneHotEncoder处理，逻辑回归需做处理
        modelColumns = new ModelColumns(
                new String[]{"agecat","gender","marital","active"},
                null,
                "preferbfast");
        loadDataSet(FilePathUtil.getAbsolutePath("data/cereal_multinomial.csv", false),"csv");
    }

    protected void initTrainingParams(){

    }
}
