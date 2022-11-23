package com.ibda.spark.regression;

import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.regression.GeneralizedLinearRegression;
import org.apache.spark.ml.regression.GeneralizedLinearRegressionModel;
import org.junit.Test;

import java.io.IOException;

/**
 * 广义线性回归
 */
public class GLRegressionTest extends SparkRegressionTest<GeneralizedLinearRegression, GeneralizedLinearRegressionModel> {
    @Override
    public void test02MachineLearning() throws IOException {
        super.test02MachineLearning();
    }

    @Override
    protected void loadTest01Data() {
        System.out.println("加载car_sales_linear数据集，Gaussian回归,链接函数Identity--------");
        super.loadTest01Data();
        trainingParams.put("family","Gaussian");
        trainingParams.put("link","Identity");
    }

    @Override
    protected void loadTest02Data() {
        System.out.println("加载ships数据集，Poisson回归,链接函数Log--------");
        //type,construction,operation,months_service,log_months_service,damage_incidents
        modelColumns = new ModelColumns(
                null, //new String[]{"log_months_service"}
                new String[]{"type","construction","operation"},
                null,
                "damage_incidents");
        modelColumns.setAdditionCols(new String[]{"log_months_service"});
        loadDataSet(FilePathUtil.getAbsolutePath("data/ships_glm.csv", false), "csv");
        trainingParams.put("family","Poisson");
        trainingParams.put("link","Log");
        trainingParams.put("offsetCol","log_months_service");
    }



    protected void loadTest03Data() {
        //this.scaleByMinMax = true;
        System.out.println("加载car_insurance_claims数据集，Gamma回归,链接函数Inverse--------");
        //holderage,vehiclegroup,vehicleage,claimamt,nclaims
        modelColumns = new ModelColumns(
                null, //new String[]{"log_months_service"}
                new String[]{"holderage","vehiclegroup","vehicleage"},
                null,
                "claimamt");
        modelColumns.setWeightCol("nclaims");
        loadDataSet(FilePathUtil.getAbsolutePath("data/car_insurance_claim.csv", false), "csv");
        trainingParams.put("family","gamma");
        trainingParams.put("link","inverse");
        trainingParams.put("weightCol","nclaims");
    }

    @Test
    public void test03MachineLearning() throws IOException {
        this.loadTest03Data();
        test01LearningEvaluatingPredicting();
    }

    @Override
    public void initTrainingParams() {
        trainingParams.put("maxIter",1000);
        trainingParams.put("linkPredictionCol","linkPrediction");
        trainingParams.put("tol",1E-20);

        /*trainingParams.put("regParam",0.05d);
        trainingParams.put("aggregationDepth",10);*/
    }
}
