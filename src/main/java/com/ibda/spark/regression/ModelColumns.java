package com.ibda.spark.regression;

import java.util.Arrays;

public class ModelColumns {
    String[] noneCategoryFeatures; //特征属性列表
    String[] categoryFeatures;   //特征属性中的分类属性，需要进行OneHotEncoder处理
    String labelCol;         //观察结果列
    String predictCol;       //预测结果列
    String probabilityCol;   //或然列，逻辑回归时的概率值

    public ModelColumns(String[] noneCategoryFeatures, String[] categoryFeatures, String labelCol, String predictCol, String probabilityCol) {
        this.noneCategoryFeatures = noneCategoryFeatures;
        this.categoryFeatures = categoryFeatures;
        this.labelCol = labelCol;
        this.predictCol = predictCol;
        this.probabilityCol = probabilityCol;
    }

    public String[] getNoneCategoryFeatures() {
        return noneCategoryFeatures;
    }

    public void setNoneCategoryFeatures(String[] noneCategoryFeatures) {
        this.noneCategoryFeatures = noneCategoryFeatures;
    }

    public String[] getCategoryFeatures() {
        return categoryFeatures;
    }

    public void setCategoryFeatures(String[] categoryFeatures) {
        this.categoryFeatures = categoryFeatures;
    }

    public String getLabelCol() {
        return labelCol;
    }

    public void setLabelCol(String labelCol) {
        this.labelCol = labelCol;
    }

    public String getPredictCol() {
        return predictCol;
    }

    public void setPredictCol(String predictCol) {
        this.predictCol = predictCol;
    }

    public String getProbabilityCol() {
        return probabilityCol;
    }

    public void setProbabilityCol(String probabilityCol) {
        this.probabilityCol = probabilityCol;
    }

    @Override
    public String toString() {
        return "ModelColumns{" +
                "featureColumns=" + Arrays.toString(noneCategoryFeatures) +
                ", categoryCols=" + Arrays.toString(categoryFeatures) +
                ", labelCol='" + labelCol + '\'' +
                ", predictCol='" + predictCol + '\'' +
                ", probabilityCol='" + probabilityCol + '\'' +
                '}';
    }
}
