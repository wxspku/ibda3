package com.ibda.spark.regression;

import java.util.Arrays;

public class ModelColumns {
    String[] featureColumns; //特征属性列表
    String[] categoryCols;   //特征属性中的分类属性，需要进行OneHotEncoder处理
    String labelCol;         //观察结果列
    String predictCol;       //预测结果列
    String probabilityCol;   //或然列，逻辑回归时的概率值

    public ModelColumns(String[] featureColumns, String[] categoryCols, String labelCol, String predictCol, String probabilityCol) {
        this.featureColumns = featureColumns;
        this.categoryCols = categoryCols;
        this.labelCol = labelCol;
        this.predictCol = predictCol;
        this.probabilityCol = probabilityCol;
    }

    public String[] getFeatureColumns() {
        return featureColumns;
    }

    public void setFeatureColumns(String[] featureColumns) {
        this.featureColumns = featureColumns;
    }

    public String[] getCategoryCols() {
        return categoryCols;
    }

    public void setCategoryCols(String[] categoryCols) {
        this.categoryCols = categoryCols;
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
                "featureColumns=" + Arrays.toString(featureColumns) +
                ", categoryCols=" + Arrays.toString(categoryCols) +
                ", labelCol='" + labelCol + '\'' +
                ", predictCol='" + predictCol + '\'' +
                ", probabilityCol='" + probabilityCol + '\'' +
                '}';
    }
}
