package com.ibda.util;

import java.io.Serializable;

public class AnalysisConst implements Serializable {
    /**
     * 显著性水平，低
     */
    public static final double SIGNIFICANCE_LEVEL_LOW = 0.1d;
    /**
     * 显著性水平，中，一般取中值
     */
    public static final double SIGNIFICANCE_LEVEL_MEDIAN = 0.05d;
    /**
     * 显著性水平，高，一般取中值
     */
    public static final double SIGNIFICANCE_LEVEL_HIGH = 0.01d;

    /**
     * 缺省离群点、缺失值样本删除比例阈值，低于该比例时，包含离群点、缺失值的样本允许直接删除
     */
    public static final double OUTLIER_SAMPLE_REMOVE_THRESHOLD = 0.05d;

    /**
     * 缺省缺失值变量删除比例阈值，高于该比例时，包含缺失值的字段允许直接删除
     */
    public static final double MISSING_VALUE_FEATURE_REMOVE_THRESHOLD = 0.6d;
    /**
     * 所有描述性统计指标
     */
    public static final DescriptiveTarget[] DESCRIPTIVE_TARGET_ALL = DescriptiveTarget.values();
    /**
     * 常用描述性统计指标
     */
    public static final DescriptiveTarget[] DESCRIPTIVE_TARGET_COMMON = new DescriptiveTarget[]{
            DescriptiveTarget.count,
            DescriptiveTarget.min,
            DescriptiveTarget.max,
            DescriptiveTarget.mean,
            DescriptiveTarget.std,
            DescriptiveTarget.sum
    };

    /**
     * 描述性统计指标，不要改变拼写及大小写，名称需与DescriptiveStatistics的字段一致
     */
    public enum DescriptiveTarget{
        count,
        min,
        max,
        mean,
        normL1, //Σ(|x|)
        normL2, //sqrt(Σx^2*w)
        numNonZeros,
        std,    //样本标准差
        sum,
        variance //方差
    }

    /**
     * 相关矩阵计算方法
     */
    public enum CorrelationMethod
    {
        pearson, //default
        spearman
    }

    /**
     * 离群点、缺失值处理方法
     */
    public enum OutlierProcessMethod {
        REMOVE,             //简单去除样本或变量，离群或缺失样本比例较低，或该变量的样本缺失值过多
        REPLACE_WITH_MEAN,  //使用均值、中位数、众数、离群临界值、最邻近正常值替换，仅用于数值类型
        REPLACE_WITH_MEDIAN,
        REPLACE_WITH_MODE,
        REPLACE_WITH_THRESHOLD,
        REPLACE_WITH_NEAREST
    }
}
