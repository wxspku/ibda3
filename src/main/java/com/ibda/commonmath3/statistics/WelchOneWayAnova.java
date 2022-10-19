package com.ibda.commonmath3.statistics;

import org.apache.commons.math3.distribution.FDistribution;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.inference.OneWayAnova;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Welch单因素检验，用于检验方差不齐时的单因素方差分析
 * https://www.real-statistics.com/one-way-analysis-of-variance-anova/welchs-procedure/
 */
public class WelchOneWayAnova extends OneWayAnova  {
    /**
     * Welch检验
     */
    private class WelchTest{
        Collection<double[]> categoryData = null;
        List<DescriptiveStatistics> descStatisticsList = new ArrayList<>(); //分组权重
        List<Integer> numbers = new ArrayList<>(); //各分组数量
        List<Double> weights = new ArrayList<>(); //分组权重
        Double sumOfWeight = 0d; //权重和
        double weightedMean = 0d; //加权平均
        double dfbg = 0d; //df1, k-1
        double dfwg = 0d; //df2
        double msbg = 0d;
        double mswg = 0d;
        double fValue = 0d;
        /**
         * 构造函数
         * @param categoryData
         */
        public WelchTest(final Collection<double[]> categoryData) {
            this.categoryData = categoryData;
            dfbg = categoryData.size() - 1;
            //一次扫描，计算权重及加权均值数据
            for (double[] group : categoryData) {
                DescriptiveStatistics descStat = new DescriptiveStatistics(group);
                descStatisticsList.add(descStat);
                numbers.add(group.length);
                double weight = descStat.getN() / descStat.getVariance();
                weights.add(weight);
                sumOfWeight += weight;
                weightedMean += weight * descStat.getMean();
            }
            weightedMean = weightedMean / sumOfWeight;

            //二次扫描，计算msbg、mswg、dfwg、mswgPart
            double mswgPart = 0d;
            int k = numbers.size();
            for (int i = 0; i < k; i++) {
                msbg += weights.get(i) * Math.pow(descStatisticsList.get(i).getMean() - weightedMean, 2);
                mswgPart = mswgPart + (1 / (numbers.get(i) - 1d)) * Math.pow((1 - weights.get(i) / sumOfWeight), 2);
            }
            msbg = msbg / dfbg;
            mswg = 1 + 2 * (k - 2) * mswgPart / (k * k - 1);
            dfwg = (k * k - 1) / (3 * mswgPart);
            fValue = msbg / mswg;
        }
    }


    public WelchOneWayAnova() {
        super();
    }


    @Override
    public double anovaFValue(final Collection<double[]> categoryData){
        WelchTest welchTest = new WelchTest(categoryData);
        return welchTest.fValue;
    }

    @Override
    public double anovaPValue(final Collection<double[]> categoryData){
        WelchTest welchTest = new WelchTest(categoryData);
        final FDistribution fdist = new FDistribution(null, welchTest.dfbg, welchTest.dfwg);
        return 1.0 - fdist.cumulativeProbability(welchTest.fValue);
    }

    @Override
    public boolean anovaTest(final Collection<double[]> categoryData,
                             final double alpha){
        if ((alpha <= 0) || (alpha > 0.5)) {
            throw new OutOfRangeException(
                    LocalizedFormats.OUT_OF_BOUND_SIGNIFICANCE_LEVEL,
                    alpha, 0, 0.5);
        }
        return anovaPValue(categoryData) < alpha;
    }
}
