package com.ibda.spark.statistics;

import org.apache.hadoop.shaded.org.apache.commons.beanutils.BeanUtils;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.sql.Row;

import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;

/**
 * 描述性统计结果
 */
public class DescriptiveStatistics {
    Long count;
    double[] min;
    double[] max;
    double[] mean;
    double[] normL1; //Σ(|x|)
    double[] normL2; //sqrt(Σx^2*w)
    double[] numNonZeros;
    double[] std;    //样本标准差
    double[] sum;
    double[] variance;//方差

    public static DescriptiveStatistics buildStatistics(Row row, String[] fields) {
        DescriptiveStatistics statistics = new DescriptiveStatistics();
        Arrays.stream(fields).forEach(target -> {
            Object value = row.getAs(target);
            try {
                BeanUtils.setProperty(statistics, target, value);
                if (value instanceof DenseVector) {
                    DenseVector vector = (DenseVector) value;
                    BeanUtils.setProperty(statistics, target, vector.toArray());
                } else {
                    BeanUtils.setProperty(statistics, target, value);
                }
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            } catch (InvocationTargetException e) {
                e.printStackTrace();
            }
        });
        return statistics;
    }

    public DescriptiveStatistics() {
        super();
    }

    public Long getCount() {
        return count;
    }

    public void setCount(Long count) {
        this.count = count;
    }

    public double[] getMin() {
        return min;
    }

    public void setMin(double[] min) {
        this.min = min;
    }

    public double[] getMax() {
        return max;
    }

    public void setMax(double[] max) {
        this.max = max;
    }

    public double[] getMean() {
        return mean;
    }

    public void setMean(double[] mean) {
        this.mean = mean;
    }

    public double[] getNormL1() {
        return normL1;
    }

    public void setNormL1(double[] normL1) {
        this.normL1 = normL1;
    }

    public double[] getNormL2() {
        return normL2;
    }

    public void setNormL2(double[] normL2) {
        this.normL2 = normL2;
    }

    public double[] getNumNonZeros() {
        return numNonZeros;
    }

    public void setNumNonZeros(double[] numNonZeros) {
        this.numNonZeros = numNonZeros;
    }

    public double[] getStd() {
        return std;
    }

    public void setStd(double[] std) {
        this.std = std;
    }

    public double[] getSum() {
        return sum;
    }

    public void setSum(double[] sum) {
        this.sum = sum;
    }

    public double[] getVariance() {
        return variance;
    }

    public void setVariance(double[] variance) {
        this.variance = variance;
    }

    @Override
    public String toString() {
        return "DescriptiveStatistics{" +
                "count=" + count +
                ", min=" + Arrays.toString(min) +
                ", max=" + Arrays.toString(max) +
                ", mean=" + Arrays.toString(mean) +
                ", normL1=" + Arrays.toString(normL1) +
                ", normL2=" + Arrays.toString(normL2) +
                ", numNonZeros=" + Arrays.toString(numNonZeros) +
                ", std=" + Arrays.toString(std) +
                ", sum=" + Arrays.toString(sum) +
                ", variance=" + Arrays.toString(variance) +
                '}';
    }
}
