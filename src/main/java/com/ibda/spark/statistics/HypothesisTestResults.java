package com.ibda.spark.statistics;

import com.ibda.util.AnalysisConst;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.sql.Row;
import scala.collection.mutable.ArraySeq;

import java.util.Arrays;

/**
 * 假设检验结果，返回多个特征列检验结果
 */
public class HypothesisTestResults {
    double[] pValues; //伴随概率
    long[] degreesOfFreedom;  //自由度
    double[] statistics; //统计量

    /**
     *
     * @param resultRow
     * @return
     */
    public static HypothesisTestResults buildTestResults(Row resultRow) {
        HypothesisTestResults results = null;
        if (resultRow.get(0) instanceof Double){ //KS TEST返回pValue，statistics的单个元素
            results = new HypothesisTestResults(new double[]{(double)resultRow.get(0)},new double[]{(double)resultRow.get(1)});
        }
        else{
            double[] pValues = ((DenseVector) resultRow.get(0)).values();
            //可能是Integer或Long
            ArraySeq seq = (ArraySeq) resultRow.get(1);
            Long[] degreesOfFreedom = new Long[seq.size()];
            //可能是Integer或Long
            if (seq.head() instanceof Long){
                seq.copyToArray(degreesOfFreedom);
            }
            else{
                Integer[] df = new Integer[seq.size()];
                seq.copyToArray(df);
                for (int i=0 ;i<df.length;i++){
                    degreesOfFreedom[i] = new Long(df[i]);
                }
            }
            double[] statistics = ((DenseVector)resultRow.get(2)).values();
            results = new HypothesisTestResults(pValues, ArrayUtils.toPrimitive(degreesOfFreedom),statistics);
        }
        return results;
    }


    public HypothesisTestResults(double[] pValues, double[] statistics) {
        this.pValues = pValues;
        this.statistics = statistics;
    }

    public HypothesisTestResults(double[] pValues, long[] degreesOfFreedom, double[] statistics) {
        this.pValues = pValues;
        this.degreesOfFreedom = degreesOfFreedom;
        this.statistics = statistics;
    }

    public double[] getPValues() {
        return pValues;
    }

    public void setPValues(double[] pValues) {
        this.pValues = pValues;
    }

    public long[] getDegreesOfFreedom() {
        return degreesOfFreedom;
    }

    public void setDegreesOfFreedom(long[] degreesOfFreedom) {
        this.degreesOfFreedom = degreesOfFreedom;
    }

    public double[] getStatistics() {
        return statistics;
    }

    public void setStatistics(double[] statistics) {
        this.statistics = statistics;
    }

    @Override
    public String toString() {
        return "HypothesisTestResults{" +
                "pValues=" + Arrays.toString(pValues) +
                ", degreesOfFreedom=" + Arrays.toString(degreesOfFreedom) +
                ", statistics=" + Arrays.toString(statistics) +
                '}';
    }

    /**
     * 是否接受原假设，中值显著性 0.05
     *
     * @param featureIndex
     * @return
     */
    public boolean nullHypothesisAccepted(int featureIndex) {
        return nullHypothesisAccepted(featureIndex, AnalysisConst.SIGNIFICANCE_LEVEL_MEDIAN);
    }

    /**
     * 是否接受原假设
     *
     * @param featureIndex
     * @param alpha
     * @return
     */
    public boolean nullHypothesisAccepted(int featureIndex, double alpha) {
        return pValues[featureIndex] >= alpha;
    }

}
