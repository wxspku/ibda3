package com.ibda.commonmath3.statistics;


import com.ibda.commonmath3.util.CommonMathAnalysis;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.inference.TestUtils;
import tech.tablesaw.api.Row;
import tech.tablesaw.api.Table;
import tech.tablesaw.columns.Column;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 假设校验
 */
public class HypothesisTest {


    /**
     * 使用KS检验确定数据是否正态分布
     *
     * @param data  待检验的数据
     * @param alpha 显著性水平
     * @return
     */
    public static boolean isNormalDistribution(double[] data, Double alpha) {
        return isNormalDistribution(data, alpha, null);
    }

    /**
     * 使用KS检验确定数据是否正态分布
     *
     * @param data     待检验的数据
     * @param alpha    显著性水平
     * @param descStat data对应的描述性统计对象，从性能优化考虑，无需二次计算
     * @return
     */
    public static boolean isNormalDistribution(double[] data, Double alpha, DescriptiveStatistics descStat) {
        if (descStat == null) {
            descStat = new DescriptiveStatistics(data);
        }
        final NormalDistribution unitNormal = new NormalDistribution(descStat.getMean(), descStat.getStandardDeviation());
        double pValue = TestUtils.kolmogorovSmirnovTest(unitNormal, data, true);
        System.out.println("偏度：" + descStat.getSkewness() + " | 峰度：" + descStat.getKurtosis() + " | KS test pValue=" + pValue);
        return pValue > alpha;
    }

    /**
     * 使用卡方检验检测是否属性相关的列联表，即二维表的行和列是否相关，检测行列相关性contingency table analysis
     *
     * @param counts
     * @param alpha
     * @return
     */
    public static boolean isDependentContingencyTable(long[][] counts, Double alpha) {
        double pValue = TestUtils.chiSquareTest(counts);
        System.out.println("chiSquare test pValue:" + pValue);
        return pValue < alpha;
    }

    /**
     * 多样本方差齐检验，使用莱文检验方差齐
     * 双样本方差齐直接使用TestUtils.homoscedasticTTest;也可以使用F右尾检验s2^2/s1^2，大的方差作为分母，显著性水平alpha/2
     * 莱文检验方差齐检验，基本过程如下：
     * 1：数据按列表示：r个组为r列，行为记录
     * 2：计算各组的均值 ：Xi均值
     * 3：计算各元素与本组均值的离差绝对值：|Xij-Xi均值|
     * 4：对转换后的数据进行单因素方差分析
     * @param categoryData
     * @param alpha
     * @return
     */
    public static boolean isVarianceEqual(List<double[]> categoryData, Double alpha){
        List<double[]> deviations = new ArrayList<>();
        for (double[] group : categoryData) {
            DescriptiveStatistics descStat = new DescriptiveStatistics(group);
            double[] deviation = Arrays.stream(group).map(item->Math.abs(item-descStat.getMean())).toArray();
            deviations.add(deviation);
        }
        //方差齐性检验
        boolean isVarianceEqual = TestUtils.oneWayAnovaTest(deviations,alpha);
        return !isVarianceEqual;
    }

    /**
     * 分组数据的均值是否相等
     *
     * @param data
     * @param alpha
     * @return
     */
    public static boolean isMeanEqual(List<double[]> data, Double alpha) {
        //正态性检验
        int i = 1;
        List<String> unNormalGroups = new ArrayList<>();
        List<double[]> deviations = new ArrayList<>();
        for (double[] group : data) {
            DescriptiveStatistics descStat = new DescriptiveStatistics(group);
            boolean isNormal = isNormalDistribution(group, alpha, descStat);
            if (!isNormal){
                unNormalGroups.add(String.valueOf(i-1));
            }
            double[] deviation = Arrays.stream(group).map(item->Math.abs(item-descStat.getMean())).toArray();
            deviations.add(deviation);
            System.out.println(String.format("第%1$d组：样本数 %2$d，均值 %3$.3f，标准差 %4$.3f,是否正态分布 %5$s",
                    i, group.length, descStat.getMean(), descStat.getStandardDeviation(),isNormal));
            i++;
        }
        //只需有一个非正态分布，则不能使用单因素方差检验

        if (unNormalGroups.size() > 1){
            throw new RuntimeException("如下组数据不满足正态分布，不支持单因素方差分析：" + String.join(",",unNormalGroups));
        }
        //方差齐性检验
        double pVarianceEqual = TestUtils.oneWayAnovaPValue(deviations);
        boolean isVarianceEqual = pVarianceEqual > alpha;
        //方差分析p值
        double pValue = 0d;
        if (isVarianceEqual ){
            pValue = TestUtils.oneWayAnovaPValue(data);
        }
        else{
            System.out.println("方差不齐，使用Welch检验......");
            pValue = new WelchOneWayAnova().anovaPValue(data);
        }

        System.out.println("one Way Anova pValue:" + pValue);
        return pValue > alpha;
    }

    public static void main(String[] args) {

        String path = CommonMathAnalysis.getClassRoot() + "/data/stattest.xlsx";
        System.out.println("正态检验------------");
        Table normalDistTest = CommonMathAnalysis.tablesawReadFile(path, 0);
        Column<Double> column = (Column<Double>) normalDistTest.column(1);
        double[] values = ArrayUtils.toPrimitive(column.asObjectArray());
        System.out.println("正态检验结果：" + isNormalDistribution(values, CommonMathAnalysis.SIGNIFICANCE_LEVEL_MEDIAN));

        System.out.println("列联表相关检验------------");
        Table table = CommonMathAnalysis.tablesawReadFile(path, 1);
        //首行数据为表头信息，不计入行号，因此行数据需从0行开始
        long[][] counts = new long[table.rowCount()][table.columnCount() - 1];
        for (int i = 0; i < table.rowCount(); i++) {
            Row row = table.row(i);
            for (int j = 1; j < table.columnCount(); j++) {
                counts[i][j - 1] = row.getInt(j);
            }
        }
        System.out.println("列联表相关检验结果：" + isDependentContingencyTable(counts, CommonMathAnalysis.SIGNIFICANCE_LEVEL_MEDIAN));

        System.out.println("单因素方差分析（多样本均值相等检验）------------");
        Table categoryData = CommonMathAnalysis.tablesawReadFile(path, 2);
        List<double[]> columns = new ArrayList<>();
        for (int i = 0; i < categoryData.columnCount(); i++) {
            Column<Double> categoryColumn = (Column<Double>) categoryData.column(i);
            columns.add(ArrayUtils.toPrimitive(categoryColumn.asObjectArray()));
        }
        System.out.println("多样本均值相等检验结果：" + isMeanEqual(columns, CommonMathAnalysis.SIGNIFICANCE_LEVEL_MEDIAN));
    }
}
