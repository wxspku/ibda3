package com.ibda.commonmath3.regression;

import com.ibda.commonmath3.util.CommonMathAnalysis;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.distribution.FDistribution;
import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.regression.AbstractMultipleLinearRegression;
import org.apache.commons.math3.stat.regression.GLSMultipleLinearRegression;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import tech.tablesaw.api.Table;
import tech.tablesaw.columns.Column;

import java.util.Arrays;
import java.util.Random;


public class LinearRegression {
    /**
     * 求
     * @param x
     * @return
     */
    public static double[] estimateVIF(double[][] x){
        //自变量只有两列
        if (x[1].length == 2){
            PearsonsCorrelation correlation = new PearsonsCorrelation();
            RealMatrix matrix = correlation.computeCorrelationMatrix(x);
            double v = matrix.getData()[0][1];
            double[] result = {1/(1-Math.pow(v, 2)), 1/(1-Math.pow(v, 2))};
            return result;
        }
        throw new RuntimeException("Not implemented");
    }
    public static void main(String[] args) {
        String path = CommonMathAnalysis.getClassRoot() + "/data/stattest.xlsx";
        Table regressionData = CommonMathAnalysis.tablesawReadFile(path, 3);

        Column<Double> yColumn = (Column<Double>) regressionData.column(0);
        double[] y = ArrayUtils.toPrimitive(yColumn.asObjectArray());
        final int df1 = 2;
        int df2 = y.length - df1 - 1;
        double[][] x = new double[y.length][df1];
        for (int i = 1; i <= 2; i++) {
            Column<Double> xColumn = (Column<Double>) regressionData.column(i);
            double[] x_i = ArrayUtils.toPrimitive(xColumn.asObjectArray());
            for (int j = 0; j < xColumn.size(); j++) {
                x[j][i - 1] = x_i[j];
            }
        }
        System.out.print("检测多重共线性，VIF分别为：");
        printArray(estimateVIF(x));

        System.out.println("\nOLS 回归-----------------");
        OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression(1d);
        ols.newSampleData(y, x);
        double SSR = ols.calculateTotalSumOfSquares() - ols.calculateResidualSumOfSquares();
        System.out.println(String.format("RSquared %1$.3f, adjustedRSquared:%2$.3f, SST:%3$.3f, SSE:%4$.3f, SSR：%5$.3f",
                ols.calculateRSquared(),
                ols.calculateAdjustedRSquared(),
                ols.calculateTotalSumOfSquares(),
                ols.calculateResidualSumOfSquares(),
                SSR));
        double F = (SSR / df1) / (ols.calculateResidualSumOfSquares() / df2);
        FDistribution fdist = new FDistribution(null, df1, df2);
        double pValue = 1 - fdist.cumulativeProbability(F);
        System.out.println(String.format("整体显著性检验——F检验，右尾，自由度p、n-p-1,F:%1$.3f,p:%2$.5f,通过：%3$s",
                F, pValue, pValue < CommonMathAnalysis.SIGNIFICANCE_LEVEL_MEDIAN));
        /*RealMatrix matrix = ols.calculateHat();
        System.out.println(matrix);*/
        printRegression(ols);

        //omega的对角线表示每个残差项的权重,非0,权重相同时与OLS方法的结果相同，计算合理的omega具有专门的算法
        double[] residuals = ols.estimateResiduals();
        GLSMultipleLinearRegression gls = new GLSMultipleLinearRegression();
        System.out.println("\nGLS 回归，随机生成ω权重矩阵（n*n对角阵），回归参数根据ω变化，-----------------");
        double[][] omega = new double[y.length][y.length];
        for (int i = 0; i < y.length; i++) {
            omega[i][i] = new Random(System.currentTimeMillis() + i * 1000).nextDouble();
        }
        gls.newSampleData(y, x, omega);
        printGLSRegressionInfo(gls, ols.calculateTotalSumOfSquares(), df1, df2);
        printRegression(gls);


        //利用指数平滑法计算ω，https://www.statlect.com/fundamentals-of-statistics/generalized-least-squares
        //Ω[i,i] = α*Ω[i-1,i-1]+(1-α)*residuals[i]
        System.out.println("\nGLS 回归，指数平滑生成对角阵，回归参数根据ω变化，-----------------");
        double alpha = 0.4d;
        double[][] omega2 = new double[y.length][y.length];
        for (int i = 0; i < y.length; i++) {
            double prev = (i == 0) ? Math.abs(residuals[i]) : omega2[i - 1][i - 1];
            omega2[i][i] = alpha * prev + (1 - alpha) * Math.abs(residuals[i]);
        }
        gls.newSampleData(y, x, omega2);
        printGLSRegressionInfo(gls, ols.calculateTotalSumOfSquares(), df1, df2);
        printRegression(gls);

        System.out.println("\nGLS 回归，根据OLS权重生成对角阵，ols残差越大，权重ω越小，-----------------");
        double sumOfResiduals = Arrays.stream(residuals).map(residual -> Math.abs(residual)).sum();
        double[][] omega3 = new double[y.length][y.length];
        for (int i = 0; i < y.length; i++) {
            omega3[i][i] = sumOfResiduals - Math.abs(residuals[i]);
        }
        gls.newSampleData(y, x, omega3);
        printGLSRegressionInfo(gls, ols.calculateTotalSumOfSquares(), df1, df2);
        printRegression(gls);

    }

    private static void printRegression(AbstractMultipleLinearRegression regression) {
        System.out.println(String.format("ErrorVariance %1$.3f, RegressandVariance %2$.3f,RegressionStandardError %3$.3f",
                regression.estimateErrorVariance(),
                regression.estimateRegressandVariance(),
                regression.estimateRegressionStandardError()));

        System.out.print("Regression Parameters:");
        double[] parameters = regression.estimateRegressionParameters();
        printArray(parameters);
        System.out.print("Regression Parameters Standard Errors:");
        double[] parametersStandardErrors = regression.estimateRegressionParametersStandardErrors();
        printArray(parametersStandardErrors);
        System.out.print("Residuals:");
        printArray(regression.estimateResiduals());

        System.out.println("Regression Parameters Variance:");
        Arrays.stream(regression.estimateRegressionParametersVariance()).forEach(params -> printArray(params));
        System.out.println("参数显著性检验——T检验，双尾检验，自由度：n-p-1 ------- ");
        for (int i = 0; i < parameters.length; i++) {
            double t = parameters[i] / parametersStandardErrors[i];
            TDistribution tdist = new TDistribution(null, 5);
            double pValue = 2 * tdist.cumulativeProbability(-t);
            System.out.println(String.format("参数%1$d 显著性检验—T检验，t:%2$.3f,pValue:%3$.5f,通过情况：%4$s",
                    i, t, pValue, pValue < CommonMathAnalysis.SIGNIFICANCE_LEVEL_MEDIAN));
        }
    }

    private static void printArray(double[] params) {
        Arrays.stream(params).forEach(param -> System.out.print(String.format("  %1$.3f", param)));
        System.out.println();
    }

    private static void printGLSRegressionInfo(AbstractMultipleLinearRegression regression, double SST, int df1, int df2) {
        //计算SSE、SST、RSquared、adjustedRSquared
        double[] residuals = regression.estimateResiduals();
        double SSE = Arrays.stream(residuals).map(residual -> residual * residual).sum();
        double SSR = SST - SSE;
        double RSquared = SSR / SST;
        double adjustedRSquared = 1 - (1 - RSquared) * (df1 + df2) / df2;
        System.out.println(String.format("RSquared %1$.3f, adjustedRSquared:%2$.3f, SST:%3$.3f, SSE:%4$.3f, SSR：%5$.3f",
                RSquared,
                adjustedRSquared,
                SST,
                SSE,
                SSR));

        double F = (SSR / df1) / (SSE / df2);
        FDistribution fdist = new FDistribution(null, df1, df2);
        double pValue = 1 - fdist.cumulativeProbability(F);
        System.out.println(String.format("整体显著性检验——F检验，右尾，自由度p、n-p-1,F:%1$.3f,p:%2$.5f,通过：%3$s",
                F, pValue, pValue < CommonMathAnalysis.SIGNIFICANCE_LEVEL_MEDIAN));
    }
}
