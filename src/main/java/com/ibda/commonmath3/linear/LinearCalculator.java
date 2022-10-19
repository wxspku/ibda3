package com.ibda.commonmath3.linear;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

public class LinearCalculator {
    public static void main(String[] args) {
        //The eigen decomposition of matrix A is a set of two matrices: V and D such that A = V × D × V'. A, V and D are all m × m matrices
        //RealMatrix matrix = new Array2DRowRealMatrix(new double[][]{{-1,1,0},{-4,3,0},{1,0,2}});
        RealMatrix matrix = new Array2DRowRealMatrix(new double[][]{
                {1,1/2d,4,3,3},
                {2,1,7,5,5},
                {1/4d,1/7d,1,1/2d,1/3d},
                {1/3d,1/5d,2,1,1},
                {1/3d,1/5d,3,1,1}
        });

        System.out.println("特征值矩阵分解==========================");
        EigenDecomposition ed = new EigenDecomposition(matrix);
        for (int i=0 ;i<matrix.getColumnDimension();i++){
            System.out.println("--------------------: " + i);
            System.out.println(String.format("Eigen Value %1$.4f+%2$.4f*i:" ,ed.getRealEigenvalue(i), ed.getImagEigenvalue(i)));
            System.out.println("Eigen Vector :" + ed.getEigenvector(i));
            System.out.println("A*V:" +  matrix.multiply(new Array2DRowRealMatrix(ed.getEigenvector(i).toArray())));
            System.out.println("λ*V:" +  ed.getEigenvector(i).mapMultiply(ed.getRealEigenvalue(i)));
        }
        RealMatrix d = ed.getD();
        System.out.println("EigenValues:"  + d);
        System.out.println(ed.getDeterminant());
        RealMatrix v = ed.getV();
        System.out.println("Eigen Vector Matrix:" + v);
        RealMatrix vt = ed.getVT();
        System.out.println("Eigen Vector Matrix Transpose:" + vt);


        //The Singular Value Decomposition of matrix A is a set of three matrices: U, Σ and V such that A = U × Σ × VT
        System.out.println("奇异值矩阵分解==========================");
        SingularValueDecomposition svd = new SingularValueDecomposition(matrix);
        System.out.println("Singular Values:"  + svd.getS());
        System.out.println("U:"  + svd.getU());
        System.out.println("V Transpose:"  + svd.getVT());
    }
}
