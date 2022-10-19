package com.ibda.commonmath3.linear;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;

public class JamaCalculator {
    public static void main(String[] args) {
        double[][] data =new double[][]{
                {1,1/2d,4,3,3},
                {2,1,7,5,5},
                {1/4d,1/7d,1,1/2d,1/3d},
                {1/3d,1/5d,2,1,1},
                {1/3d,1/5d,3,1,1}
        };
        Matrix matrix = new Matrix(data);
        EigenvalueDecomposition ed = new EigenvalueDecomposition(matrix);
        System.out.println("real:" + ed.getRealEigenvalues());
        System.out.println("imagine:" + ed.getImagEigenvalues());
        System.out.println("D:" + ed.getD());
        System.out.println("V:" + ed.getV());
    }
}
