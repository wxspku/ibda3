package com.ibda.commonmath3.util;

import com.ibda.util.AnalysisConst;
import org.apache.commons.lang3.ArrayUtils;
import tech.tablesaw.api.Table;
import tech.tablesaw.columns.Column;
import tech.tablesaw.io.xlsx.XlsxReadOptions;
import tech.tablesaw.io.xlsx.XlsxReader;

public class CommonMathAnalysis extends AnalysisConst {

    /**
     *
     * @return
     */
    public static String getClassRoot(){
        return CommonMathAnalysis.class.getResource("/").getFile();
    }

    /**
     * 使用tablesaw读取文件数据
     * @param path
     * @param sheetIndex  xlsx文件的sheet索引号，从0开始
     * @return
     */
    public static Table tablesawReadFile(String path, Integer sheetIndex){
        Table data = null;
        sheetIndex = (sheetIndex == null)?0:sheetIndex;
        if (path.toLowerCase().endsWith(".xlsx") || path.toLowerCase().endsWith(".xls")){
            data = new XlsxReader().read(XlsxReadOptions.builder(path).sheetIndex(sheetIndex).build());
        }
        else{
            data = Table.read().file(path);
        }

        System.out.println(data.columnNames());
        System.out.println(data.shape());
        return data;
    }

    public static <T> T[] getColumn(Table table,int columnIndex){
        Column<T> column = (Column<T>) table.column(columnIndex);
        return (T[])ArrayUtils.toPrimitive(column.asObjectArray());
    }
}
