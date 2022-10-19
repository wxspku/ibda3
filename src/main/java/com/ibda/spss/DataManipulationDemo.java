package com.ibda.spss;

import com.ibda.util.FilePathUtil;
import com.ibm.statistics.plugin.*;

import java.util.Calendar;

public class DataManipulationDemo extends PluginTemplate {
    @Override
    public void batchStatistics() {
        try {
            commandEcho(true);
            readAll();
            filterColumns();
            doMissingValue();
            splitData();
            addVariables(FilePathUtil.getAbsolutePath("data/spss/mydata.sav",true));
            addMean(FilePathUtil.getAbsolutePath("data/spss/mymean.sav",true));
            appendCase(FilePathUtil.getAbsolutePath("data/spss/mycases.sav",true));
        } catch (StatsException e) {
            e.printStackTrace();
        }

    }
    private void appendCase(String absolutePath) throws StatsException{
        String[] command={"DATA LIST FREE /case (F) value (A1) date(ADATE).",
                "BEGIN DATA",
                "1 a 01/01/2012",
                "END DATA."};
        StatsUtil.submit(command);
        Case newcase = new Case(3);
        DataUtil datautil = new DataUtil();
        Calendar date = Calendar.getInstance();
        date.set(Calendar.YEAR, 2013);
        date.set(Calendar.MONTH, Calendar.JANUARY);
        date.set(Calendar.DAY_OF_MONTH, 1);
        newcase.setCellValue(0, 2);
        newcase.setCellValue(1, "b");
        newcase.setCellValue(2, date);
        datautil.appendCase(newcase);
        datautil.release();
        saveFileByCommand(absolutePath);
    }

    private void addVariables(String absolutePath) throws StatsException {
        String[] command={"DATA LIST FREE /case (A5).",
                "BEGIN DATA",
                "case1",
                "case2",
                "case3",
                "END DATA."};
        execCommand(command);
        Variable numVar = new Variable("numvar",0);
        Variable strVar = new Variable("strvar",1);
        Variable dateVar = new Variable("datevar",0);
        dateVar.setFormatType(VariableFormat.DATE);
        double[] numValues = new double[]{1.0,2.0,3.0};
        String[] strValues = new String[]{"a","b","c"};
        Calendar dateValue = Calendar.getInstance();
        dateValue.set(Calendar.YEAR, 2012);
        dateValue.set(Calendar.MONTH, Calendar.JANUARY);
        dateValue.set(Calendar.DAY_OF_MONTH, 1);
        Calendar[] dateValues = new Calendar[]{dateValue,dateValue,Calendar.getInstance()};
        DataUtil datautil = new DataUtil();
        datautil.addVariableWithValue(numVar, numValues, 0);
        datautil.addVariableWithValue(strVar, strValues, 0);
        datautil.addVariableWithValue(dateVar, dateValues, 0);
        datautil.release();
        saveFileByCommand(absolutePath);
    }

    private void addMean(String absolutePath) throws StatsException {
        String[] command={"DATA LIST FREE /var (F).",
                "BEGIN DATA",
                "40200",
                "21450",
                "21900",
                "END DATA."};
        StatsUtil.submit(command);
        Double total = 0.0;
        DataUtil datautil = new DataUtil();
        Case[] data = datautil.fetchCases(false, 0);
        for(Case onecase: data){
            total = total + onecase.getDoubleCellValue(0);
        }
        Double meanval = total/data.length;
        Variable mean = new Variable("mean",0);
        double[] meanVals = new double[data.length];
        for (int i=0;i<data.length;i++){
            meanVals[i]=meanval;
        }
        datautil.addVariableWithValue(mean, meanVals, 0);
        datautil.release();
        saveFileByCommand(absolutePath);
    }

    private void splitData() throws StatsException {
        int splitindex;
        System.out.println("Handling Data with Splits ---------------");
        String[] commands={"DATA LIST FREE /salary (F) jobcat (F).",
        "BEGIN DATA",
        "21450 1",
        "45000 1",
        "30000 2",
        "30750 2",
        "103750 3",
        "72500 3",
        "57000 3",
        "END DATA.",
        "SPLIT FILE BY jobcat."};
        execCommand(commands);
        DataUtil datautil4 = new DataUtil();
        splitindex = datautil4.getSplitIndex(0);
        while(splitindex!=-1){
            System.out.println("A new split begins at case: " + splitindex);
            splitindex = datautil4.getSplitIndex(splitindex);
        }
        datautil4.release();
    }

    private void doMissingValue() throws StatsException {
        System.out.println("自定义数据，空值处理---------------------");
        String[] command={"DATA LIST LIST (',') /numVar(f) stringVar(a4).",
                "BEGIN DATA",
                "1,a",
                ",b",
                "3,,",
                "9,d",
                "END DATA.",
                "MISSING VALUES numVar(9) stringVar(' ')."};
        execCommand(command);
        DataUtil datautil3 = new DataUtil();
        printStrings(",",datautil3.getVariableNames());
        Case[] data3 = datautil3.fetchCases(true, 0);
        printCases(data3);
        datautil3.release();
    }

    private void filterColumns() throws StatsException {
        System.out.println("读取部分列数据 /data/spss/Employee data.sav---------------------");
        readFileByCommand("/data/spss/Employee data.sav", true);
        //按指定字段排序及划分字段
        execCommand("SORT CASES BY educ (A).");
        execCommand("SPLIT FILE BY educ.");
        DataUtil datautil2 = new DataUtil();
        String[] varNames2 = new String[]{"id","educ","salary"};

        datautil2.setVariableFilter(varNames2);

        int splitindex = datautil2.getSplitIndex(0);
        while(splitindex!=-1){
            System.out.println("A new split BY educ begins at case: " + splitindex);
            splitindex = datautil2.getSplitIndex(splitindex);
        }
        printStrings(",",varNames2);
        Case[] data2 = datautil2.fetchCases(false, 0);
        printCases(data2);
        datautil2.release();
    }

    private void readAll() throws StatsException {
        System.out.println("读取全部数据 /data/spss/demo.sav---------------------");
        readFileByCommand("/data/spss/demo.sav", true);
        DataUtil datautil = new DataUtil();
        // setConvertDateTypes method specifies that values  with date or datetime formats will be converted to Java Calendar objects
        datautil.setConvertDateTypes(true);
        System.out.println("Variable Names");
        printStrings(",",datautil.getVariableNames());
        System.out.println("\n");
        //The first argument specifies that user-missing values will be treated as missing and thus converted to the Java null value.
        Case[] data = datautil.fetchCases(false, 0);
        printCases(data);
        datautil.release();
    }

    public static void main(String[] args) {
        DataManipulationDemo demo = new DataManipulationDemo();
        demo.pluginStats();
    }
}
