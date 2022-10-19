package com.ibda.spss;

import com.ibm.statistics.plugin.DataUtil;
import com.ibm.statistics.plugin.StatsException;

public class DictInfoDemo extends PluginTemplate {
    @Override
    public void batchStatistics() {
        execCommand("SET PRINTBACK ON MPRINT ON.");
        readFileByCommand("/data/spss/Employee data.sav", true);
        DataUtil datautil = null;
        try {
            //dataUtil是Cursor的封装类,同时只有一个Cursor处于打开状态，需要cursor.close或datautil.release()才能执行后续命令
            datautil = new DataUtil();
            String[] varNames = datautil.getVariableNames();
            datautil.release();
            for (String name : varNames) {
                if (name.equalsIgnoreCase("gender")) {
                    String[] command = {"SORT CASES BY " + name + ".",
                            "SPLIT FILE LAYERED BY " + name + "."};
                    execCommand(command);
                }
            }

        } catch (StatsException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        DictInfoDemo dictInfo = new DictInfoDemo();
        dictInfo.pluginStats();
    }
}
