package com.ibda.spss;

import com.ibm.statistics.plugin.StatsException;
import com.ibm.statistics.plugin.StatsUtil;

public class PluginReadFile extends PluginTemplate {

    public static void main(String[] args) throws StatsException {
        PluginReadFile readFile = new PluginReadFile();
        readFile.pluginStats();

    }

    @Override
    public void batchStatistics() {
        execCommand("SET PRINTBACK ON MPRINT ON.");
        readFileByCommand("/data/spss/Employee data.sav",true);
        execCommand(new String[]{"OMS /SELECT TABLES ",
                "/IF COMMANDS = ['Descriptives' 'Frequencies'] ",
                "/DESTINATION FORMAT = HTML ",
                "IMAGES = NO OUTFILE = '/output/stats.html'.",
                "DESCRIPTIVES SALARY SALBEGIN.",
                "FREQUENCIES EDUC JOBCAT.",
                "OMSEND."});
        String varName = null;
        try {
            varName = StatsUtil.getVariableName(1);
            execCommand("FREQUENCIES /VARIABLES=" + varName + ".");
        } catch (StatsException e) {
            e.printStackTrace();
        }
    }
}
