package com.ibda.spss;

import com.ibm.statistics.plugin.OutputLanguages;
import com.ibm.statistics.plugin.StatsException;
import com.ibm.statistics.plugin.StatsUtil;

import java.io.IOException;

public class XmlOutputDemo extends PluginTemplate {
    @Override
    public void batchStatistics() {
        try {
            commandEcho(true);
            //保存XML文件，通过XML文件查看XPath
            saveXmlOutput();
            getByXPath();
        } catch (StatsException e) {
            e.printStackTrace();
        }
    }

    private void saveXmlOutput() throws StatsException {
        String[] command = {"GET FILE='D:/Source/Java/ibda2/data/spss/Employee data.sav'.",
                "OMS SELECT TABLES ",
                "/IF COMMANDS=['Descriptives'] SUBTYPES=['Descriptive Statistics'] ",
                "/DESTINATION FORMAT=OXML OUTFILE='D:/Source/Java/ibda2/output/descriptives_table.xml' ", //XMLWORKSPACE='desc_table2'
                "/TAG='desc_out'.",
                "DESCRIPTIVES VARIABLES=salary, salbegin, jobtime, prevexp ",
                "/STATISTICS=MEAN.",
                "OMSEND TAG='desc_out'."};
        StatsUtil.submit(command);
    }

    private void getByXPath() throws StatsException {
        readFileByCommand("/data/spss/Employee data.sav", true);
        String[] command = {"OMS SELECT TABLES ",
                "/IF COMMANDS=['Descriptives'] SUBTYPES=['Descriptive Statistics'] ",
                "/DESTINATION FORMAT=OXML XMLWORKSPACE='desc_table' ",
                "/TAG='desc_out'.",
                "DESCRIPTIVES VARIABLES=salary, salbegin, jobtime, prevexp ",
                "/STATISTICS=MEAN.",
                "OMSEND TAG='desc_out'."};

        StatsUtil.submit(command);
        String handle = "desc_table";
        String context = "/outputTree";

        //通过输出的XML文件结构，查看xpath
        String xpath = "//pivotTable[@subType='Descriptive Statistics']" +
                "/dimension[@axis='row']" +
                "/category[@varName='salary']" +
                "/dimension[@axis='column']" +
                "/category[@text='Mean']" +
                "/cell/@text";

        String[] result = StatsUtil.evaluateXPath(handle, context, xpath);
        printStrings(",", result);
        StatsUtil.deleteXPathHandle(handle);
    }

    public static void main(String[] args) throws StatsException, IOException {
        //new XmlOutputDemo().pluginStats();
        StatsUtil.start();
        StatsUtil.setOutputLanguage(OutputLanguages.ENGLISH);

        /*String result = StatsUtil.getXMLUTF16("desc_out");
        Writer out = new OutputStreamWriter(new FileOutputStream("output/descriptives_table.xml"));
        out.write(result);
        out.close();*/
        //StatsUtil.deleteXPathHandle("desc_table2");
        StatsUtil.stop();
    }
}
