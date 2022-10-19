package com.ibda.spss;

import com.ibm.statistics.plugin.FormatSpec;
import com.ibm.statistics.plugin.PivotTable;
import com.ibm.statistics.plugin.StatsException;
import com.ibm.statistics.plugin.StatsUtil;

public class CustomOutputDemo extends PluginTemplate{
    @Override
    public void batchStatistics() {
        try {
            createPivotTable();
            createTextBlock();
        } catch (StatsException e) {
            e.printStackTrace();
        }
    }

    private void createPivotTable() throws StatsException {
        String[] command={
                "OMS SELECT TABLES",
                "/IF SUBTYPES=['pivotTableDemo']",
                "/DESTINATION FORMAT=HTML OUTFILE='D:/Source/Java/ibda2/output/pivottable.html'."
        };
        execCommand(command);
        Object[] rowLabels = new Object[] {"row1", "row2"};
        Object[] colLabels = new Object[] { "columnA", "columnB"};
        Object[][] cells = new Object[][] {{"1A","1B"}, {"2A","2B"}};
        String title = "Sample pivot table";
        String templateName = "pivotTableDemo";
        String outline = "";
        String caption = "";
        String rowDim = "Row dimension";
        String columnDim = "Column dimension";
        boolean hideRowDimTitle = false;
        boolean hideRowDimLabel = false;
        boolean hideColDimTitle = false;
        boolean hideColDimLabel = false;
        PivotTable table = new PivotTable(cells, rowLabels, colLabels,
                title, templateName, outline, caption, rowDim,
                columnDim, hideRowDimTitle, hideRowDimLabel,
                hideColDimTitle, hideColDimLabel, FormatSpec.COEFFICIENT);
        table.createSimplePivotTable();
        execCommand("OMSEND.");
    }

    private void createTextBlock() throws StatsException {
        String[] command={"OMS SELECT TEXTS",
                "/IF LABELS = ['Text block name']",
                "/DESTINATION FORMAT=HTML OUTFILE='D:/Source/Java/ibda2/output/textblock.htm'."
        };
        execCommand(command);
        StatsUtil.startProcedure("demo");
        StatsUtil.addTextBlock("Text block name", "The first line of text.");
        StatsUtil.addTextBlock("Text block name", "The second line of text.",1);
        StatsUtil.endProcedure();
        execCommand("OMSEND.");

    }

    public static void main(String[] args) {
        new CustomOutputDemo().pluginStats();
    }
}
