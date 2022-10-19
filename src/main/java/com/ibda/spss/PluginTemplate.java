package com.ibda.spss;

import com.ibda.util.FilePathUtil;
import com.ibm.statistics.plugin.Case;
import com.ibm.statistics.plugin.CellValueFormat;
import com.ibm.statistics.plugin.StatsException;
import com.ibm.statistics.plugin.StatsUtil;

import java.io.File;
import java.util.Calendar;

/**
 * SPSS插件
 */
public abstract class PluginTemplate {
    protected void printStrings(String delimiter, String[] varNames) {
        System.out.println(String.join(delimiter,varNames));
    }

    protected void printCases(Case[] data) throws StatsException {
        for (Case onecase : data) {
            System.out.print("{");
            for (int i = 0; i < onecase.getCellNumber(); i++) {
                CellValueFormat format = onecase.getCellValueFormat(i);
                if (format == CellValueFormat.DOUBLE) {
                    Double numvar = onecase.getDoubleCellValue(i);
                    System.out.print(String.format("%1$.2f", numvar));
                } else if (format == CellValueFormat.DATE) {
                    Calendar datevar = onecase.getDateCellValue(i);
                    System.out.println(datevar.toString());
                } else {
                    String strvar = onecase.getStringCellValue(i);
                    System.out.print(strvar);
                }
                System.out.print(",");
            }
            System.out.println("}");
        }
    }

    /**
     * 保存文件，如有数据处理，需先调用datautil.release
     * @param absolutePath
     */
    protected void saveFileByCommand(String absolutePath) {
        execCommand("SAVE OUTFILE='" + absolutePath + "'.");
    }

    /**
     * 纯执行通用命令的插件类
     */
    public static class CommandsPlugin extends PluginTemplate {

        private final String[] syntaxCommands;

        public CommandsPlugin(String[] syntaxCommands) {
            this.syntaxCommands = syntaxCommands;
        }

        @Override
        public void batchStatistics() {
            execCommand(syntaxCommands);
        }
    }

    /**
     * Spss统计，具体统计工作由实现类的batchStatistics定义
     */
    public void pluginStats() {
        try {
            StatsUtil.start();
        } catch (StatsException e) {
            throw new RuntimeException("启动spss插件失败", e);
        }
        try {
            batchStatistics();
        } finally {
            try {
                StatsUtil.stop();
            } catch (StatsException e) {
                throw new RuntimeException("停止spss插件失败", e);
            }
        }
    }

    /**
     * 设置命令回显状态
     * @param echo
     */
    protected void commandEcho(boolean echo) {
        String command = String.format("SET PRINTBACK ON MPRINT %1$s .", echo ? "ON" : "OFF");
        execCommand(command);
    }

    /**
     * 执行单命令
     *
     * @param syntaxCommand
     */
    protected void execCommand(String syntaxCommand) {
        execCommand(new String[]{syntaxCommand});
    }

    /**
     * 执行命令组
     *
     * @param syntaxCommands
     */
    protected void execCommand(String[] syntaxCommands) {
        try {
            StatsUtil.submit(syntaxCommands);
        } catch (StatsException e) {
            throw new RuntimeException("执行spss命令失败：" + String.join(";", syntaxCommands), e);
        }
    }

    /**
     * 通过SPSS命令读取文件
     *
     * @param filePath
     * @param relativeToWorkingDirectory
     */
    protected void readFileByCommand(String filePath, boolean relativeToWorkingDirectory) {
        if (!new File(filePath).exists()) {
            filePath = FilePathUtil.getAbsolutePath(filePath, relativeToWorkingDirectory);
        }
        if (!new File(filePath).exists()) {
            throw new RuntimeException("文件不存在：" + filePath);
        }
        execCommand("GET FILE='" + filePath + "'.");
    }

    /**
     * 批量统计，由实现类具体实现
     */
    public abstract void batchStatistics();


}
