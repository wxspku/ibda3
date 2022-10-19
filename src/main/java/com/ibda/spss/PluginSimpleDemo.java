package com.ibda.spss;

import com.ibm.statistics.plugin.StatsException;

public class PluginSimpleDemo {
    public static void main(String[] args) throws StatsException {
        String[] commands={"OMS",
                "/DESTINATION FORMAT=HTML OUTFILE='/output/demo.html'.",
                "DATA LIST FREE /salary (F).",
                "BEGIN DATA",
                "21450",
                "30000",
                "57000",
                "60000",
                "65000",
                "END DATA.",
                "DESCRIPTIVES salary.",
                "OMSEND."};
        PluginTemplate.CommandsPlugin plugin = new PluginTemplate.CommandsPlugin(commands);
        plugin.pluginStats();

    }
}
